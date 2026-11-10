"""Phase 5: cluster every graph condition with every method and compute graph-theory metrics.

  uv run python -m graphrag_study.eval.graph_metrics            # both graphs (llm, nlp)

Writes:
  runs/partitions/{graph}/{method}/{condition}_s{seed}.parquet   node, level, community
  runs/graph_metrics.csv                                         one row per graph/method/cond/seed/level
  runs/seed_noise.csv                                            ARI between seeds on the clean graph
  runs/gold_paths.csv                                            per graph/method/cond/seed/level

Gold-path metrics (multi-hop specific; gold facts used ONLY for evaluation):
  gold_connected    % of questions whose two gold paragraphs are linked: some entity mentioned in
                    paragraph 1 and some entity mentioned in paragraph 2 lie in the same connected
                    component of the (perturbed) graph -> a graph walk CAN reach hop 2
  gold_same_comm    % of questions where such a pair lies in the same COMMUNITY at that level ->
                    community-scoped retrieval can find both hops together

Metrics (per level; levels matched by index, coarse -> fine):
  lcc_frac          LCC size of the perturbed graph / original node count
  ari               ARI(clean partition, perturbed partition), same method & seed
  conn_orig_pct     % of the CLEAN non-singleton communities still internally connected in the
                    perturbed graph  (damage to the original communities)
  conn_new_pct      % of the perturbed graph's own non-singleton communities that are connected
                    (Leiden guarantees 100; Louvain does not)
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import igraph as ig
import pandas as pd
import typer
import yaml
from sklearn.metrics import adjusted_rand_score

from graphrag_study.cluster.methods import cluster
from graphrag_study.graph.load import load_graph
from graphrag_study.graph.provenance import paragraph_entities
from graphrag_study.perturb.edges import conditions

app = typer.Typer(add_completion=False)
RUNS = Path("runs")


def connected_pct(g: ig.Graph, membership: list[int]) -> float:
    groups: dict[int, list[int]] = {}
    for v, c in enumerate(membership):
        groups.setdefault(c, []).append(v)
    multi = [vs for vs in groups.values() if len(vs) > 1]
    if not multi:
        return 100.0
    ok = sum(g.induced_subgraph(vs).is_connected() for vs in multi)
    return round(100 * ok / len(multi), 2)


def lcc_frac(g: ig.Graph, n_orig: int) -> float:
    return round(max(g.connected_components().sizes()) / n_orig, 4) if g.vcount() else 0.0


def save_partition(levels, g: ig.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(g.vs[v]["name"], lvl, c) for lvl, m in enumerate(levels) for v, c in enumerate(m)]
    pd.DataFrame(rows, columns=["node", "level", "community"]).to_parquet(path, index=False)


def gold_pairs(qs: list[dict], pe: dict, idx: dict[str, int]) -> list[tuple[set, set]]:
    """Per question: (node ids mentioned in gold paragraph 1, ... in gold paragraph 2)."""
    out = []
    for q in qs:
        titles = list(dict.fromkeys(t for t, _ in q["supporting_facts"]))[:2]
        out.append(tuple({idx[e] for e in pe.get(t, ()) if e in idx} for t in titles))
    return out


def pct_linked(pairs, membership: list[int]) -> float:
    ok = sum(any(membership[a] == membership[b] for a in A for b in B) for A, B in pairs)
    return round(100 * ok / max(len(pairs), 1), 1)


@app.command()
def main(config: str = "configs/experiment.yaml", graphs: str = "llm,llm_merged,nlp",
         corpus: str = "proto") -> None:
    cfg = yaml.safe_load(open(config))
    cc, pc = cfg["clustering"], cfg["perturbation"]
    qs = [json.loads(l) for l in open(f"data/processed/questions_{corpus}.jsonl", encoding="utf-8")]
    rows, noise, gold = [], [], []
    for gname in graphs.split(","):
        g = load_graph(gname)
        n = g.vcount()
        pairs = gold_pairs(qs, paragraph_entities(gname, corpus), {v: i for i, v in enumerate(g.vs["name"])})
        conds = conditions(g, pc["fraction"], pc["random_repeats"])
        for cname, h in conds.items():
            gold.append({"graph": gname, "method": "-", "condition": cname, "seed": 0, "level": -1,
                         "gold_connected": pct_linked(pairs, h.connected_components().membership)})
        for method in cc["methods"]:
            seeds = [0] if method == "kcore" else cc["seeds"]
            clean_by_seed = {}
            for seed in seeds:
                clean = cluster(g, method, seed, cc.get("max_cluster_size", 10))
                clean_by_seed[seed] = clean
                for cname, h in conds.items():
                    lv = clean if cname == "clean" else cluster(h, method, seed,
                                                                cc.get("max_cluster_size", 10))
                    save_partition(lv, h, RUNS / "partitions" / gname / method / f"{cname}_s{seed}.parquet")
                    for level, memb in enumerate(lv):
                        gold.append({"graph": gname, "method": method, "condition": cname,
                                     "seed": seed, "level": level,
                                     "gold_same_comm": pct_linked(pairs, memb)})
                    for level in range(max(len(clean), len(lv))):
                        a = clean[min(level, len(clean) - 1)]
                        b = lv[min(level, len(lv) - 1)]
                        rows.append({
                            "graph": gname, "method": method, "condition": cname, "seed": seed,
                            "level": level, "n_communities": len(set(b)),
                            "lcc_frac": lcc_frac(h, n), "ari": round(adjusted_rand_score(a, b), 4),
                            "conn_orig_pct": connected_pct(h, a), "conn_new_pct": connected_pct(h, b),
                            "edges": h.ecount(),
                        })
            for s1, s2 in combinations(seeds, 2):
                for level in range(min(len(clean_by_seed[s1]), len(clean_by_seed[s2]))):
                    noise.append({"graph": gname, "method": method, "level": level,
                                  "seed_a": s1, "seed_b": s2,
                                  "ari": round(adjusted_rand_score(clean_by_seed[s1][level],
                                                                   clean_by_seed[s2][level]), 4)})
        print(f"[{gname}] {n} nodes, {g.ecount()} edges done")
    RUNS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "graph_metrics.csv", index=False)
    pd.DataFrame(noise).to_csv(RUNS / "seed_noise.csv", index=False)
    gd = pd.DataFrame(gold)
    gd.to_csv(RUNS / "gold_paths.csv", index=False)
    kind = gd.condition.str.replace(r"\d+$", "", regex=True)
    print("\ngold paragraphs connected in graph (% of questions):")
    print(gd[gd.method == "-"].assign(kind=kind).groupby(["graph", "kind"]).gold_connected.mean()
          .unstack().round(1).to_string())
    print("\ngold paragraphs in the SAME community, level 0 / finest level (% of questions):")
    gm = gd[gd.method != "-"].assign(kind=kind)
    fin = gm.groupby(["graph", "method", "condition", "seed"])["level"].transform("max")
    for name, part in [("level0", gm[gm.level == 0]), ("finest", gm[gm.level == fin])]:
        print(f"-- {name}")
        print(part.groupby(["graph", "method", "kind"]).gold_same_comm.mean().unstack().round(1).to_string())
    summary = (df[df.condition != "clean"]
               .assign(kind=lambda d: d.condition.str.replace(r"\d+$", "", regex=True))
               .groupby(["graph", "method", "kind", "level"])
               [["n_communities", "lcc_frac", "ari", "conn_orig_pct", "conn_new_pct"]].mean().round(3))
    print(summary.to_string())
    if noise:
        print("\nseed-noise ARI (clean graph):")
        print(pd.DataFrame(noise).groupby(["graph", "method", "level"]).ari.mean().round(3).to_string())


if __name__ == "__main__":
    app()
