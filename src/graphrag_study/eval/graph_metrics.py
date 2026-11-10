"""Phase 5: cluster every graph condition with every method and compute graph-theory metrics.

  uv run python -m graphrag_study.eval.graph_metrics            # both graphs (llm, nlp)

Writes:
  runs/partitions/{graph}/{method}/{condition}_s{seed}.parquet   node, level, community
  runs/graph_metrics.csv                                         one row per graph/method/cond/seed/level
  runs/seed_noise.csv                                            ARI between seeds on the clean graph

Metrics (per level; levels matched by index, coarse -> fine):
  lcc_frac          LCC size of the perturbed graph / original node count
  ari               ARI(clean partition, perturbed partition), same method & seed
  conn_orig_pct     % of the CLEAN non-singleton communities still internally connected in the
                    perturbed graph  (damage to the original communities)
  conn_new_pct      % of the perturbed graph's own non-singleton communities that are connected
                    (Leiden guarantees 100; Louvain does not)
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import igraph as ig
import pandas as pd
import typer
import yaml
from sklearn.metrics import adjusted_rand_score

from graphrag_study.cluster.methods import cluster
from graphrag_study.graph.load import GRAPHS, load_graph
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


@app.command()
def main(config: str = "configs/experiment.yaml", graphs: str = "llm,nlp") -> None:
    cfg = yaml.safe_load(open(config))
    cc, pc = cfg["clustering"], cfg["perturbation"]
    rows, noise = [], []
    for gname in graphs.split(","):
        g = load_graph(GRAPHS[gname])
        n = g.vcount()
        conds = conditions(g, pc["fraction"], pc["random_repeats"])
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
