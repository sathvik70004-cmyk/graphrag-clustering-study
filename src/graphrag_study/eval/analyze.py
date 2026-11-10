"""Phase 9: turn the prototype runs into a readable report (results/prototype_report.md).

  uv run python -m graphrag_study.eval.analyze
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path("results")


def md(df: pd.DataFrame) -> str:
    return df.to_markdown()


def main() -> None:
    qa = pd.read_csv(R / "qa_proto.csv")
    qa["kind"] = qa.condition.str.replace(r"\d+$", "", regex=True)
    gm = pd.read_csv(R / "graph_metrics.csv")
    gp = pd.read_csv(R / "gold_paths.csv")
    noise = pd.read_csv(R / "seed_noise.csv")
    n_q = qa.qid.nunique()
    out = [f"# Prototype results ({n_q} HotpotQA questions, 200 paragraphs)\n",
           f"> One question = {100 / n_q:.0f} points of exact match. Differences of 1-2 questions "
           "are noise at this size; the prototype shows the pipeline works, not which method wins.\n"]

    # 1. answers + evidence, clean graph
    clean = qa[qa.kind.isin(["clean", "-"])]
    t = (clean.groupby(["retrieval", "graph", "method"])[["em", "f1", "recall@10", "ctx_tokens"]]
         .mean().round(2))
    out += ["## 1. Answer quality and evidence (clean graph)\n", md(t), ""]

    # 2. effect of edge removal
    pert = qa[qa.retrieval.isin(["local", "routed"])]
    t = (pert.groupby(["retrieval", "graph", "method", "kind"])[["em", "f1", "recall@10"]]
         .mean().round(2).unstack("kind"))
    out += ["## 2. Effect of edge removal (local and routed retrieval)\n", md(t), ""]

    # 3. graph stability
    g = gm[gm.condition != "clean"].assign(kind=lambda d: d.condition.str.replace(r"\d+$", "", regex=True))
    t = (g[g.level == 0].groupby(["graph", "method", "kind"])
         [["lcc_frac", "ari", "conn_orig_pct", "n_communities"]].mean().round(3).unstack("kind"))
    nz = noise[noise.level == 0].groupby(["graph", "method"]).ari.mean().round(3).rename("seed_noise_ari")
    out += ["## 3. Graph stability after edge removal (level 0)\n",
            "ARI = agreement with the clean graph's communities (1 = identical). "
            "Seed-noise ARI = agreement between two runs on the SAME clean graph.\n",
            md(t), "", md(nz.to_frame()), ""]

    gc = gp[gp.method == "-"].assign(kind=lambda d: d.condition.str.replace(r"\d+$", "", regex=True))
    t = gc.groupby(["graph", "kind"]).gold_connected.mean().round(1).unstack()
    out += ["## 4. Gold-path connectivity (% of questions whose two gold paragraphs are linked)\n",
            md(t), "",
            "Unchanged by edge removal: the two gold paragraphs are almost always joined by one "
            "entity named in both, not by a graph path.\n"]

    # 5. failure examples
    out += ["## 5. Examples\n"]
    loc = qa[(qa.retrieval == "local") & (qa.kind == "clean")]
    wrong = loc[(loc["recall@10"] == 1.0) & (loc.f1 == 0)].drop_duplicates("qid").head(3)
    out += ["### Evidence fully retrieved, answer still wrong (answer-model limit)\n"]
    out += [f"- gold **{r.gold}** / predicted *{r.pred}* ({r.graph}, {r.method})" for r in wrong.itertuples()]
    base = qa[qa.retrieval.isin(["local", "routed"]) & (qa.kind == "clean")].set_index(
        ["retrieval", "graph", "method", "qid"])
    crit = qa[qa.retrieval.isin(["local", "routed"]) & (qa.kind == "critical")].set_index(
        ["retrieval", "graph", "method", "qid"])
    j = base[["gold", "pred", "f1"]].join(
        crit[["pred", "f1"]].rename(columns={"pred": "pred_crit", "f1": "f1_crit"})).dropna()
    flips = j[(j.f1 > 0.5) & (j.f1_crit == 0)].head(4)
    out += ["", "### Correct on the clean graph, wrong after critical-edge removal\n"]
    out += [f"- {i[0]}/{i[1]}/{i[2]}: gold **{r.gold}**: clean *{r.pred}*, critical *{r.pred_crit}*"
            for i, r in flips.iterrows()] or ["- none"]
    glob = qa[qa.retrieval == "global"]
    best = glob[glob.em == 1].drop_duplicates("qid")
    only_glob = set(best.qid) - set(qa[(qa.retrieval == "dense") & (qa.em == 1)].qid)
    out += ["", f"### Global search answered {len(only_glob)} question(s) that dense retrieval missed\n"]
    out += [f"- gold **{r.gold}** ({r.graph}, {r.method})"
            for r in best[best.qid.isin(only_glob)].drop_duplicates("qid").itertuples()]

    (R / "prototype_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
