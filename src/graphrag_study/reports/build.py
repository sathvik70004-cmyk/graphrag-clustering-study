"""Phase 6: community reports (short summaries) for every graph x method x condition partition.

  uv run python -m graphrag_study.reports.build              # seed 0, finest level, local LLM

Report input = the community's member entities (names + descriptions where the graph has them)
plus up to 8 corpus sentences that mention members. The same recipe is used for every graph, so
the NLP graph (no descriptions) is summarised from text, as GraphRAG's fast mode does.

Deviation from GraphRAG (documented): GraphRAG writes long JSON reports (~1-2k tokens) with
findings and ratings; on a laptop GPU we write a short plain summary (<= ~120 words). Report
content depends only on the member set, so identical communities across conditions/methods are
summarised once (cache) — only communities changed by edge removal cost new LLM calls.

Writes runs/reports/{graph}/{method}/{condition}.parquet: community, members, report
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from graphrag_study.graph.load import load_tables
from graphrag_study.graph.provenance import entity_sentences
from graphrag_study.llm import chat

app = typer.Typer(add_completion=False)
RUNS = Path("runs")
SYSTEM = ("You summarise a group of related entities from Wikipedia for a question-answering "
          "system. Be factual and specific; use only the information given.")


def leaf_partition(graph: str, method: str, cond: str, seed: int = 0) -> dict[int, list[str]]:
    p = pd.read_parquet(RUNS / "partitions" / graph / method / f"{cond}_s{seed}.parquet")
    p = p[p.level == p.level.max()]
    return {int(c): sorted(g.node) for c, g in p.groupby("community")}


def report_prompt(members: list[str], desc: dict[str, str], es: dict, sents: dict) -> str:
    lines = [f"- {m}: {desc.get(m, '')[:200]}".rstrip(": ") for m in members[:25]]
    # sentences mentioning the most members first
    score: dict = {}
    for m in members:
        for k in es.get(m, ()):
            score[k] = score.get(k, 0) + 1
    top = sorted(score, key=lambda k: (-score[k], k))[:8]
    ev = [f"- ({t}) {sents[(t, i)]}" for t, i in top]
    return ("Entities in this group:\n" + "\n".join(lines) +
            ("\n\nEvidence sentences:\n" + "\n".join(ev) if ev else "") +
            "\n\nWrite a short summary (at most 120 words) of what this group is about, naming the "
            "key entities and the most important facts and relationships between them.")


@app.command()
def main(graphs: str = "llm_merged,nlp", methods: str = "leiden,louvain,kcore",
         conditions: str = "clean,random0,random1,critical", corpus: str = "proto") -> None:
    s = pd.read_parquet(f"data/processed/sentences_{corpus}.parquet")
    sents = {(t, i): x for t, i, x in zip(s.title, s.sent_id, s.text)}
    jobs = []
    for g in graphs.split(","):
        ents, _ = load_tables(g)
        desc = dict(zip(ents.title, ents.get("description", pd.Series([""] * len(ents))).fillna("")))
        es = entity_sentences(g, corpus)
        for m in methods.split(","):
            for c in conditions.split(","):
                part = {k: v for k, v in leaf_partition(g, m, c).items() if len(v) >= 2}
                jobs.append((g, m, c, part, desc, es))
    unique = {(j[0], tuple(v)) for j in jobs for v in j[3].values()}
    print(f"{len(jobs)} partitions, {len(unique)} unique communities to summarise")
    done: dict = {}
    for g, m, c, part, desc, es in tqdm(jobs, desc="partitions"):
        rows = []
        for cid, members in part.items():
            key = (g, tuple(members))
            if key not in done:
                done[key] = chat(report_prompt(members, desc, es, sents), SYSTEM, max_tokens=200)
            rows.append({"community": cid, "members": members, "report": done[key]})
        out = RUNS / "reports" / g / m / f"{c}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)


if __name__ == "__main__":
    app()
