"""Demo: ask one question with a chosen retrieval method, clustering, graph and condition.

  uv run python -m graphrag_study.query "Who developed the jet fighters operated by the 41st
      Tactical Squadron of the Polish Air Force?" --method global --clustering leiden

Prototype scope: the corpus is the 20-question prototype (200 Wikipedia paragraphs); community
reports exist for graphs llm_merged/nlp, clusterings leiden/louvain/kcore and conditions
clean/random0/random1/critical. If the question is one of the dataset questions, the gold answer
and supporting facts are shown too.
"""
from __future__ import annotations

import json

import numpy as np
import typer
import yaml
from rank_bm25 import BM25Okapi

from graphrag_study.answer.run import answer, build_context
from graphrag_study.eval.qa_metrics import exact_match, f1
from graphrag_study.eval.sp_metrics import sp_scores
from graphrag_study.retrieve.baselines import embed, tok
from graphrag_study.retrieve.graph_retrieval import Retrieval, corpus_index, retrieve

app = typer.Typer(add_completion=False)


@app.command()
def main(question: str,
         method: str = typer.Option("routed", help="local | routed | global | dense | bm25"),
         clustering: str = typer.Option("leiden", help="leiden | louvain | kcore"),
         graph: str = typer.Option("llm_merged", help="llm_merged | nlp"),
         condition: str = typer.Option("clean", help="clean | random0 | random1 | critical"),
         corpus: str = "proto") -> None:
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    k, m = cfg["retrieval"]["top_k_sentences"], cfg["models"]
    keys, _, E, texts = corpus_index(corpus)
    qv = embed([question], m["embedding"], m["ollama_host"], f"adhoc_{abs(hash(question))}")[0]
    if method == "bm25":
        bm = BM25Okapi([tok(t + ". " + texts[(t, i)]) for t, i in keys])
        r = Retrieval([keys[i] for i in np.argsort(-bm.get_scores(tok(question)))[:k]])
    elif method == "dense":
        r = Retrieval([keys[i] for i in np.argsort(-(E @ qv))[:k]])
    else:
        r = retrieve(method, qv, question, graph, clustering, condition, k, corpus)
    ctx = build_context(r, texts)
    pred = answer(question, ctx)

    typer.secho(f"\n[{method} | {clustering} | {graph} | {condition}]", bold=True)
    for rep in r.reports[:3]:
        typer.echo(f"  report: {rep[:160]}...")
    for t, i in r.sentences:
        typer.echo(f"  ({t}, {i}) {texts[(t, i)][:110]}")
    typer.secho(f"\nAnswer: {pred}", fg="green", bold=True)

    for line in open(f"data/processed/questions_{corpus}.jsonl", encoding="utf-8"):
        q = json.loads(line)
        if q["question"].strip().lower() == question.strip().lower():
            s = sp_scores(r.sentences, q["supporting_facts"], k) if r.sentences else {}
            typer.echo(f"Gold:   {q['answer']}  (EM {exact_match(pred, q['answer']):.0f}, "
                       f"F1 {f1(pred, q['answer']):.2f}"
                       + (f", evidence recall@{k} {s[f'recall@{k}']:.2f})" if s else ")"))
            break


if __name__ == "__main__":
    app()
