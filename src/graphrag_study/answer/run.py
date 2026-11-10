"""Phase 8: answer every question under every configuration and score it.

  uv run python -m graphrag_study.answer.run                      # full prototype grid
  uv run python -m graphrag_study.answer.run --limit 2 --graphs nlp --methods leiden  # smoke test

Grid: baselines (bm25, dense) once; graph retrieval local/routed for each graph x clustering x
condition (seed 0); global only on the clean graph. Same answer model, prompt and evidence budget
(k sentences + <= 3 reports) everywhere. Writes runs/qa_{corpus}.csv (one row per question x config).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
import typer
import yaml
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from graphrag_study.eval.qa_metrics import exact_match, f1
from graphrag_study.eval.sp_metrics import sp_scores
from graphrag_study.llm import chat
from graphrag_study.retrieve.baselines import embed, tok
from graphrag_study.retrieve.graph_retrieval import Retrieval, corpus_index, retrieve

app = typer.Typer(add_completion=False)
ENC = tiktoken.get_encoding("o200k_base")
SYSTEM = ("You answer multi-hop questions using only the provided context. Reply with the "
          "shortest possible answer: a name, number, date, short phrase, or yes/no. "
          "No explanation.")


def build_context(r: Retrieval, texts: dict) -> str:
    parts = []
    if r.reports:
        parts.append("Community summaries:\n" + "\n\n".join(r.reports[:3]))
    if r.sentences:
        parts.append("Evidence:\n" + "\n".join(f"- ({t}) {texts[(t, i)]}" for t, i in r.sentences))
    return "\n\n".join(parts) or "(no context)"


def answer(question: str, context: str) -> str:
    out = chat(f"Context:\n{context}\n\nQuestion: {question}\nAnswer:", SYSTEM, max_tokens=40)
    return out.splitlines()[0].strip() if out else ""


@app.command()
def main(corpus: str = "proto", graphs: str = "llm_merged,nlp",
         methods: str = "leiden,louvain,kcore", conditions: str = "clean,random0,random1,critical",
         retrievals: str = "local,routed,global", limit: int = 0) -> None:
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    k, m = cfg["retrieval"]["top_k_sentences"], cfg["models"]
    all_qs = [json.loads(l) for l in open(f"data/processed/questions_{corpus}.jsonl", encoding="utf-8")]
    Q = embed([q["question"] for q in all_qs], m["embedding"], m["ollama_host"], f"q_{corpus}")
    qs = all_qs[:limit] if limit else all_qs
    keys, _, E, texts = corpus_index(corpus)
    bm = BM25Okapi([tok(t + ". " + texts[(t, i)]) for t, i in keys])

    configs = [("-", "-", "-", "bm25"), ("-", "-", "-", "dense")]
    for g in graphs.split(","):
        for meth in methods.split(","):
            for c in conditions.split(","):
                for r in retrievals.split(","):
                    if r == "global" and c != "clean":
                        continue
                    configs.append((g, meth, c, r))

    rows = []
    for g, meth, c, r in tqdm(configs, desc="configs"):
        for qi, q in enumerate(qs):
            qv = Q[qi]
            t0 = time.time()
            if r == "bm25":
                ret = Retrieval([keys[i] for i in np.argsort(-bm.get_scores(tok(q["question"])))[:k]])
            elif r == "dense":
                ret = Retrieval([keys[i] for i in np.argsort(-(E @ qv))[:k]])
            else:
                ret = retrieve(r, qv, q["question"], g, meth, c, k, corpus)
            ctx = build_context(ret, texts)
            pred = answer(q["question"], ctx)
            row = {"graph": g, "method": meth, "condition": c, "retrieval": r, "qid": q["_id"],
                   "type": q["type"], "gold": q["answer"], "pred": pred,
                   "em": exact_match(pred, q["answer"]), "f1": f1(pred, q["answer"]),
                   "ctx_tokens": len(ENC.encode(ctx)), "n_sents": len(ret.sentences),
                   "n_reports": len(ret.reports), "seconds": round(time.time() - t0, 2)}
            if r != "global":
                row.update(sp_scores(ret.sentences, q["supporting_facts"], k))
            rows.append(row)
    out = Path("runs") / f"qa_{corpus}{'_limit' + str(limit) if limit else ''}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    cols = ["em", "f1", f"recall@{k}", f"both_gold_paras@{k}", "ctx_tokens"]
    print(df.groupby(["retrieval", "graph", "method", "condition"], dropna=False)[cols]
          .mean().round(3).to_string())


if __name__ == "__main__":
    app()
