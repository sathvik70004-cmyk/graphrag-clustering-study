"""Phase 4: no-graph retrieval baselines over the sentence corpus (same budget as graph methods).

  uv run python -m graphrag_study.retrieve.baselines --corpus proto --methods bm25,dense

Writes runs/retrieval_baselines_{corpus}.csv (one row per question x method) and prints means.
Sentences are scored individually; each is prefixed with its paragraph title, as titles carry the
entity name that HotpotQA sentences often refer to only by pronoun.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
import typer
import yaml
from rank_bm25 import BM25Okapi

from graphrag_study.eval.sp_metrics import sp_scores

app = typer.Typer(add_completion=False)
PROC, RUNS = Path("data/processed"), Path("runs")
EMB_CACHE = RUNS / "cache"


def tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def load(corpus: str):
    sents = pd.read_parquet(PROC / f"sentences_{corpus}.parquet")
    sents["doc"] = sents.title + ". " + sents.text
    qs = [json.loads(l) for l in open(PROC / f"questions_{corpus}.jsonl", encoding="utf-8")]
    return sents, qs


def embed(texts: list[str], model: str, host: str, name: str) -> np.ndarray:
    """Embeddings cached on disk (keyed by model + corpus name + count)."""
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    path = EMB_CACHE / f"emb_{model.replace(':', '_')}_{name}_{len(texts)}.npy"
    if path.exists():
        return np.load(path)
    client, out = ollama.Client(host=host), []
    for i in range(0, len(texts), 64):
        out += client.embed(model=model, input=texts[i:i + 64])["embeddings"]
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    np.save(path, arr)
    return arr


@app.command()
def main(config: str = "configs/experiment.yaml", corpus: str = "proto",
         methods: str = "bm25,dense") -> None:
    cfg = yaml.safe_load(open(config))
    k = cfg["retrieval"]["top_k_sentences"]
    m = cfg["models"]
    sents, qs = load(corpus)
    keys = list(zip(sents.title, sents.sent_id))
    rankers = {}  # name -> f(question_index) -> sentence indices, best first
    if "bm25" in methods:
        bm = BM25Okapi([tok(d) for d in sents.doc])
        rankers["bm25"] = lambda i: np.argsort(-bm.get_scores(tok(qs[i]["question"])))
    if "dense" in methods:
        E = embed(sents.doc.tolist(), m["embedding"], m["ollama_host"], f"sents_{corpus}")
        Q = embed([q["question"] for q in qs], m["embedding"], m["ollama_host"], f"q_{corpus}")
        rankers["dense"] = lambda i: np.argsort(-(E @ Q[i]))
    rows = []
    for i, q in enumerate(qs):
        for name, rank in rankers.items():
            order = rank(i)
            retrieved = [keys[i] for i in order[: max(k, 20)]]
            row = {"qid": q["_id"], "type": q["type"], "method": name}
            for kk in sorted({2, 5, k}):
                row.update(sp_scores(retrieved, q["supporting_facts"], kk))
            rows.append(row)
    df = pd.DataFrame(rows)
    RUNS.mkdir(exist_ok=True)
    df.to_csv(RUNS / f"retrieval_baselines_{corpus}.csv", index=False)
    print(df.drop(columns=["qid"]).groupby(["method"]).mean(numeric_only=True).round(3).T.to_string())


if __name__ == "__main__":
    app()
