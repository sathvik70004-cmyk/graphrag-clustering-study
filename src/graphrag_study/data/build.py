"""Phase 2: stratified question sample + deduplicated paragraph corpus with sentence provenance.

Outputs (data/processed/):
  questions_{full,pilot,proto}.jsonl   sampled questions (pilot/proto = prefixes of full)
  paragraphs_{...}.parquet             para_id, title, text, n_sents, doc_id
  sentences_{...}.parquet              title, sent_id, text  (gold-fact lookup key)
  stats.json
And graphrag_index/input/: the chosen corpus (default: proto), `group_size` paragraphs per .txt.
Grouping is by sorted title (p00000-p00004, ...), independent of which question a paragraph
belongs to, so gold paragraphs are not systematically co-located in one extraction call.

Gold supporting facts are NOT used to build anything; they are only checked for presence.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer
import yaml

app = typer.Typer(add_completion=False)
OUT = Path("data/processed")
GR_INPUT = Path("graphrag_index/input")
NLP_INPUT = Path("graphrag_index_nlp/input")  # NLP graph: 1 paragraph per file (co-occurrence is per file)


def stratified_sample(data: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n questions keeping the bridge/comparison ratio; result order is shuffled so any
    prefix (e.g. the pilot) is also approximately stratified."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in data:
        by_type[q["type"]].append(q)
    total = len(data)
    picked: list[dict] = []
    types = sorted(by_type)
    quotas = {t: round(n * len(by_type[t]) / total) for t in types}
    quotas[types[0]] += n - sum(quotas.values())  # fix rounding
    for t in types:
        picked += rng.sample(by_type[t], quotas[t])
    # interleave so prefixes keep the ratio
    rng.shuffle(picked)
    return picked


def build_corpus(questions: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    paras: dict[str, list[str]] = {}
    conflicts = 0
    for q in questions:
        for title, sents in q["context"]:
            if title in paras:
                conflicts += paras[title] != sents
                continue
            paras[title] = sents
    p_rows, s_rows = [], []
    for i, (title, sents) in enumerate(sorted(paras.items())):
        p_rows.append({"para_id": f"p{i:05d}", "title": title,
                       "text": "".join(sents).strip(), "n_sents": len(sents)})
        for j, s in enumerate(sents):
            s_rows.append({"title": title, "sent_id": j, "text": s.strip()})
    return pd.DataFrame(p_rows), pd.DataFrame(s_rows), conflicts


def check_gold(questions: list[dict], sents: pd.DataFrame) -> int:
    keys = set(zip(sents.title, sents.sent_id))
    return sum((t, i) not in keys for q in questions for t, i in q["supporting_facts"])


def export_graphrag_input(paras: pd.DataFrame, group_size: int) -> pd.DataFrame:
    """Write paragraphs to GraphRAG input, `group_size` per document (amortises the ~1.9k-token
    extraction prompt). Returns paras with a doc_id column for provenance."""
    if GR_INPUT.exists():
        shutil.rmtree(GR_INPUT)
    GR_INPUT.mkdir(parents=True)
    (GR_INPUT / ".gitkeep").touch()
    paras = paras.copy()
    paras["doc_id"] = [f"d{i // group_size:05d}" for i in range(len(paras))]
    for doc_id, grp in paras.groupby("doc_id", sort=True):
        # each paragraph keeps its Wikipedia title as a header line
        body = "\n\n".join(f"{r.title}\n{r.text}" for r in grp.itertuples())
        (GR_INPUT / f"{doc_id}.txt").write_text(body + "\n", encoding="utf-8")
    return paras


def export_nlp_input(paras: pd.DataFrame) -> None:
    if NLP_INPUT.exists():
        shutil.rmtree(NLP_INPUT)
    NLP_INPUT.mkdir(parents=True)
    (NLP_INPUT / ".gitkeep").touch()
    for r in paras.itertuples():
        (NLP_INPUT / f"{r.para_id}.txt").write_text(f"{r.title}\n{r.text}\n", encoding="utf-8")


def _safe(s: str) -> str:
    return re.sub(r"\s+", " ", s)


@app.command()
def main(config: str = "configs/experiment.yaml", export: str = "proto") -> None:
    cfg = yaml.safe_load(open(config))
    d = cfg["data"]
    data = json.load(open(d["hotpot_file"], encoding="utf-8"))
    full = stratified_sample(data, d["n_questions"], cfg["seed"])
    pilot = full[: d["n_pilot"]]
    proto = full[: d["n_proto"]]
    OUT.mkdir(parents=True, exist_ok=True)

    stats = {"source_questions": len(data)}
    for name, qs in [("full", full), ("pilot", pilot), ("proto", proto)]:
        with open(OUT / f"questions_{name}.jsonl", "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        paras, sents, conflicts = build_corpus(qs)
        if name == export:
            paras = export_graphrag_input(paras, d["group_size"])
            export_nlp_input(paras)
            stats["graphrag_input"] = {"corpus": name, "paragraphs": len(paras),
                                       "files": paras.doc_id.nunique(),
                                       "group_size": d["group_size"]}
        paras.to_parquet(OUT / f"paragraphs_{name}.parquet", index=False)
        sents.to_parquet(OUT / f"sentences_{name}.parquet", index=False)
        missing = check_gold(qs, sents)
        types = pd.Series([q["type"] for q in qs]).value_counts().to_dict()
        stats[name] = {
            "questions": len(qs), "types": types,
            "paragraphs_raw": sum(len(q["context"]) for q in qs),
            "paragraphs_dedup": len(paras), "sentences": len(sents),
            "title_conflicts": int(conflicts), "gold_facts_missing": int(missing),
            "approx_tokens": int(paras.text.str.split().str.len().sum() * 1.3),
        }

    json.dump(stats, open(OUT / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    app()
