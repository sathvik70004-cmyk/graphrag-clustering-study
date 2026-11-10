"""Entity -> paragraph provenance by mention: an entity belongs to every paragraph whose title or
text mentions one of its names (accent/case-insensitive, whole words). Used for gold-path metrics
and to map graph retrieval back to (title, sent_id) sentences."""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from graphrag_study.graph.load import aliases
from graphrag_study.graph.merge import norm

PROC = "data/processed"


@lru_cache(maxsize=None)
def paragraph_entities(graph: str, corpus: str = "proto") -> dict[str, frozenset[str]]:
    """paragraph title -> set of entity titles mentioned in it."""
    paras = pd.read_parquet(f"{PROC}/paragraphs_{corpus}.parquet")
    texts = {t: " " + norm(t + " " + x) + " " for t, x in zip(paras.title, paras.text)}
    out: dict[str, set[str]] = {t: set() for t in texts}
    for ent, names in aliases(graph).items():
        pats = {norm(n) for n in names if len(norm(n)) > 2}
        for t, txt in texts.items():
            if any(f" {p} " in txt for p in pats):
                out[t].add(ent)
    return {t: frozenset(v) for t, v in out.items()}
