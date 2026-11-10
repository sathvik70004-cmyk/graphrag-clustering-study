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


@lru_cache(maxsize=None)
def entity_sentences(graph: str, corpus: str = "proto") -> dict[str, tuple[tuple[str, int], ...]]:
    """entity title -> sentence keys (title, sent_id) that mention it. If the entity IS the
    paragraph's subject (a name equals the paragraph title), every sentence of that paragraph
    counts: HotpotQA sentences usually refer to the subject by pronoun ("He was born ...")."""
    sents = pd.read_parquet(f"{PROC}/sentences_{corpus}.parquet")
    keys = list(zip(sents.title, sents.sent_id))
    texts = [" " + norm(x) + " " for x in sents.text]
    by_title: dict[str, list[int]] = {}
    for i, t in enumerate(sents.title):
        by_title.setdefault(norm(t), []).append(i)
    out = {}
    for ent, names in aliases(graph).items():
        pats = {norm(n) for n in names if len(norm(n)) > 2}
        hit = {i for i, txt in enumerate(texts) if any(f" {p} " in txt for p in pats)}
        for p in pats:
            hit.update(by_title.get(p, ()))
        out[ent] = tuple(keys[i] for i in sorted(hit))
    return out
