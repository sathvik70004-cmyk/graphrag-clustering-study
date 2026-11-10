"""Entity merging (resolution) for the LLM-extracted graph.

GraphRAG links two paragraphs only when the LLM wrote the *exact same* entity name in both.
Naming drift ("KALĀKAUĀ" vs "KING DAVID KALĀKAUA") breaks multi-hop bridges. This step merges
entity nodes that are clearly the same thing, using conservative, deterministic rules (no LLM,
not tuned on QA results):

  R1 exact      names equal after normalisation (case, accents, punctuation, leading "THE")
  R2 surname   PERSON only: a one-word name A ("KALAKAUA") joins the full name B that ENDS with
                it ("KING DAVID KALAKAUA"), if A is the last word of exactly ONE other person
                (ambiguity guard) and B adds no identity token
  R3 spelling   near-identical spelling (ratio >= 0.9), SAME number of words (typos inside words
                only: GEORGE H. W. BUSS -> BUSH, but never GEORGE W. BUSH -> GEORGE H. W. BUSH),
                identical identity tokens, same type, descriptions similar

Identity tokens = anything with a digit (1st vs 41st, 1986), roman numerals (II), jr/sr.
A first version also merged any name contained in another name (e.g. SAXONY -> SAXONY-ANHALT,
GEORGE W. BUSH -> GEORGE H. W. BUSH); manual review of merges.csv showed most such merges were
wrong, so containment is restricted to the surname rule above.

Description similarity = cosine of local embeddings (embeddinggemma). Output: a new GraphRAG-
shaped output folder (entities/relationships parquet) + merges.csv listing every merge and its
rule, for manual inspection.

  uv run python -m graphrag_study.graph.merge            # graphrag_index/output -> output_merged
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

from graphrag_study.retrieve.baselines import embed

app = typer.Typer(add_completion=False)
ROMAN = re.compile(r"^(?=[ivxlcdm]+$)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"^the ", "", s)


def identity_tokens(tokens) -> set[str]:
    return {t for t in tokens
            if any(c.isdigit() for c in t) or ROMAN.match(t) or t in {"jr", "sr"}}


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


@app.command()
def main(index_dir: str = "graphrag_index", out_name: str = "output_merged",
         config: str = "configs/experiment.yaml", desc_sim: float = 0.5,
         spell_ratio: float = 0.9) -> None:
    m = yaml.safe_load(open(config))["models"]
    src = Path(index_dir) / "output"
    ents = pd.read_parquet(src / "entities.parquet").reset_index(drop=True)
    rels = pd.read_parquet(src / "relationships.parquet")
    n = len(ents)
    names = ents.title.tolist()
    nn = [norm(x) for x in names]
    toks = [set(x.split()) for x in nn]
    types = ents.type.fillna("").tolist()
    E = embed((ents.title + ": " + ents.description.fillna("")).tolist(), m["embedding"],
              m["ollama_host"], f"entdesc_{Path(index_dir).name}")
    sim = lambda a, b: float(E[a] @ E[b])  # noqa: E731

    uf, merges = UF(n), []

    def merge(a, b, rule):
        if uf.find(a) != uf.find(b):
            merges.append({"a": names[a], "b": names[b], "rule": rule, "type_a": types[a],
                           "type_b": types[b], "desc_sim": round(sim(a, b), 3)})
            uf.union(a, b)

    # R1: identical normalised names
    first: dict[str, int] = {}
    for i, x in enumerate(nn):
        if x in first:
            merge(i, first[x], "R1_exact")
        else:
            first.setdefault(x, i)

    # R2: person surname / mononym -> unique full name ending with it
    words = [x.split() for x in nn]
    for a in range(n):
        if types[a] != "PERSON" or len(words[a]) != 1 or len(nn[a]) < 4:
            continue
        hosts = {uf.find(b) for b in range(n) if b != a and types[b] == "PERSON"
                 and len(words[b]) > 1 and words[b][-1] == words[a][0]}
        if len(hosts) != 1:
            continue
        b = next(b for b in range(n) if uf.find(b) in hosts and len(words[b]) > 1
                 and words[b][-1] == words[a][0])
        if identity_tokens(toks[b]) or sim(a, b) < desc_sim:
            continue
        merge(a, b, "R2_surname")

    # R3: near-identical spelling
    for a in range(n):
        for b in range(a + 1, n):
            if uf.find(a) == uf.find(b) or types[a] != types[b]:
                continue
            if min(len(nn[a]), len(nn[b])) < 6 or len(nn[a].split()) != len(nn[b].split()):
                continue
            if identity_tokens(toks[a]) != identity_tokens(toks[b]):
                continue
            if SequenceMatcher(None, nn[a], nn[b]).ratio() >= spell_ratio and sim(a, b) >= desc_sim:
                merge(a, b, "R3_spelling")

    # canonical name per group: most frequent, then longest
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    canon = {}
    for members in groups.values():
        c = max(members, key=lambda i: (ents.frequency.iloc[i], len(names[i])))
        for i in members:
            canon[names[i]] = names[c]

    rows = []
    for members in groups.values():
        c = canon[names[members[0]]]
        sub = ents.iloc[members]
        rows.append({
            "title": c,
            "type": ents.type.iloc[next(i for i in members if names[i] == c)],
            "description": " ".join(dict.fromkeys(sub.description.fillna(""))).strip(),
            "aliases": sorted({names[i] for i in members}),
            "frequency": int(sub.frequency.sum()),
            "text_unit_ids": sorted({t for ids in sub.text_unit_ids for t in ids}),
        })
    new_ents = pd.DataFrame(rows)
    new_rels = rels.copy()
    new_rels["source"] = new_rels.source.map(lambda x: canon.get(x, x))
    new_rels["target"] = new_rels.target.map(lambda x: canon.get(x, x))
    new_rels = new_rels[new_rels.source != new_rels.target]

    out = Path(index_dir) / out_name
    out.mkdir(exist_ok=True)
    new_ents.to_parquet(out / "entities.parquet", index=False)
    new_rels.to_parquet(out / "relationships.parquet", index=False)
    mdf = pd.DataFrame(merges)
    mdf.to_csv(out / "merges.csv", index=False)
    print(f"{n} entities -> {len(new_ents)} after {len(mdf)} merges "
          f"({mdf.rule.value_counts().to_dict() if len(mdf) else {}})")


if __name__ == "__main__":
    app()
