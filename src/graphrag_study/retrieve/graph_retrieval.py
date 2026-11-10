"""Phase 7: graph retrieval methods. Each returns a Retrieval: ranked evidence sentences
(title, sent_id) within the same budget k, plus community reports for the answer context.

  local   (R1) GraphRAG-style local search: question -> top seed entities (embedding match)
               -> + 1-hop neighbours in the (perturbed) graph -> their sentences, ranked by
               similarity to the question. Context also gets the reports of the seed entities'
               communities. Clustering only affects which reports are added.
  routed  (R2) Community-routed: question -> top communities (embedding match on reports) ->
               sentences of their member entities, ranked by similarity. Clustering decides
               which evidence is reachable at all.
  global  (R3) GraphRAG-style global search (map-reduce over community reports). Prototype
               simplification: map over the 20 reports most similar to the question (2 batches
               of 10), not all reports. No sentence evidence -> no supporting-fact metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import igraph as ig
import numpy as np
import pandas as pd
import yaml

from graphrag_study.graph.load import load_graph, load_tables
from graphrag_study.graph.provenance import entity_sentences
from graphrag_study.llm import chat
from graphrag_study.perturb.edges import conditions
from graphrag_study.reports.build import leaf_partition
from graphrag_study.retrieve.baselines import embed

CFG = yaml.safe_load(open("configs/experiment.yaml"))
M = CFG["models"]


@dataclass
class Retrieval:
    sentences: list[tuple[str, int]] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)


def _emb(texts: list[str], name: str) -> np.ndarray:
    return embed(texts, M["embedding"], M["ollama_host"], name)


@lru_cache(maxsize=None)
def corpus_index(corpus: str = "proto"):
    s = pd.read_parquet(f"data/processed/sentences_{corpus}.parquet")
    keys = list(zip(s.title, s.sent_id))
    E = _emb((s.title + ". " + s.text).tolist(), f"sents_{corpus}")
    return keys, {k: i for i, k in enumerate(keys)}, E, dict(zip(keys, s.text))


@lru_cache(maxsize=None)
def graph_condition(graph: str, cond: str) -> ig.Graph:
    pc = CFG["perturbation"]
    return conditions(load_graph(graph), pc["fraction"], pc["random_repeats"])[cond]


@lru_cache(maxsize=None)
def entity_index(graph: str):
    ents, _ = load_tables(graph)
    desc = ents.get("description", pd.Series([""] * len(ents))).fillna("")
    E = _emb((ents.title + ": " + desc).tolist(), f"ent_{graph}")
    return ents.title.tolist(), E


@lru_cache(maxsize=None)
def community_index(graph: str, method: str, cond: str):
    rep = pd.read_parquet(f"runs/reports/{graph}/{method}/{cond}.parquet")
    E = _emb(rep.report.tolist(), f"rep_{graph}_{method}_{cond}")
    node2comm = {n: c for c, ms in leaf_partition(graph, method, cond).items() for n in ms}
    return rep, E, node2comm


def _rank_sentences(qv: np.ndarray, cands: set, corpus: str, k: int) -> list:
    keys, kidx, E, _ = corpus_index(corpus)
    c = [x for x in cands if x in kidx]
    return sorted(c, key=lambda x: -float(E[kidx[x]] @ qv))[:k]


def local(qv, graph, method, cond, k, corpus="proto", n_seeds=5, n_nbrs=20, n_reports=2):
    names, EE = entity_index(graph)
    h = graph_condition(graph, cond)
    seeds = [names[i] for i in np.argsort(-(EE @ qv))[:n_seeds]]
    ents = set(seeds)
    for s in seeds:
        v = h.vs.find(name=s)
        nb = sorted(h.incident(v), key=lambda e: -h.es[e]["weight"])[:n_nbrs]
        ents.update(h.vs[h.es[e].target if h.es[e].source == v.index else h.es[e].source]["name"]
                    for e in nb)
    es = entity_sentences(graph, corpus)
    sents = _rank_sentences(qv, {x for e in ents for x in es.get(e, ())}, corpus, k)
    rep, RE, n2c = community_index(graph, method, cond)
    comms = {n2c[s] for s in seeds if s in n2c}
    cand = rep[rep.community.isin(comms)]
    if len(cand):
        idx = cand.index[np.argsort(-(RE[cand.index] @ qv))][:n_reports]
        reports = rep.report.loc[idx].tolist()
    else:
        reports = []
    return Retrieval(sents, reports)


def routed(qv, graph, method, cond, k, corpus="proto", n_comms=3):
    rep, RE, _ = community_index(graph, method, cond)
    top = np.argsort(-(RE @ qv))[:n_comms]
    es = entity_sentences(graph, corpus)
    members = {m for i in top for m in rep.members.iloc[i]}
    sents = _rank_sentences(qv, {x for e in members for x in es.get(e, ())}, corpus, k)
    return Retrieval(sents, rep.report.iloc[top].tolist())


MAP_SYS = "You extract facts from community summaries that help answer a question."
REDUCE_SYS = "You answer questions from extracted facts."


def global_(qv, question, graph, method, cond, k, corpus="proto", n_reports=20, batch=10):
    rep, RE, _ = community_index(graph, method, cond)
    top = np.argsort(-(RE @ qv))[:n_reports]
    notes = []
    for b in range(0, len(top), batch):
        block = "\n\n".join(f"[{j + 1}] {rep.report.iloc[i]}" for j, i in enumerate(top[b:b + batch]))
        out = chat(f"Question: {question}\n\nCommunity summaries:\n{block}\n\nList the facts from "
                   "these summaries that are relevant to the question, one per line. If none are "
                   "relevant, reply NONE.", MAP_SYS, max_tokens=250)
        if out.strip().upper() != "NONE":
            notes.append(out)
    return Retrieval([], notes)


def retrieve(name, qv, question, graph, method, cond, k, corpus="proto") -> Retrieval:
    if name == "local":
        return local(qv, graph, method, cond, k, corpus)
    if name == "routed":
        return routed(qv, graph, method, cond, k, corpus)
    if name == "global":
        return global_(qv, question, graph, method, cond, k, corpus)
    raise ValueError(name)
