"""Hierarchical clustering methods with one common output format.

Every method returns a list of levels, coarse -> fine. Each level is a partition of ALL nodes:
a list `membership[i] = community id of node i`. Finer levels refine coarser ones, so
"same level" comparisons (ARI) are well defined.

- leiden / louvain: GraphRAG-style hierarchical scheme (as graspologic's hierarchical_leiden):
  level 0 = clustering of the whole graph; level l+1 = re-cluster every community that is larger
  than `max_cluster_size`, leaving smaller ones unchanged. Stops when nothing is too large or
  `max_levels` is reached.
- kcore: core-based hierarchy (Hossain & Sariyuce, 2026, simplified): level j = connected
  components of the k_j-core; nodes outside the k_j-core keep their level j-1 community.
  Deterministic (seed ignored).
"""
from __future__ import annotations

import random

import igraph as ig
import leidenalg
import numpy as np

Levels = list[list[int]]


def _relabel(m: list[int]) -> list[int]:
    ids: dict[int, int] = {}
    return [ids.setdefault(c, len(ids)) for c in m]


def _flat(sub: ig.Graph, method: str, seed: int) -> list[int]:
    w = "weight" if "weight" in sub.es.attributes() else None
    if method == "leiden":
        p = leidenalg.find_partition(sub, leidenalg.ModularityVertexPartition, weights=w, seed=seed)
        return list(p.membership)
    if method == "louvain":
        ig.set_random_number_generator(random.Random(seed))
        return list(sub.community_multilevel(weights=w).membership)
    raise ValueError(method)


def hierarchical(g: ig.Graph, method: str, seed: int, max_cluster_size: int = 10,
                 max_levels: int = 4) -> Levels:
    levels: Levels = [_relabel(_flat(g, method, seed))]
    while len(levels) < max_levels:
        prev = levels[-1]
        groups: dict[int, list[int]] = {}
        for v, c in enumerate(prev):
            groups.setdefault(c, []).append(v)
        big = {c: vs for c, vs in groups.items() if len(vs) > max_cluster_size}
        if not big:
            break
        nxt = [f"{c}" for c in prev]  # string keys keep sub-communities distinct
        changed = False
        for c, vs in big.items():
            sub = g.induced_subgraph(vs)
            if sub.ecount() == 0:
                continue
            m = _flat(sub, method, seed)
            if len(set(m)) > 1:
                changed = True
            for v, sc in zip(vs, m):
                nxt[v] = f"{c}.{sc}"
        if not changed:
            break
        levels.append(_relabel([hash(x) for x in nxt]))
    return levels


def kcore(g: ig.Graph, n_levels: int = 4) -> Levels:
    core = np.array(g.coreness())
    ks = sorted(set(core[core > 0].tolist()))
    # pick up to n_levels thresholds spread over the core range (always include k=1)
    if len(ks) > n_levels:
        idx = np.linspace(0, len(ks) - 1, n_levels).round().astype(int)
        ks = [ks[i] for i in sorted(set(idx))]
    levels: Levels = []
    prev = [f"r{v}" for v in range(g.vcount())]  # before level 0: every node alone
    first = True
    for k in ks:
        inside = [v for v in range(g.vcount()) if core[v] >= k]
        sub = g.induced_subgraph(inside)
        comp = sub.connected_components().membership
        cur = list(prev)
        for v, c in zip(inside, comp):
            cur[v] = f"{prev[v] if not first else ''}|k{k}c{c}"
        prev, first = cur, False
        levels.append(_relabel(cur))
    return levels or [list(range(g.vcount()))]


def cluster(g: ig.Graph, method: str, seed: int = 0, max_cluster_size: int = 10) -> Levels:
    if method in ("leiden", "louvain"):
        return hierarchical(g, method, seed, max_cluster_size)
    if method == "kcore":
        return kcore(g)
    raise ValueError(f"unknown method {method}")
