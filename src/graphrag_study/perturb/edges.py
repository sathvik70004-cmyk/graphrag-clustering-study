"""Edge-removal perturbations. Node set is always preserved (only edges are removed)."""
from __future__ import annotations

import random

import igraph as ig


def remove_random(g: ig.Graph, fraction: float, seed: int) -> ig.Graph:
    rng = random.Random(seed)
    k = round(fraction * g.ecount())
    h = g.copy()
    h.delete_edges(rng.sample(range(g.ecount()), k))
    return h


def remove_critical(g: ig.Graph, fraction: float) -> ig.Graph:
    """Remove the `fraction` of edges with the highest edge betweenness (unweighted shortest
    paths: betweenness measures structural bridging, not relation strength). Deterministic."""
    k = round(fraction * g.ecount())
    eb = g.edge_betweenness(directed=False)
    top = sorted(range(g.ecount()), key=lambda e: (-eb[e], e))[:k]
    h = g.copy()
    h.delete_edges(top)
    return h


def conditions(g: ig.Graph, fraction: float, random_repeats: int) -> dict[str, ig.Graph]:
    out = {"clean": g}
    for r in range(random_repeats):
        out[f"random{r}"] = remove_random(g, fraction, seed=1000 + r)
    out["critical"] = remove_critical(g, fraction)
    return out
