"""Load a frozen GraphRAG output (entities/relationships) as an undirected igraph graph.

Usage:  uv run python -m graphrag_study.graph.load graphrag_index graphrag_index_nlp
"""
from __future__ import annotations

import sys
from pathlib import Path

import igraph as ig
import pandas as pd

GRAPHS = {"llm": "graphrag_index", "nlp": "graphrag_index_nlp"}


def load_tables(index_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(index_dir) / "output"
    return pd.read_parquet(out / "entities.parquet"), pd.read_parquet(out / "relationships.parquet")


def load_graph(index_dir: str | Path) -> ig.Graph:
    """Nodes = all entities (isolated ones included); edges = relationships, weight kept.
    Parallel edges are merged (weights summed) and self-loops dropped, so every clustering
    method sees the same simple graph."""
    ents, rels = load_tables(index_dir)
    g = ig.Graph()
    g.add_vertices(ents.title.tolist())
    g.vs["type"] = ents.get("type", pd.Series([""] * len(ents))).fillna("").tolist()
    known = set(ents.title)
    rels = rels[rels.source.isin(known) & rels.target.isin(known)]
    g.add_edges(list(zip(rels.source, rels.target)))
    g.es["weight"] = rels.get("weight", pd.Series([1.0] * len(rels))).astype(float).tolist()
    g.simplify(multiple=True, loops=True, combine_edges={"weight": "sum"})
    return g


def stats(g: ig.Graph) -> dict:
    cc = g.connected_components()
    lcc = max(cc.sizes()) if g.vcount() else 0
    return {
        "nodes": g.vcount(),
        "edges": g.ecount(),
        "isolated": sum(d == 0 for d in g.degree()),
        "components": len(cc),
        "lcc_nodes": lcc,
        "lcc_frac": round(lcc / max(g.vcount(), 1), 3),
        "mean_degree": round(2 * g.ecount() / max(g.vcount(), 1), 2),
    }


if __name__ == "__main__":
    for d in sys.argv[1:] or GRAPHS.values():
        print(d, stats(load_graph(d)))
