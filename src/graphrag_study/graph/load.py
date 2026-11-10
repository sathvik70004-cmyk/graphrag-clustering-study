"""Load a frozen GraphRAG output (entities/relationships) as an undirected igraph graph.

Usage:  uv run python -m graphrag_study.graph.load            # all graphs in GRAPHS
"""
from __future__ import annotations

import sys
from pathlib import Path

import igraph as ig
import pandas as pd

# graph key -> folder holding entities.parquet / relationships.parquet
GRAPHS = {
    "llm": "graphrag_index/output",                # standard GraphRAG LLM extraction
    "llm_merged": "graphrag_index/output_merged",  # + entity merging (graph/merge.py)
    "nlp": "graphrag_index_nlp/output",            # FastGraphRAG noun-phrase graph
}


def _dir(key_or_path: str | Path) -> Path:
    return Path(GRAPHS.get(str(key_or_path), key_or_path))


def load_tables(key_or_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = _dir(key_or_path)
    return pd.read_parquet(out / "entities.parquet"), pd.read_parquet(out / "relationships.parquet")


def aliases(key_or_path: str | Path) -> dict[str, list[str]]:
    """entity title -> all surface names (just the title unless the graph was merged)."""
    ents, _ = load_tables(key_or_path)
    if "aliases" in ents.columns:
        return {t: list(a) for t, a in zip(ents.title, ents.aliases)}
    return {t: [t] for t in ents.title}


def load_graph(key_or_path: str | Path) -> ig.Graph:
    """Nodes = all entities (isolated ones included); edges = relationships, weight kept.
    Parallel edges are merged (weights summed) and self-loops dropped, so every clustering
    method sees the same simple graph."""
    ents, rels = load_tables(key_or_path)
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
    for k in sys.argv[1:] or GRAPHS:
        if (_dir(k) / "entities.parquet").exists():
            print(k, stats(load_graph(k)))
