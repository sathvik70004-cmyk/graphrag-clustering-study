# Does More Stable Clustering Improve GraphRAG Answers?

A focused study of how GraphRAG community detection (Leiden, Louvain, k-core hierarchy) responds to
missing graph edges, and whether that changes evidence retrieval and answer quality on
multi-hop QA ([HotpotQA](https://arxiv.org/abs/1809.09600), dev distractor set).

## Design
- **Indexing:** Microsoft GraphRAG (pinned `graphrag==3.3.0`) builds the knowledge graph **once**;
  entity/relationship tables are frozen. Everything downstream is our own code.
- **Clustering:** Leiden, Louvain, k-core hierarchy on the same graph (3 seeds for stochastic methods).
- **Perturbation:** 5% random edge removal (×5) and top-5% edge-betweenness removal.
- **Retrieval (selectable per query):** `local` (R1), `routed` community-routed (R2), `global` (R3).
- **Models (local, zero cost):** Ollama `gemma3:4b` + `embeddinggemma` (dev; swappable in configs/experiment.yaml) on an RTX 4060 (8 GB).
- **Metrics:** LCC fraction, ARI, % connected communities, supporting-fact Recall@k/F1,
  HotpotQA EM/F1, indexing time, context tokens.

## Phases
1. Setup: env, Ollama models, data download ✅
2. Data: stratified 500-q sample (100 pilot), deduped paragraph corpus with `(title, sent_id)` ✅ (`uv run python -m graphrag_study.data.build`)
3. Index once with GraphRAG; entity→sentence provenance
4. Baselines: BM25 / dense RAG
5. Clustering × perturbation, graph metrics
6. Community reports (member-set cache)
7. Retrieval + supporting-fact metrics
8. Answering + EM/F1
9. Analysis + error inspection

## Setup
```bash
pip install uv
uv sync
ollama pull gemma3:4b
ollama pull embeddinggemma
bash scripts/download_data.sh
uv run python scripts/check_env.py
```

## Layout
```
configs/          frozen experiment settings
graphrag_index/   MS GraphRAG workspace (settings, prompts, input/output)
data/             raw HotpotQA + processed sample/corpus (gitignored)
src/graphrag_study/{data,graph,cluster,perturb,reports,retrieve,answer,eval}
runs/             cached outputs per (method, clustering, condition, seed) (gitignored)
notebooks/        analysis
```
