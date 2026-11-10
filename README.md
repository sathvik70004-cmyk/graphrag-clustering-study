# Does More Stable Clustering Improve GraphRAG Answers?

A focused study of how GraphRAG community detection (Leiden, Louvain, k-core hierarchy) responds to
missing graph edges, and whether that changes evidence retrieval and answer quality on
multi-hop QA ([HotpotQA](https://arxiv.org/abs/1809.09600), dev distractor set).

**Current stage: working prototype** on 20 questions (200 Wikipedia paragraphs), built to show the
full pipeline end to end before scaling to the planned 500 questions.

## Design
- **Graph construction:** Microsoft GraphRAG (pinned `graphrag==3.3.0`) is used **only** to extract
  the knowledge graph, once; its entity/relationship tables are frozen. Clustering, perturbation,
  community reports, retrieval and evaluation are our own code (`src/graphrag_study`).
- **Three graphs:**
  - `llm`: standard GraphRAG LLM extraction (local `qwen3:8b`)
  - `llm_merged`: `llm` + conservative entity merging (fixes naming drift)
  - `nlp`: GraphRAG "fast" mode (noun phrases + co-occurrence, no LLM)
- **Clustering:** hierarchical Leiden, hierarchical Louvain (GraphRAG scheme, max community size 10),
  k-core hierarchy. 3 seeds for the random methods.
- **Perturbation:** remove 5% of edges at random (2 repeats in the prototype, 5 in the full study)
  and the top-5% edges by edge betweenness ("critical" edges).
- **Retrieval (selectable per query):** `local` (R1), `routed` community-routed (R2), `global` (R3),
  plus no-graph baselines `bm25` and `dense`.
- **Models (local, zero cost, RTX 4060 8 GB):** Ollama `qwen3:8b` (thinking off) for extraction,
  reports and answers; `embeddinggemma` for embeddings.
- **Metrics:**
  - Graph: LCC fraction, ARI vs. the clean graph, % of communities still connected, seed-noise ARI.
  - Multi-hop: gold-path connectivity (are the two gold paragraphs linked?).
  - Evidence: supporting-fact Recall@k / F1.
  - Answers: HotpotQA EM / F1.
  - Cost: context tokens, time.

## Phases
| # | Phase | Status |
|---|---|---|
| 1 | Setup: env, models, data download | ✅ |
| 2 | Data: stratified sample, deduplicated corpus with `(title, sent_id)` | ✅ |
| 3 | Knowledge graphs (LLM + NLP) + entity merging | ✅ prototype |
| 4 | Baselines: BM25 / dense | ✅ prototype |
| 5 | Clustering × perturbation, graph + gold-path metrics | ✅ prototype |
| 6 | Community reports | ✅ prototype |
| 7 | Retrieval: local / routed / global | ✅ prototype |
| 8 | Answering + EM/F1 | ✅ prototype |
| 9 | Analysis, error inspection, demo | ✅ prototype ([report](results/prototype_report.md)) |

## Project log: what we tried, what went wrong, how we worked around it

### Phase 1 — Setup
- **Goal:** a reproducible Python environment, local LLMs (zero spending), the HotpotQA dataset, and
  a private GitHub repo.
- **Challenges and workarounds:**
  - **Dataset server down.** The official host (`curtis.ml.cmu.edu`) no longer resolves.
    → We download the official HuggingFace mirror (parquet) and convert it to the original JSON
    format (`scripts/convert_hotpot.py`), so the official evaluation script still works.
  - **Model downloads were slow.** We started development on already-installed `gemma3:4b`, then
    moved to `qwen3:8b` for better graph quality.
  - **`qwen3-coder:30b`** (already installed) doesn't fit in 8 GB of VRAM, so it wasn't used.

### Phase 2 — Data
- **Goal:** sample questions (stratified by bridge/comparison type) and build one corpus from their
  paragraphs, deduplicated, keeping every sentence's `(title, sent_id)` so supporting-fact recall
  can be measured.
- **Result:**
  - 500-question sample, with a 100-question pilot and a 20-question prototype as prefixes of it.
  - The prototype has 200 paragraphs and 816 sentences.
  - No gold fact is missing from the corpus.
- **Challenge:** two different paragraphs share one Wikipedia title (full set only).
  → We keep the first; no gold fact is affected.

### Phase 3 — Knowledge graphs
- **Goal:** build the GraphRAG knowledge graph once and freeze it.
- **Challenges and workarounds:**
  1. **Local extraction was very slow** (~1.5–2 min per paragraph, GPU at full load).
     → Root cause: qwen3 was silently "thinking". litellm drops a top-level `reasoning_effort`
     for non-OpenAI models, so we now pass it through `call_args.extra_body` (~3× faster).
     → We also send 5 paragraphs per extraction call (the ~1.9k-token instruction prompt is sent
     once per 5 paragraphs instead of once per paragraph) and turned off gleaning.
     → Running 2 requests in parallel on an 8 GB GPU was ~3× slower, so we use 1.
  2. **Ollama's default context window can silently truncate prompts.**
     → A model variant with an 8k context: `ollama/qwen3-8b-gr.Modelfile`.
  3. **We tried free online APIs instead of the GPU.**
     - `gemini-2.5-flash-lite` is closed to new keys.
     - `gemini-3.5-flash-lite` garbled GraphRAG's `<|>` delimiters, so ~70% of relationships were
       silently dropped.
     - `gemini-3.5-flash` produced clean output, but the free tier allows only 20 requests per
       day.
     → We went back to the local GPU and shrank the prototype to 20 questions (200 paragraphs,
     40 extraction calls, ~29 min). Online settings remain available in
     `graphrag_index/.env.example`.
  4. **Second graph type.** GraphRAG's "fast" NLP mode builds a graph in ~1 min on the CPU, with no
     LLM.
     → The NLP graph needs one paragraph per file, because co-occurrence is counted per file;
     grouping 5 unrelated paragraphs would invent links.
  5. **The LLM graph is fragmented** (565 entities, 159 pieces). The LLM names the same thing
     differently ("KALĀKAUĀ" vs "KING DAVID KALĀKAUA") or misspells it ("MIKODYAN MIG-29").
     → **Entity merging** (`graph/merge.py`), with conservative, deterministic rules.
     → The first version wrongly merged e.g. *George W. Bush → George H. W. Bush* and
     *Saxony → Saxony-Anhalt*. Every merge is listed in `merges.csv` and was checked by hand, and
     the rules were tightened until only correct merges remained (4 merges).
     → **Limitation:** merging can't fix links the LLM never extracted (e.g. no "MiG-29" entity in
     the squadron paragraph).

### Phase 4 — Baselines
- **Goal:** no-graph reference points with the same evidence budget (top-10 sentences).
- **Result (20 questions):**
  - Gold sentences in the top 10: dense 81%, BM25 69%.
  - Both gold paragraphs in the top 10: dense 75%, BM25 65%.

### Phase 5 — Clustering × perturbation
- **Goal:** cluster every graph condition with every method and measure stability (LCC fraction,
  ARI, % connected communities).
- **Challenges and findings:**
  - **Seed noise.** Leiden run twice on the *same, unperturbed* graph only agrees at ARI ≈ 0.88,
    close to the effect of removing edges (≈ 0.79).
    → We report a seed-noise baseline next to every perturbation result.
  - **Hierarchy levels don't match across methods** (k-core gives 4–15 communities, Leiden 17–42
    on the NLP graph).
    → Results are reported per level; matching levels by community count is planned for the
    analysis.
  - **LCC fraction says little on a fragmented graph.**
    → We added a multi-hop-specific metric: **gold-path connectivity**.
  - **Key finding:** gold-path connectivity does not change under edge removal (70% for the LLM
    graph, 100% for the NLP graph, every method and condition). In HotpotQA the two gold
    paragraphs are almost always joined by **one entity named in both** (all 14 linked questions
    in the LLM graph, 19/20 in the NLP graph). So removing graph edges cannot break these bridges.
    → Decision: keep the planned design and report this (option A), and measure what clustering
    *does* change, namely which community summaries and evidence retrieval sees (option C).
    Also damaging entity ↔ paragraph links (option B) is to be discussed with the supervisor.

### Phase 6 — Community reports
- **Goal:** a summary for every community, used by local search (context), routed search
  (choosing communities) and global search (map-reduce).
- **Challenge:** GraphRAG's own reports are long JSON (~1–2k tokens each), which is too slow on a
  laptop GPU for 24 partitions.
  → Short plain summaries (≤ 120 words), built from member entities plus sentences mentioning them
  (the same recipe for every graph).
  → Cached by member set, so a community that edge removal leaves unchanged is summarised once.
  337 unique reports, ~3 s each.

### Phase 7 — Retrieval
- **Goal:** three retrieval methods with the same evidence budget.
  - **local (R1):** match question → entities → 1-hop neighbours → their sentences, plus the seed
    entities' community reports. Clustering only affects the reports.
  - **routed (R2):** match question → community reports → member entities' sentences. Clustering
    decides what evidence is reachable.
  - **global (R3):** map-reduce over community reports.
- **Challenge:** full global search maps over *every* report for *every* question, which is too
  slow locally.
  → Prototype simplification: map over the 20 most relevant reports, and run only on the clean
  graph.

### Phase 8 — Answering
- **Goal:** the same answer model, prompt and budget for every configuration; EM/F1 identical to
  the official HotpotQA script.
- **Smoke test (2 questions):** global search answered the MiG-29 question correctly even though
  the graph link was broken, because the fact was in a community summary. Local and routed found
  the gold evidence but the 8B model still answered wrongly, which shows the answer model's limits.

### Phase 9 — Analysis
- **Prototype results:** see [results/prototype_report.md](results/prototype_report.md)
  (regenerate with `uv run python -m graphrag_study.eval.analyze`).
- **Takeaways (20 questions, so indicative only):**
  - The whole pipeline runs end to end. The QA grid (1,120 answers) took ~10 min on the GPU.
  - Edge removal barely changes evidence retrieval (recall ±1 point), consistent with the
    gold-path finding. But critical-edge removal did flip some answers, e.g. *Buck Owens* became
    *The Beatles* (local and routed retrieval).
  - Global search answered 5 questions that dense retrieval missed: community summaries can carry
    a fact whose graph link is broken (the MiG-29 example).
  - The 8B answer model is a bottleneck: it often has all the gold evidence and still answers
    wrongly.
  - Leiden's seed noise is comparable to the perturbation effect on the NLP graph.

## What we are doing now
- **Prototype complete:** ready to demo to the supervisor.
- **Open questions for the supervisor:**
  1. Also damage entity ↔ paragraph links (option B), since edge removal alone can't break
     HotpotQA bridges.
  2. Keep entity merging as part of the method.
  3. Use a stronger answer model for the full run.
- **After approval:** scale to 500 questions. Build the LLM graph on free cloud GPUs (Kaggle) or a
  few dollars of paid API; everything else runs locally.

## Setup
```bash
pip install uv
uv sync
ollama pull qwen3:8b
ollama pull embeddinggemma
ollama create qwen3-8b-gr -f ollama/qwen3-8b-gr.Modelfile
bash scripts/download_data.sh
uv run python scripts/check_env.py
```
Copy `graphrag_index/.env.example` to `graphrag_index/.env` (local Ollama or an online provider).

## Pipeline
```bash
uv run python -m graphrag_study.data.build                 # Phase 2: sample + corpus + GraphRAG inputs
uv run graphrag index -r graphrag_index                     # Phase 3: LLM graph
uv run graphrag index -r graphrag_index_nlp                 #          NLP graph
uv run python -m graphrag_study.graph.merge                 #          entity merging -> llm_merged
uv run python -m graphrag_study.retrieve.baselines          # Phase 4
uv run python -m graphrag_study.eval.graph_metrics          # Phase 5
uv run python -m graphrag_study.reports.build               # Phase 6
uv run python -m graphrag_study.answer.run                  # Phases 7-8
```

## Demo
```bash
uv run python -m graphrag_study.query "Who developed the jet fighters operated by the 41st Tactical Squadron of the Polish Air Force?" --method global --clustering leiden
```
Options: `--method local|routed|global|dense|bm25`, `--clustering leiden|louvain|kcore`,
`--graph llm_merged|nlp`, `--condition clean|random0|random1|critical`.

## Layout
```
configs/              frozen experiment settings
graphrag_index/       GraphRAG workspace, LLM graph (settings, prompts; input/output gitignored)
graphrag_index_nlp/   GraphRAG workspace, NLP graph
ollama/               local model definition (8k context)
data/                 raw HotpotQA + processed sample/corpus (gitignored)
src/graphrag_study/   data, graph, cluster, perturb, reports, retrieve, answer, eval, query
runs/                 partitions, reports, metrics, QA results, caches (gitignored)
```
