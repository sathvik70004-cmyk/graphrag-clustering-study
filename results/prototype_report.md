# Prototype results (20 HotpotQA questions, 200 paragraphs)

> One question = 5 points of exact match. Differences of 1-2 questions are noise at this size; the prototype shows the pipeline works, not which method wins.

## 1. Answer quality and evidence (clean graph)

|                                     |   em |   f1 |   recall@10 |   ctx_tokens |
|:------------------------------------|-----:|-----:|------------:|-------------:|
| ('bm25', '-', '-')                  | 0.25 | 0.51 |        0.69 |       420.8  |
| ('dense', '-', '-')                 | 0.25 | 0.54 |        0.81 |       371.15 |
| ('global', 'llm_merged', 'kcore')   | 0.3  | 0.48 |      nan    |        48    |
| ('global', 'llm_merged', 'leiden')  | 0.3  | 0.55 |      nan    |        60.6  |
| ('global', 'llm_merged', 'louvain') | 0.3  | 0.55 |      nan    |        59.4  |
| ('global', 'nlp', 'kcore')          | 0.1  | 0.19 |      nan    |         7.4  |
| ('global', 'nlp', 'leiden')         | 0.2  | 0.41 |      nan    |        45.65 |
| ('global', 'nlp', 'louvain')        | 0.25 | 0.42 |      nan    |        37.9  |
| ('local', 'llm_merged', 'kcore')    | 0.2  | 0.52 |        0.76 |       655.65 |
| ('local', 'llm_merged', 'leiden')   | 0.15 | 0.47 |        0.76 |       655    |
| ('local', 'llm_merged', 'louvain')  | 0.15 | 0.47 |        0.76 |       654.95 |
| ('local', 'nlp', 'kcore')           | 0.3  | 0.6  |        0.8  |       659.6  |
| ('local', 'nlp', 'leiden')          | 0.2  | 0.53 |        0.8  |       694.7  |
| ('local', 'nlp', 'louvain')         | 0.25 | 0.61 |        0.8  |       693.25 |
| ('routed', 'llm_merged', 'kcore')   | 0.25 | 0.53 |        0.7  |       795.35 |
| ('routed', 'llm_merged', 'leiden')  | 0.2  | 0.51 |        0.8  |       794.15 |
| ('routed', 'llm_merged', 'louvain') | 0.2  | 0.51 |        0.8  |       793.45 |
| ('routed', 'nlp', 'kcore')          | 0.25 | 0.5  |        0.64 |       789.45 |
| ('routed', 'nlp', 'leiden')         | 0.2  | 0.51 |        0.79 |       829.45 |
| ('routed', 'nlp', 'louvain')        | 0.25 | 0.56 |        0.81 |       824.45 |

## 2. Effect of edge removal (local and routed retrieval)

|                                     |   ('em', 'clean') |   ('em', 'critical') |   ('em', 'random') |   ('f1', 'clean') |   ('f1', 'critical') |   ('f1', 'random') |   ('recall@10', 'clean') |   ('recall@10', 'critical') |   ('recall@10', 'random') |
|:------------------------------------|------------------:|---------------------:|-------------------:|------------------:|---------------------:|-------------------:|-------------------------:|----------------------------:|--------------------------:|
| ('local', 'llm_merged', 'kcore')    |              0.2  |                 0.2  |               0.2  |              0.52 |                 0.52 |               0.52 |                     0.76 |                        0.75 |                      0.76 |
| ('local', 'llm_merged', 'leiden')   |              0.15 |                 0.15 |               0.15 |              0.47 |                 0.43 |               0.47 |                     0.76 |                        0.75 |                      0.76 |
| ('local', 'llm_merged', 'louvain')  |              0.15 |                 0.15 |               0.15 |              0.47 |                 0.47 |               0.47 |                     0.76 |                        0.75 |                      0.76 |
| ('local', 'nlp', 'kcore')           |              0.3  |                 0.25 |               0.2  |              0.6  |                 0.56 |               0.54 |                     0.8  |                        0.8  |                      0.8  |
| ('local', 'nlp', 'leiden')          |              0.2  |                 0.15 |               0.12 |              0.53 |                 0.5  |               0.48 |                     0.8  |                        0.8  |                      0.8  |
| ('local', 'nlp', 'louvain')         |              0.25 |                 0.2  |               0.15 |              0.61 |                 0.58 |               0.54 |                     0.8  |                        0.8  |                      0.8  |
| ('routed', 'llm_merged', 'kcore')   |              0.25 |                 0.3  |               0.3  |              0.53 |                 0.58 |               0.54 |                     0.7  |                        0.74 |                      0.69 |
| ('routed', 'llm_merged', 'leiden')  |              0.2  |                 0.2  |               0.2  |              0.51 |                 0.48 |               0.48 |                     0.8  |                        0.8  |                      0.78 |
| ('routed', 'llm_merged', 'louvain') |              0.2  |                 0.2  |               0.2  |              0.51 |                 0.48 |               0.49 |                     0.8  |                        0.8  |                      0.78 |
| ('routed', 'nlp', 'kcore')          |              0.25 |                 0.2  |               0.25 |              0.5  |                 0.51 |               0.48 |                     0.64 |                        0.75 |                      0.62 |
| ('routed', 'nlp', 'leiden')         |              0.2  |                 0.15 |               0.2  |              0.51 |                 0.54 |               0.53 |                     0.79 |                        0.79 |                      0.79 |
| ('routed', 'nlp', 'louvain')        |              0.25 |                 0.2  |               0.2  |              0.56 |                 0.59 |               0.52 |                     0.81 |                        0.81 |                      0.8  |

## 3. Graph stability after edge removal (level 0)

ARI = agreement with the clean graph's communities (1 = identical). Seed-noise ARI = agreement between two runs on the SAME clean graph.

|                           |   ('lcc_frac', 'critical') |   ('lcc_frac', 'random') |   ('ari', 'critical') |   ('ari', 'random') |   ('conn_orig_pct', 'critical') |   ('conn_orig_pct', 'random') |   ('n_communities', 'critical') |   ('n_communities', 'random') |
|:--------------------------|---------------------------:|-------------------------:|----------------------:|--------------------:|--------------------------------:|------------------------------:|--------------------------------:|------------------------------:|
| ('llm', 'kcore')          |                      0.032 |                    0.058 |                 0.746 |               0.912 |                           96.47 |                         82.94 |                         172     |                       176.5   |
| ('llm', 'leiden')         |                      0.032 |                    0.058 |                 0.869 |               0.949 |                           95.4  |                         82.76 |                         172     |                       178     |
| ('llm', 'louvain')        |                      0.032 |                    0.058 |                 0.869 |               0.949 |                           95.4  |                         82.76 |                         172     |                       178     |
| ('llm_merged', 'kcore')   |                      0.041 |                    0.058 |                 0.785 |               0.911 |                           96.43 |                         83.33 |                         171     |                       175     |
| ('llm_merged', 'leiden')  |                      0.041 |                    0.058 |                 0.873 |               0.944 |                           95.35 |                         83.14 |                         172     |                       176.5   |
| ('llm_merged', 'louvain') |                      0.041 |                    0.058 |                 0.873 |               0.944 |                           95.35 |                         83.14 |                         172     |                       176.5   |
| ('nlp', 'kcore')          |                      0.941 |                    0.971 |                 0.648 |               1     |                           50    |                        100    |                           5     |                         4     |
| ('nlp', 'leiden')         |                      0.941 |                    0.971 |                 0.785 |               0.786 |                          100    |                        100    |                          17     |                        15.667 |
| ('nlp', 'louvain')        |                      0.941 |                    0.971 |                 0.886 |               0.875 |                          100    |                        100    |                          16.333 |                        15.833 |

|                           |   seed_noise_ari |
|:--------------------------|-----------------:|
| ('llm', 'leiden')         |             1    |
| ('llm', 'louvain')        |             1    |
| ('llm_merged', 'leiden')  |             1    |
| ('llm_merged', 'louvain') |             1    |
| ('nlp', 'leiden')         |             0.88 |
| ('nlp', 'louvain')        |             1    |

## 4. Gold-path connectivity (% of questions whose two gold paragraphs are linked)

| graph      |   clean |   critical |   random |
|:-----------|--------:|-----------:|---------:|
| llm        |      70 |         70 |       70 |
| llm_merged |      70 |         70 |       70 |
| nlp        |     100 |        100 |      100 |

Unchanged by edge removal: the two gold paragraphs are almost always joined by one entity named in both, not by a graph path.

## 5. Examples

### Evidence fully retrieved, answer still wrong (answer-model limit)

- gold **Colonel Charles Hastings Judd** / predicted *George W. Macfarlane* (llm_merged, leiden)
- gold **Bob Day** / predicted *Lucy Gichuhi* (llm_merged, leiden)
- gold **was a rock band** / predicted *Virgin Records* (llm_merged, leiden)

### Correct on the clean graph, wrong after critical-edge removal

- local/llm_merged/leiden: gold **Johnny D. Bright**: clean *Johnny Bright*, critical *Damon Allen*
- routed/llm_merged/leiden: gold **Buck Owens**: clean *Buck Owens and the Buckaroos*, critical *The Beatles*
- routed/llm_merged/louvain: gold **Buck Owens**: clean *Buck Owens and the Buckaroos*, critical *The Beatles*
- local/nlp/leiden: gold **Buck Owens**: clean *Buck Owens*, critical *Gene Watson*

### Global search answered 5 question(s) that dense retrieval missed

- gold **the Mikoyan design bureau** (llm_merged, leiden)
- gold **1995** (llm_merged, leiden)
- gold **Manchester** (llm_merged, leiden)
- gold **Colonel Charles Hastings Judd** (nlp, louvain)
- gold **President of the United States** (nlp, kcore)
