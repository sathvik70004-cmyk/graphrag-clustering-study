"""Supporting-fact (evidence) metrics, sentence level, keyed by (title, sent_id).

recall@k : fraction of gold supporting sentences among the top-k retrieved sentences
f1@k     : F1 between the top-k set and the gold set (HotpotQA sp-F1 computed on the retrieved set)
both_gold_paras@k : 1 if every gold paragraph has at least one sentence in the top k
                    (multi-hop needs BOTH hops)
"""
from __future__ import annotations

Key = tuple[str, int]


def sp_scores(retrieved: list[Key], gold: list[Key], k: int) -> dict[str, float]:
    top = set(map(tuple, retrieved[:k]))
    g = set(map(tuple, gold))
    tp = len(top & g)
    p = tp / len(top) if top else 0.0
    r = tp / len(g) if g else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    gold_titles = {t for t, _ in g}
    hit_titles = {t for t, _ in top}
    return {
        f"recall@{k}": r,
        f"f1@{k}": f1,
        f"both_gold_paras@{k}": float(gold_titles <= hit_titles),
    }
