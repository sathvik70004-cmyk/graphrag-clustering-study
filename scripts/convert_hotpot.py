"""Convert the HuggingFace HotpotQA parquet to the original hotpot_dev_distractor_v1.json format."""
import json

import pandas as pd

SRC = "data/raw/hotpot_dev_distractor.parquet"
DST = "data/raw/hotpot_dev_distractor_v1.json"

df = pd.read_parquet(SRC)
out = []
for r in df.itertuples(index=False):
    sf, ctx = r.supporting_facts, r.context
    out.append({
        "_id": r.id,
        "question": r.question,
        "answer": r.answer,
        "type": r.type,
        "level": r.level,
        "supporting_facts": [[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])],
        "context": [[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])],
    })
json.dump(out, open(DST, "w", encoding="utf-8"), ensure_ascii=False)
print(f"wrote {len(out)} questions to {DST}")
