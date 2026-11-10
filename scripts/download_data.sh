#!/usr/bin/env bash
# Downloads HotpotQA dev (distractor) from the HuggingFace mirror (the original CMU host is
# often unreachable), converts it to the original JSON format, and fetches the official eval script.
set -euo pipefail
mkdir -p data/raw
curl -sSL -o data/raw/hotpot_dev_distractor.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet
curl -sSL -o data/raw/hotpot_evaluate_v1.py \
  https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
uv run python scripts/convert_hotpot.py
