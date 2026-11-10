"""Local LLM calls through Ollama with a disk cache (runs are resumable and reproducible).

Thinking is disabled (think=False); temperature 0. Cache key = model + options + messages.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import ollama
import yaml

CACHE = Path("runs/cache/llm")


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return yaml.safe_load(open("configs/experiment.yaml"))["models"]


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(host=_cfg()["ollama_host"])


def chat(prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
    m = _cfg()
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    opts = {"temperature": m.get("temperature", 0), "num_predict": max_tokens, "num_ctx": 8192}
    key = hashlib.sha256(json.dumps([m["llm"], opts, msgs], sort_keys=True).encode()).hexdigest()
    path = CACHE / key[:2] / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]
    r = _client().chat(model=m["llm"], messages=msgs, think=False, options=opts)
    text = r["message"]["content"].strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text, "eval_count": r.get("eval_count")}), encoding="utf-8")
    return text
