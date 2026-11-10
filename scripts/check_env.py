"""Sanity check: Ollama models respond, data present, key libraries import."""
import json
import pathlib
import sys

import igraph
import leidenalg
import ollama
import yaml

cfg = yaml.safe_load(open("configs/experiment.yaml"))
m = cfg["models"]
client = ollama.Client(host=m["ollama_host"])

r = client.chat(model=m["llm"], messages=[{"role": "user", "content": "Reply with just: OK /no_think"}],
                options={"temperature": 0})
print("LLM:", m["llm"], "->", r["message"]["content"].strip()[-40:])
e = client.embed(model=m["embedding"], input=["hello world"])
print("Embedding:", m["embedding"], "dim =", len(e["embeddings"][0]))

p = pathlib.Path(cfg["data"]["hotpot_file"])
if not p.exists():
    sys.exit(f"missing {p}; run scripts/download_data.sh")
d = json.load(open(p, encoding="utf-8"))
print("HotpotQA dev distractor:", len(d), "questions")
print("igraph", igraph.__version__, "| leidenalg", leidenalg.version)
print("ALL OK")
