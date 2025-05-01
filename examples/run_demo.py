#!/usr/bin/env python
"""
examples/run_demo.py
End-to-end smoke-test for the chat + memory stack.

Scenario
--------
1. Ask:  “What is square root of 16?”
2. Ask:  “What about 81?”
3. Fetch the full conversation from /memory_retrieval
"""

from __future__ import annotations
import time, json, sys
import httpx

CHATBOT = "http://localhost:8000/predict"
MEM_RAG = "http://memory:8001/memory_retrieval"

def pretty(label: str, obj):
    print(f"\n— {label} —")
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def ask(message: str) -> None:
    r = httpx.post(CHATBOT, json={"user_message": message})
    r.raise_for_status()
    pretty(f"/predict  ↳  {message!r}", r.json())
    # allow async memory write to finish
    time.sleep(0.4)

def main() -> None:
    ask("What is square root of 16?")
    ask("What about 81?")

    # empty query → MemoryManager returns the **whole** history
    r = httpx.post(MEM_RAG, json={"query": "", "k": 3})
    r.raise_for_status()
    pretty("/memory_retrieval  (full history)", r.json())

    print("\n✅ demo finished – all good!")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)
