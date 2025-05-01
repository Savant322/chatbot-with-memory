from __future__ import annotations
import logging
from functools import lru_cache
import torch
from transformers import pipeline

log = logging.getLogger(__name__)


def best_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _summariser():
    return pipeline(
        "summarization",
        model="t5-small",
        device=best_device(),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )


def summarise_text(text: str, max_tokens: int = 120) -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        out = _summariser()(text, max_length=max_tokens, min_length=10, do_sample=False)[0]
        return out["summary_text"].strip()
    except Exception as exc:  # fallback
        log.warning("summarisation failed: %s", exc)
        return (text[:max_tokens] + " …") if len(text) > max_tokens else text
