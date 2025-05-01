from __future__ import annotations
import os, time, httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

MEMORY_HOST = os.getenv("MEMORY_HOST", "http://memory:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")

app = FastAPI(title="Chatbot API", version="0.3.0")


class ChatRequest(BaseModel):
    user_message: str


class ChatResponse(BaseModel):
    answer: str


def _call_openai(prompt: str) -> str:
    import openai

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = OPENAI_API_KEY
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    return rsp.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = httpx.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)

    if r.status_code == 404:                    # auto-pull then retry
        p = httpx.post(f"{OLLAMA_HOST}/api/pull", json={"name": OLLAMA_MODEL})
        for line in p.iter_text():
            if '"status":"success"' in line or '"status":"done"' in line:
                break
        time.sleep(1)
        r = httpx.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)

    r.raise_for_status()
    return r.json()["response"].strip()


@app.post("/predict", response_model=ChatResponse)
async def predict(req: ChatRequest, request: Request):
    try:
        mem_rsp = httpx.post(f"{MEMORY_HOST}/memory_retrieval", json={"query": req.user_message})
        mem_rsp.raise_for_status()
        context = mem_rsp.json()["context"]
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"memory error: {exc}") from exc

    prompt = (
        "You are a helpful assistant. "
        "If no relevant memory is given, answer normally.\n"
        f"{context}\n\n"
        f"User: {req.user_message}\nAssistant:"
    )

    try:
        answer = _call_openai(prompt) if OPENAI_API_KEY else _call_ollama(prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    # async fire-and-forget
    httpx.post(
        f"{MEMORY_HOST}/memory_retrieval/add",
        json={"user": req.user_message, "assistant": answer},
        timeout=5,
    )
    return ChatResponse(answer=answer)
