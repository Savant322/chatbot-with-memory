from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .memory import MemoryManager

mem = MemoryManager(
    os.getenv("MEMORY_PERSIST_DIR", "./chroma_store"),
    int(os.getenv("BUFFER_SIZE", 6)),
    int(os.getenv("SUMMARY_EVERY", 40)),
)

app = FastAPI(title="Memory Service", version="0.3.0")


class RetrievalRequest(BaseModel):
    query: str
    k: int = Field(3, ge=1, le=10)


class AddTurnRequest(BaseModel):
    user: str
    assistant: str


@app.post("/memory_retrieval")
def retrieve(req: RetrievalRequest):
    return {"context": mem.retrieve(req.query, k=req.k)}


@app.post("/memory_retrieval/add")
def add_turn(req: AddTurnRequest):
    if not req.user.strip() or not req.assistant.strip():
        raise HTTPException(400, "empty user or assistant message")
    mem.add_turn(req.user, req.assistant)
    return {"status": "ok"}


@app.get("/healthz")
def health():
    return {"status": "up"}
