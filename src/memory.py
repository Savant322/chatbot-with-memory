from __future__ import annotations
import hashlib
from datetime import datetime
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .utils import summarise_text, best_device

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MemoryManager:
    def __init__(self, persist_dir: str | Path, buffer_size: int, summary_every: int):
        self.persist_dir = Path(persist_dir)
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=EMBED_MODEL, model_kwargs={"device": best_device()}
        )
        self.vstore = Chroma(
            collection_name="chat_messages",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        self.buffer = ConversationBufferMemory(k=buffer_size, return_messages=True)
        self.summary_every = summary_every
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def add_turn(self, user: str, assistant: str) -> None:
        self.buffer.save_context({"input": user}, {"output": assistant})
        for role, msg in (("user", user), ("assistant", assistant)):
            h = hashlib.sha1(msg.encode()).hexdigest()
            if self.vstore.get(where={"hash": h})["ids"]:
                continue
            meta = {"role": role, "hash": h, "created": datetime.utcnow().isoformat()}
            self.vstore.add_documents([Document(page_content=msg, metadata=meta)])
        self._maybe_summarise()

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 3) -> str:
        raw = self.buffer.load_memory_variables({})["history"]
        history = (
            "\n".join(m.content for m in raw) if isinstance(raw, list) else (raw or "")
        )
        if not query.strip() or len(self.vstore) == 0:
            return history
        snippets = "\n".join(d.page_content for d in self.vstore.similarity_search(query, k))
        return f"{history}\n\n# Relevant memories:\n{snippets}"

    # ------------------------------------------------------------------
    def _maybe_summarise(self):
        turns = len(self.buffer.chat_memory.messages) // 2
        if turns < self.summary_every:
            return
        summary = summarise_text(self.buffer.load_memory_variables({})["history"])
        self.buffer.clear()
        self.buffer.save_context({}, {"output": f"\n# Summary so far:\n{summary}"})
