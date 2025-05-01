# Chatbot-with-Memory
A tiny stack that gives a normal LLM chatbot a long-term memory.
Everything runs with one Docker Compose command and fits on an 8 GB GPU.

## Quick-start

clone the repo
git clone git@github.com:Savant322/test-assignment.git
cd test-assignment

first run – builds images and pulls a 7-B model (~2 GB once)

``` bash
docker compose up --build
```

Swagger UI will be at http://localhost:8000/docs

restart instantly later
docker compose up -d

## One-shot demo

docker compose exec chatbot python /app/examples/run_demo.py

You should see

   Q: What is square root of 16??  -> 4
   Q: What about 81??  -> 9 (it means that chathistory works)
   /memory_retrieval returns both turns

## Key REST endpoints

POST /predict { "user_message": "Hi" } -> { "answer": "…" }
POST /memory_retrieval { "query": "City", "k": 3 } -> { "context": "…" }
POST /memory_retrieval/add { "user": "...", "assistant": "..." } -> { "status":"ok" }
GET /healthz -> { "status":"up" }

## Environment variables (defaults)

OLLAMA_MODEL mistral:7b-instruct-q4_K_M (any model tag that fits)
OPENAI_API_KEY (blank) – if set, uses GPT-3.5 instead of Ollama
BUFFER_SIZE 6 – turns kept verbatim in RAM
SUMMARY_EVERY 40 – summarise buffer every N turns
MEMORY_PERSIST_DIR ./chroma_store

## Folder map

src/
├ chatbot_api.py prompt builder + LLM client
├ memory_service.py REST wrapper around MemoryManager
├ memory.py buffer + Chroma vector DB + T5 summaries
└ utils.py device helper + summariser
examples/run_demo.py end-to-end smoke test
Dockerfile single Python image (chatbot & memory)
docker-compose.yml adds Ollama GPU container + volume

## How it works (short version)

The chatbot receives a user message on /predict.

It calls the memory service which returns the most relevant past snippets (MiniLM embeddings stored in Chroma).

It builds a prompt: fixed instruction + snippets (if any) + the new user message.

That prompt is sent to an LLM:
– OpenAI GPT-3.5 when OPENAI_API_KEY is set
– otherwise Ollama running a local model (auto-pulls the first time).

The answer is returned to the client and, in the background, the pair {user, assistant} is posted to /memory_retrieval/add.

Memory saves the turn (both in its rolling buffer and in the vector store); every SUMMARY_EVERY turns it summarises the buffer with a tiny T5-small model to avoid context bloat.

That is all – small, fast, self-contained.

## First-run download & other tips

- First request feels slow?
Ollama is downloading the model weights (≈ 2 GB for Mistral-7B Q4).
Watch with: docker compose logs -f ollama. Subsequent starts are instant.

- Smaller GPU / CPU-only?
Set OLLAMA_MODEL=phi3:mini (~1.8 GB) or any CPU model, and remove the GPU reservation blocks in docker-compose.yml.
