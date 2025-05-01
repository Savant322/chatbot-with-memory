# Chatbot-with-Memory
A tiny stack that gives a normal LLM chatbot a long-term memory.
Everything runs with one Docker Compose command and fits on an 8 GB GPU.

## Quick-start

clone the repo <br>
``` bash
git clone https://github.com/Savant322/test-assignment.git
```

``` bash
cd test-assignment
```

first run – builds images and pulls a 7-B model (~2 GB once)

``` bash
docker compose up --build
```

Building docker compose takes some time, be patient

Swagger UI will be at http://localhost:8000/docs

## One-shot demo

After docker compose is up, run the following command to see if the service with chathistory works:

``` bash
docker compose exec chatbot python /app/examples/run_demo.py
```

You should see

   Q: What is square root of 16??  -> 4 <br>
   Q: What about 81??  -> 9 (it means that chathistory works) <br>
   /memory_retrieval returns both turns

## Key REST endpoints

POST /predict { "user_message": "Hi" } -> { "answer": "…" } <br>
POST /memory_retrieval { "query": "City", "k": 3 } -> { "context": "…" } <br>
POST /memory_retrieval/add { "user": "...", "assistant": "..." } -> { "status":"ok" } <br>
GET /healthz -> { "status":"up" }

## Environment variables (defaults)

OLLAMA_MODEL mistral:7b-instruct-q4_K_M (any model tag that fits) <br>
OPENAI_API_KEY (blank) – if set, uses GPT-3.5 instead of Ollama <br>
BUFFER_SIZE 6 – turns kept verbatim in RAM <br>
SUMMARY_EVERY 40 – summarise buffer every N turns <br>
MEMORY_PERSIST_DIR ./chroma_store

## Folder map

src/ <br>
├ chatbot_api.py -> prompt builder + LLM client <br>
├ memory_service.py -> REST wrapper around MemoryManager <br>
├ memory.py -> buffer + Chroma vector DB + T5 summaries <br>
└ utils.py -> device helper + summariser <br>
examples/run_demo.py -> end-to-end smoke test <br>
Dockerfile -> single Python image (chatbot & memory) <br>
docker-compose.yml -> adds Ollama GPU container + volume

## How it works (logic of the project)

The chatbot receives a user message on /predict.

It calls the memory service which returns the most relevant past snippets (MiniLM embeddings stored in Chroma).

It builds a prompt: fixed instruction + snippets (if any) + the new user message.

That prompt is sent to an LLM: <br>
– OpenAI GPT-3.5 when OPENAI_API_KEY is set <br>
– otherwise Ollama running a local model (auto-pulls the first time).

The answer is returned to the client and, in the background, the pair {user, assistant} is posted to /memory_retrieval/add.

Memory saves the turn (both in its rolling buffer and in the vector store); every SUMMARY_EVERY turns it summarises the buffer with a tiny T5-small model to avoid context bloat.

That is all – small, fast, self-contained.

## First-run download & other tips

- First request feels slow? <br>
Ollama is downloading the model weights (≈ 2 GB for Mistral-7B Q4).
Watch with: 
``` bash
docker compose logs -f ollama
```

Subsequent starts are instant.

- Smaller GPU / CPU-only? <br>
Set OLLAMA_MODEL=phi3:mini (~1.8 GB) or any CPU model, and remove the GPU reservation blocks in docker-compose.yml.
