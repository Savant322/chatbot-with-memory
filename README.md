# Chatbot with Long-Term Memory  
*(FastAPI · LangChain · Ollama)*

A minimal micro-service stack that adds lightweight “memory” to an LLM-powered chatbot.

| Service    | Purpose                                                                                                             | Tech                                                                                               |
|------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **chatbot** | Builds the prompt, calls an LLM (Ollama *or* OpenAI), and stores new turns to memory                                | FastAPI                                                                                            |
| **memory**  | Conversation buffer + semantic search (Chroma) + optional summaries                                                 | FastAPI, LangChain                                                                                 |
| **ollama**  | Local GPU inference for open-weight models (default = Mistral-7B Q4)                                                | [ollama/ollama](https://github.com/ollama/ollama)                                                  |


## 1 · Quick start

```bash
git clone git@github.com:Savant322/test-assignment.git
cd test-assignment
```

# first run – builds images & pulls model (~2 GB once)
```bash
docker compose up --build
```

# Swagger UI: http://localhost:8000/docs

# Stop with Ctrl-C; restart in seconds:
```bash
docker compose up -d
```

## 2 · REST endpoints

| URL (path)                     | Method | JSON body                                 | Result                |
|--------------------------------|--------|-------------------------------------------|-----------------------|
| `/predict` (chatbot)           | POST   | `{ "user_message": "Hi" }`                | `{ "answer": "…" }`   |
| `/memory_retrieval` (memory)   | POST   | `{ "query": "Almaty", "k": 3 }`           | `{ "context": "…" }`  |
| `/memory_retrieval/add`        | POST   | `{ "user": "…", "assistant": "…" }`       | `{ "status": "ok" }`  |
| `/healthz` (both services)     | GET    | —                                         | `{ "status": "up" }`  |


## 3 · Environment variables

All have sane defaults; override in .env or the shell.

| Var                | Default                      | Description                                |
|--------------------|------------------------------|--------------------------------------------|
| `OLLAMA_MODEL`     | `mistral:7b-instruct-q4_K_M` | Any Ollama tag that fits your GPU          |
| `OPENAI_API_KEY`   | — (blank)                    | If set, OpenAI GPT-3.5 is used             |
| `MEMORY_PERSIST_DIR` | `./chroma_store`           | Folder for Chroma DB                       |
| `BUFFER_SIZE`      | `6`                          | Number of turns kept verbatim in RAM       |
| `SUMMARY_EVERY`    | `40`                         | Summarise buffer every *N* turns           |


## 4 · Project layout

src/
├─ chatbot_api.py        # prompt builder + LLM client
├─ memory_service.py     # REST wrapper around MemoryManager
├─ memory.py             # buffer + vector DB + summariser
├─ utils.py              # device pick + T5 summariser
Dockerfile               # single Python image (chatbot & memory)
docker-compose.yml       # adds Ollama GPU container + volumes
requirements.txt
.env.example


## 5 · How it works (short version)

/predict receives the user message.

The chatbot calls memory → gets snippets via MiniLM similarity search.

Builds the prompt: instruction + memory (if any) + user.

Sends to LLM:

Uses OpenAI if OPENAI_API_KEY is present

Otherwise uses Ollama; if the model is missing, it auto-pulls, then retries

Returns the answer and async-posts the turn to memory.

Memory saves to buffer & Chroma; every N turns it summarises with T5.


## 6 · Dev loop & tests

# unit tests (logic only, no GPU)
pytest -q

# rebuild code layers after edits
docker compose build chatbot memory
docker compose up -d


## 7 · Troubleshooting

| Symptom                     | Fix                                                                                          |
|----------------------------|----------------------------------------------------------------------------------------------|
| First request takes minutes | Ollama is downloading the model – watch `docker compose logs -f ollama`. Only happens once.  |
| GPU < 8 GB                  | `OLLAMA_MODEL=phi3:mini` (~1.8 GB)                                                          |
| CPU-only machine            | Remove the `deploy.resources` GPU blocks in the compose and use a CPU model (`mistral:7b-instruct`). |

