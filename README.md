# Agentic RAG and Evaluation Platform

I created this project to learn more about LangGraph and Qdrant and how everything works together in a real agentic RAG system.

>  This is a learning project and not production-ready code.

- **Multi-step agentic AI workflows** using **LangGraph** (plan → retrieve → verify → answer)
- **Embeddings-based retrieval** using **Qdrant** vector store
- **Evaluation pipelines** for retrieval quality + basic agent performance
- **REST API** using **FastAPI**
- **Dockerized** deployment for Linux/Unix environments

---

## Architecture

**LangGraph workflow**

1. **Planner**: creates a short plan from the question
2. **Retriever**: calls vector search (Qdrant) to fetch top-k chunks
3. **Verifier**: checks if retrieved context is sufficient (1 retry max)
4. **Answer**: generates answer using only retrieved context

---

## Quickstart (Docker)

### 1) Setup

```bash
touch .env
# If you want real LLM calls:
# set LLM_MODE=openai and add OPENAI_API_KEY in .env

pip install -r requirements.txt
```

Sample `.env` file:

```env
# LLM provider options:
# Option A: use OpenAI (recommended if you have a key)
OPENAI_API_KEY=<YOUR-API-KEY>

# Option B: no key mode (uses a simple local "stub" responder)
LLM_MODE=stub  # or openai
OPENAI_MODEL=gpt-4o-mini

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=docs

# Retrieval
TOP_K=6
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 2) Start services

```bash
docker compose up --build
```

### 3) Ingest sample docs

```bash
docker compose exec api python scripts/ingest.py --path data/docs/sample_docs.jsonl
```

### 4) Ask the agent

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is LangGraph used for?"}' | jq
```

### 5) Run evaluation

```bash
docker compose exec api python scripts/run_eval.py --path data/eval/eval_set.jsonl
```

---

## API

### GET /health

Returns service health.

### POST /ask

Request:

```json
{ "question": "..." }
```

Response includes:

- plan
- retrieved chunks (+ scores)
- verification result
- final answer

---

## Evaluation

The evaluation script computes:

- **Recall@K**, **MRR**, **nDCG@K** for retrieval
- Basic **AgentSuccess** metric

Eval file format (jsonl):

```json
{"question":"...", "gold_chunk_ids":["source:chunkIndex", "..."]}
```

---

## Notes / Extensions

Ideas if you want to extend:

- Add query rewriting before retry retrieval
- Add reranking (cross-encoder)
- Add structured tool tracing + latency/cost metrics
- Add a UI (Streamlit) on top of /ask

---

## Tech Stack

Python, LangGraph, FastAPI, Qdrant, SentenceTransformers, Docker
