# Personal Health Strategist (PHS)

Agentic RAG app: a LangGraph agent orchestrates a science-grounded RAG core to
turn a user's goals + wearable/bloodwork data into a personalized health
strategy, with ongoing Q&A. Sources are open-access sports-science
meta-analyses, systematic reviews, and position stands.

> Portfolio project for a junior/associate **AI Engineer** role. The
> differentiator is measured retrieval quality + agent evaluation, not features.

## Stack

FastAPI · PostgreSQL + pgvector · Claude (generation) · Voyage `voyage-3`
embeddings (OpenAI `text-embedding-3-large` fallback) · LangGraph + LangChain +
LangSmith · WHOOP OAuth2 · PyMuPDF · thin Next.js UI · Docker Compose.

## Quickstart

```bash
cp .env.example .env      # fill API keys as you reach each step
docker compose up --build # brings up db (pgvector) + api
```

- API: http://localhost:8000
- Health: http://localhost:8000/health
- DB + pgvector check: http://localhost:8000/db-check
- Docs: http://localhost:8000/docs

## Schema

`users` · `profiles` · `health_metrics` (normalized: source, date, type, value,
unit) · `documents` / `chunks` (text + 1024-dim embeddings + citation metadata)
· `messages` (memory). See [scripts/init_db.sql](scripts/init_db.sql).

## Agent tools (planned)

1. `knowledge_search` — RAG, returns passages + citations
2. `health_data` — queries normalized user metrics
3. `memory` — profile/goals/history read-write
4. `guardrail` — no diagnosis, flags red flags, defers to professionals

## Build progress (Week 1 · Jul 7–11)

- [x] **Mon** — repo, Docker Compose, pgvector up; `/db-check` green
- [ ] **Tue** — ingestion v0: PDF → chunks → DB
- [ ] **Wed** — embeddings (Voyage) + similarity search
- [ ] **Thu** — `/ask` endpoint: retrieval → Claude answer with citations
- [ ] **Fri** — polish, README, Week 1 shipped
