"""Personal Health Strategist — FastAPI app.

Endpoints:
  GET  /health    — liveness
  GET  /db-check  — DB reachable + pgvector present
  POST /search    — retrieval only (no LLM; works without an Anthropic key)
  POST /ask       — retrieval + Claude answer grounded in the passages, with [n] citations
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.db import get_connection
from app.qa import answer_question
from app.rag import retrieve

app = FastAPI(title="Personal Health Strategist")


class AskRequest(BaseModel):
    question: str
    k: int = 6


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        has_vector = cur.fetchone() is not None
    return {"db": "reachable", "pgvector": has_vector}


@app.post("/search")
def search(req: AskRequest):
    """Retrieval only — inspect what the RAG layer returns for a question."""
    chunks = retrieve(req.question, req.k)
    return {
        "question": req.question,
        "results": [
            {
                "n": i + 1,
                "similarity": round(c["similarity"], 3),
                "title": c["title"],
                "source": c["source"],
                "preview": " ".join(c["content"].split())[:200],
            }
            for i, c in enumerate(chunks)
        ],
    }


@app.post("/ask")
def ask(req: AskRequest):
    """Retrieve passages, then have Claude answer grounded in them with citations."""
    result = answer_question(req.question, req.k)
    chunks = result["chunks"]
    sources = [
        {
            "n": i + 1,
            "title": c["title"],
            "authors": c["authors"],
            "year": c["year"],
            "source": c["source"],
            "page": c["page"],
        }
        for i, c in enumerate(chunks)
    ]
    return {"question": req.question, "answer": result["answer"], "sources": sources}
