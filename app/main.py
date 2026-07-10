"""Personal Health Strategist — FastAPI app.

Endpoints:
  GET  /health    — liveness
  GET  /db-check  — DB reachable + pgvector present
  POST /search    — retrieval only (no LLM; works without an Anthropic key)
  POST /ask       — retrieval + Claude answer grounded in the passages, with [n] citations
"""

from __future__ import annotations

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

from app.db import get_connection
from app.rag import retrieve

app = FastAPI(title="Personal Health Strategist")

_claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are a science-grounded health strategist.

Answer ONLY using the numbered research passages provided in the user message.
- Cite every claim with [n], where n is the passage number you drew it from.
- If the passages do not cover the question, say so plainly. Do not answer from
  outside knowledge.
- You are not a doctor: never diagnose, and defer to a qualified professional
  for anything that looks like a medical red flag.
Be concise, practical, and honest about uncertainty in the evidence."""


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
    chunks = retrieve(req.question, req.k)

    context = "\n\n".join(
        f"[{i + 1}] {c['content']}\n"
        f"(Source: {c['title']}; {c['authors']}; {c['year']}. {c['source']})"
        for i, c in enumerate(chunks)
    )
    user_message = f"Research passages:\n\n{context}\n\nQuestion: {req.question}"

    message = _claude.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")

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
    return {"question": req.question, "answer": answer, "sources": sources}
