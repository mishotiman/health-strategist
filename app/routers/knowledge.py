"""The research corpus: retrieval-only /search, cited /ask, and the public
corpus stats badge.

/search and /ask stay public (the production smoke-eval and pre-login demo
depend on that) but are rate-limited per IP: both spend on the app's own
API keys."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app import ratelimit
from app.db import get_connection
from app.deps import client_ip
from app.qa import answer_question
from app.rag import retrieve

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    k: int = 6


@router.post("/search")
def search(req: AskRequest, request: Request):
    ratelimit.enforce(f"search:{client_ip(request)}", ratelimit.SEARCH_IP, "searches")
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


@router.post("/ask")
def ask(req: AskRequest, request: Request):
    ratelimit.enforce(f"ask:{client_ip(request)}", ratelimit.ASK_IP, "questions")
    result = answer_question(req.question, req.k)
    sources = [
        {"n": i + 1, "title": c["title"], "authors": c["authors"],
         "year": c["year"], "source": c["source"], "page": c["page"]}
        for i, c in enumerate(result["chunks"])
    ]
    return {"question": req.question, "answer": result["answer"], "sources": sources}


@router.get("/corpus/stats")
def corpus_stats():
    """How many papers / passages back the RAG — surfaced in the UI header so
    users can see answers are grounded in real literature. Public (no auth)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        documents = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
        passages = cur.fetchone()[0]
    return {"documents": documents, "passages": passages}
