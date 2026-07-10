"""Personal Health Strategist — FastAPI app.

Endpoints:
  GET  /health              — liveness
  GET  /db-check            — DB reachable + pgvector present
  POST /search              — retrieval only (no LLM)
  POST /ask                 — retrieval + grounded, cited Claude answer
  POST /users               — onboard: create a user + profile
  POST /metrics             — ingest normalized health metrics (any source)
  GET  /metrics/{user_id}   — query a user's metrics
  GET  /whoop/connect       — start WHOOP OAuth (redirects to WHOOP)
  GET  /whoop/callback      — OAuth return: store tokens + sync data
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app import whoop
from app.db import get_connection
from app.ingestion import query_metrics, upsert_metrics
from app.qa import answer_question
from app.rag import retrieve

app = FastAPI(title="Personal Health Strategist")


# ---- request models --------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    k: int = 6


class UserRequest(BaseModel):
    email: str
    goals: str | None = None
    sex: str | None = None
    birth_year: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    injuries: str | None = None


class MetricRecord(BaseModel):
    date: str            # YYYY-MM-DD
    metric_type: str     # canonical (see app.ingestion.CANONICAL_UNITS)
    value: float
    unit: str | None = None


class MetricsRequest(BaseModel):
    user_id: int
    source: str
    records: list[MetricRecord]


# ---- basics ----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        has_vector = cur.fetchone() is not None
    return {"db": "reachable", "pgvector": has_vector}


# ---- RAG -------------------------------------------------------------------
@app.post("/search")
def search(req: AskRequest):
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
    result = answer_question(req.question, req.k)
    sources = [
        {"n": i + 1, "title": c["title"], "authors": c["authors"],
         "year": c["year"], "source": c["source"], "page": c["page"]}
        for i, c in enumerate(result["chunks"])
    ]
    return {"question": req.question, "answer": result["answer"], "sources": sources}


# ---- onboarding + ingestion ------------------------------------------------
@app.post("/users")
def create_user(req: UserRequest):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (email) VALUES (%s) "
            "ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email RETURNING id",
            (req.email,),
        )
        user_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO profiles (user_id, goals, sex, birth_year, height_cm, weight_kg, injuries)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, req.goals, req.sex, req.birth_year,
             req.height_cm, req.weight_kg, req.injuries),
        )
        conn.commit()
    return {"user_id": user_id}


@app.post("/metrics")
def ingest_metrics(req: MetricsRequest):
    written = upsert_metrics(
        req.user_id, req.source,
        [r.model_dump() for r in req.records],
    )
    return {"user_id": req.user_id, "source": req.source, "metrics_written": written}


@app.get("/metrics/{user_id}")
def get_metrics(user_id: int, type: str | None = None, limit: int = 100):
    return {"user_id": user_id, "metrics": query_metrics(user_id, type, limit)}


# ---- WHOOP OAuth -----------------------------------------------------------
@app.get("/whoop/connect")
def whoop_connect(user_id: int):
    # For the MVP, state carries the user id. A production app would use a
    # random, signed state stored server-side to prevent CSRF.
    return RedirectResponse(whoop.authorize_url(state=str(user_id)))


@app.get("/whoop/callback")
def whoop_callback(code: str, state: str):
    user_id = int(state)
    return whoop.connect_and_sync(user_id, code)
