"""Personal Health Strategist — API entrypoint.

Week 1 scope: prove the stack is alive. /health checks the app,
/db-check confirms Postgres is reachable and pgvector is installed.
"""
from fastapi import FastAPI

from app.db import get_connection

app = FastAPI(title="Personal Health Strategist")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        has_vector = cur.fetchone() is not None
    return {"db": "reachable", "pgvector": has_vector}
