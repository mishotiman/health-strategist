"""Retrieval: embed a question and find the closest chunks in pgvector.

This is the shared retrieval layer used by both /search (retrieval only) and
/ask (retrieval + generation). A new DB connection per call is fine at this
scale; we'll pool later if it matters.
"""

from __future__ import annotations

import os

import psycopg
import voyageai

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://phs:phs@db:5432/phs")

_voyage = voyageai.Client()  # reads VOYAGE_API_KEY


def embed_query(text: str) -> list[float]:
    return _voyage.embed([text], model="voyage-3", input_type="query").embeddings[0]


def retrieve(question: str, k: int = 6) -> list[dict]:
    """Return the top-k chunks most similar to the question, with their paper
    metadata and a cosine-similarity score (1.0 = identical direction)."""
    qvec = str(embed_query(question))  # pgvector accepts the '[...]' text form
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.content, c.page,
                   d.title, d.authors, d.year, d.source,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, k),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
