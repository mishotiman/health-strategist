"""Retrieval: embed a question and find the closest chunks in pgvector.

This is the shared retrieval layer used by both /search (retrieval only) and
/ask (retrieval + generation). A new DB connection per call is fine at this
scale; we'll pool later if it matters.
"""

from __future__ import annotations

import os

import psycopg
import voyageai

from app.config import settings

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://phs:phs@db:5432/phs")

# Optional HNSW recall knob (higher = better recall, slower). Unset -> pgvector's
# default. Worth setting once the ANN index exists (scripts/index_corpus.sql).
_EF_SEARCH = os.environ.get("RAG_HNSW_EF_SEARCH")

_voyage = voyageai.Client()  # reads VOYAGE_API_KEY


def embed_query(text: str) -> list[float]:
    # Must match the corpus embeddings (scripts/embed_chunks.py): same model + dims,
    # sourced from app.config so the two can never silently drift apart.
    return _voyage.embed([text], model=settings.embedding_model, input_type="query",
                         output_dimension=settings.embedding_dim).embeddings[0]


def retrieve(question: str, k: int = 6, fetch_k: int | None = None) -> list[dict]:
    """Return the top-k chunks most similar to the question, with their paper
    metadata and a cosine-similarity score (1.0 = identical direction).

    `fetch_k` is how many candidates to pull from the DB before trimming to `k`.
    For now they're equal; a future reranker will fetch wide (fetch_k >> k) and
    reorder down to k — that's the hook the deferred quality pass plugs into.
    """
    fetch_k = fetch_k or k
    qvec = str(embed_query(question))  # pgvector accepts the '[...]' text form
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        if _EF_SEARCH:  # int() guards the interpolation (SET rejects bound params)
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(_EF_SEARCH)}")
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
            (qvec, qvec, fetch_k),
        )
        columns = [desc[0] for desc in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return rows[:k]
