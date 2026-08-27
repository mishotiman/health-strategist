"""Retrieval: embed a question and find the closest chunks in pgvector.

This is the shared retrieval layer used by both /search (retrieval only) and
/ask (retrieval + generation). Connections come from the shared pool in
app.db — this module used to read its own DATABASE_URL, a second source of
truth that could silently drift from app.config.
"""

from __future__ import annotations

import logging
import os

import voyageai

from app.config import settings
from app.db import get_connection

# Optional HNSW recall knob (higher = better recall, slower). Unset -> pgvector's
# default. Worth setting once the ANN index exists (scripts/index_corpus.sql).
log = logging.getLogger(__name__)

_EF_SEARCH = os.environ.get("RAG_HNSW_EF_SEARCH")

_voyage = voyageai.Client()  # reads VOYAGE_API_KEY


def embed_query(text: str) -> list[float]:
    # Must match the corpus embeddings (scripts/embed_chunks.py): same model + dims,
    # sourced from app.config so the two can never silently drift apart.
    return _voyage.embed([text], model=settings.embedding_model, input_type="query",
                         output_dimension=settings.embedding_dim).embeddings[0]


def diversify(rows: list[dict], k: int, per_doc: int) -> list[dict]:
    """Keep the best k passages, allowing at most `per_doc` from any one paper.

    Passages from the same paper sit close together in meaning-space, so a plain
    top-k often stacks several chunks of one document and calls it six sources.
    Capping trades a slightly weaker passage for a different paper, which is the
    right trade when the answer cites its evidence.

    Order is otherwise preserved, so this composes with whatever ranked `rows`.
    If capping cannot fill k (a narrow topic with few papers), the skipped
    passages top it back up rather than returning short.
    """
    if per_doc <= 0:
        return rows[:k]
    kept, overflow, counts = [], [], {}
    for row in rows:
        source = row["source"]
        if counts.get(source, 0) < per_doc:
            counts[source] = counts.get(source, 0) + 1
            kept.append(row)
            if len(kept) == k:
                return kept
        else:
            overflow.append(row)
    return (kept + overflow)[:k]


def rerank(question: str, rows: list[dict], k: int) -> list[dict]:
    """Reorder candidates by true relevance and keep the best k.

    The vector search ranks by bi-encoder cosine: question and passage are
    embedded SEPARATELY and compared, so the score says "these occupy a similar
    region of meaning-space", not "this passage answers this question". That is
    why cosine could not distinguish a covered topic from an uncovered one on
    this corpus (see app/agent.py). A reranker is a cross-encoder: it reads the
    question and the passage TOGETHER and scores actual relevance, which is far
    better at ordering but too slow to run over 275k passages — hence
    fetch-wide-then-rerank.

    Failures are swallowed on purpose: a reranker outage should degrade
    retrieval to plain vector order, not take /chat and /ask down with it.
    """
    if not rows:
        return rows
    try:
        result = _voyage.rerank(question, [r["content"] for r in rows],
                                model=settings.rerank_model, top_k=k)
    except Exception:  # noqa: BLE001 - degrade to vector order, never fail the request
        log.exception("rerank failed; falling back to vector order")
        return rows[:k]
    ordered = []
    for item in result.results:
        row = dict(rows[item.index])
        row["rerank_score"] = item.relevance_score
        ordered.append(row)
    return ordered


def retrieve(question: str, k: int = 6, fetch_k: int | None = None,
             use_rerank: bool | None = None) -> list[dict]:
    """Return the top-k chunks most similar to the question, with their paper
    metadata and a cosine-similarity score (1.0 = identical direction).

    `fetch_k` is how many candidates to pull from the DB before trimming to `k`.
    With reranking on it defaults to settings.rag_fetch_k (>> k): fetch wide
    cheaply, then let the reranker reorder down to k. With it off the two are
    equal and this is a plain vector search, as before.

    `use_rerank` overrides settings.rag_rerank for one call — the eval harness
    uses it to measure both configurations without a restart.
    """
    use_rerank = settings.rag_rerank if use_rerank is None else use_rerank
    # Fetch wider than k whenever a post-filter will trim: the per-document cap
    # needs spare candidates to swap in, and reranking needs a pool to reorder.
    if fetch_k is None:
        widen = use_rerank or settings.rag_max_chunks_per_doc > 0
        fetch_k = settings.rag_fetch_k if widen else k
    qvec = str(embed_query(question))  # pgvector accepts the '[...]' text form
    with get_connection() as conn, conn.cursor() as cur:
        if _EF_SEARCH:  # int() guards the interpolation (SET rejects bound params)
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(_EF_SEARCH)}")
        cur.execute( # sort every chunk by distance to the neighbors, give me the closest ones
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
    # Sample output of retrieve -> list[dict], each dict being a chunk:
    # [
    #   {
    #     "id": 118432,                    # chunk id
    #     "content": "Dosage recommendations and adverse effects ... "  # FULL passage text, not truncated
    #     "page": 14,                      # page in the source PDF
    #     "title": "Amino acids regulating skeletal muscle metabolism: ...",
    #     "authors": "Zhang, Y.; ...",
    #     "year": 2024,
    #     "source": "https://doi.org/10.1186/s12986-024-00820-0",
    #     "similarity": 0.692              # cosine, 1.0 = identical direction
    #   },
    #   { ... },   # chunk 2
    #   { ... },   # chunk 3
    # ]
    # Rerank first (it reorders), then cap per document (it filters) — capping
    # before reranking would discard passages the reranker might have promoted.
    if use_rerank:
        rows = rerank(question, rows, fetch_k)
    return diversify(rows, k, settings.rag_max_chunks_per_doc)
