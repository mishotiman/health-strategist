"""Embed all un-embedded chunks and write the vectors into chunks.embedding.

Run inside the api container:

    docker compose exec -T api python scripts/embed_chunks.py

Model and dimensions come from app.config (settings.embedding_model /
embedding_dim) so the corpus and the query side (app/rag.py) never drift apart.
Default: Voyage `voyage-3.5` at 512 dims (Matryoshka). Fallback: OpenAI
`text-embedding-3-large` at the same dims. Idempotent — only embeds rows where
embedding IS NULL, so it's safe to re-run and resumes where it left off; provider
rate limits (429) are retried with exponential backoff.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # make `app` importable

import psycopg
from pgvector.psycopg import register_vector
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://phs:phs@db:5432/phs")
BATCH = 128

# Bulk embedding makes thousands of requests and WILL hit provider rate limits
# (429); back off and resume rather than crash the whole run.
_embed_retry = retry(stop=stop_after_attempt(6),
                     wait=wait_exponential(multiplier=1, min=2, max=60))


def get_embedder():
    """Return (embed_fn, label). Prefer Voyage, fall back to OpenAI. Both embed at
    settings.embedding_dim so the vectors match app/rag.py's query embeddings and
    the vector(512) column."""
    model, dim = settings.embedding_model, settings.embedding_dim

    if os.environ.get("VOYAGE_API_KEY"):
        import voyageai
        client = voyageai.Client()  # reads VOYAGE_API_KEY

        @_embed_retry
        def embed(texts: list[str]) -> list[list[float]]:
            return client.embed(texts, model=model, input_type="document",
                                output_dimension=dim).embeddings

        return embed, f"{model} ({dim}d)"

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()

        @_embed_retry
        def embed(texts: list[str]) -> list[list[float]]:
            resp = client.embeddings.create(
                model="text-embedding-3-large", input=texts, dimensions=dim
            )
            return [d.embedding for d in resp.data]

        return embed, f"text-embedding-3-large ({dim}d)"

    raise SystemExit("Set VOYAGE_API_KEY (or OPENAI_API_KEY) in .env first.")


def main() -> None:
    embed, model = get_embedder()
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, content FROM chunks WHERE embedding IS NULL ORDER BY id")
            rows = cur.fetchall()

        if not rows:
            print("Nothing to do — all chunks already embedded.")
            return

        print(f"Embedding {len(rows)} chunks with {model} ...")
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            vectors = embed([content for _, content in batch])
            with conn.cursor() as cur:
                for (chunk_id, _), vector in zip(batch, vectors):
                    cur.execute("UPDATE chunks SET embedding = %s WHERE id = %s",
                                (vector, chunk_id))
            conn.commit()
            print(f"  {min(i + BATCH, len(rows))}/{len(rows)}")

    print("Done.")


if __name__ == "__main__":
    main()
