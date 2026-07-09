"""Embed all un-embedded chunks and write the vectors into chunks.embedding.

Run inside the api container:

    docker compose exec -T api python scripts/embed_chunks.py

Primary: Voyage `voyage-3` (1024-dim). Fallback: OpenAI
`text-embedding-3-large` requested at 1024 dims (to match the vector column).
Idempotent — only embeds rows where embedding IS NULL, so it's safe to re-run
and resumes where it left off.
"""

from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://phs:phs@db:5432/phs")
BATCH = 128


def get_embedder():
    """Return (embed_fn, model_name). Prefer Voyage, fall back to OpenAI."""
    if os.environ.get("VOYAGE_API_KEY"):
        import voyageai
        client = voyageai.Client()  # reads VOYAGE_API_KEY

        def embed(texts: list[str]) -> list[list[float]]:
            return client.embed(texts, model="voyage-3", input_type="document").embeddings

        return embed, "voyage-3"

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()

        def embed(texts: list[str]) -> list[list[float]]:
            resp = client.embeddings.create(
                model="text-embedding-3-large", input=texts, dimensions=1024
            )
            return [d.embedding for d in resp.data]

        return embed, "text-embedding-3-large (1024d)"

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
