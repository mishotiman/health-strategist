-- Build the ANN (approximate-nearest-neighbour) index on the embeddings.
-- Run this ONCE, AFTER embed_chunks.py has populated chunks.embedding — building
-- on a filled column is faster and yields a better graph than letting incremental
-- inserts build it. Re-running is a no-op (IF NOT EXISTS).
--
-- HNSW with vector_cosine_ops matches the cosine operator (<=>) used in
-- app/rag.py — they must agree or the index is bypassed. CONCURRENTLY avoids
-- locking the table while the live app serves reads; drop it for a faster build
-- on a fresh/offline DB.
--
-- maintenance_work_mem drives build speed; keep it modest on a small Burstable
-- tier (raise it on a bigger box). If RAM ever gets tight as the corpus grows,
-- Azure Database for PostgreSQL's DiskANN index is the low-memory alternative.

SET maintenance_work_mem = '512MB';

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
