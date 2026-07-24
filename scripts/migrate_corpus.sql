-- Corpus scale-up: prepare the schema for a curated ~3–5k-paper knowledge base.
-- Idempotent; safe to re-run. Run this BEFORE ingest/embed. The ANN index is
-- built separately, AFTER embedding — see scripts/index_corpus.sql.

-- 1. Embeddings shrink to 512 dims (voyage-3.5 Matryoshka) — ~half the storage
--    and RAM of 1024, with negligible quality loss. Old 1024-dim vectors can't be
--    cast down, so we null them out (embed_chunks.py refills; it only embeds
--    NULLs). The type guard makes a re-run a no-op, so embeddings are never wiped
--    a second time.
DO $$
DECLARE current_type text;
BEGIN
    SELECT format_type(a.atttypid, a.atttypmod) INTO current_type
    FROM pg_attribute a
    WHERE a.attrelid = 'chunks'::regclass AND a.attname = 'embedding';

    IF current_type IS DISTINCT FROM 'vector(512)' THEN
        UPDATE chunks SET embedding = NULL WHERE embedding IS NOT NULL;
        ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(512);
    END IF;
END $$;

-- 2. Citation + filtering metadata on each document. pillar/license come from the
--    fetch sidecar; pmcid/doi are parsed during extraction. All nullable, so the
--    original hand-picked sample (which has no PMCID) still loads cleanly.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pmcid   TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doi     TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS license TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pillar  TEXT;   -- corpus topic tag

-- 3. Natural key for idempotent ingestion. `source` (DOI / URL) is always present
--    — including for the SportRxiv PDF, which has no PMCID — so it's a more robust
--    conflict target than pmcid. ingest.py uses ON CONFLICT (source) DO NOTHING.
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_source ON documents (source);

-- Cheap metadata indexes for the deferred quality pass (filter by topic/recency).
CREATE INDEX IF NOT EXISTS ix_documents_pillar ON documents (pillar);
CREATE INDEX IF NOT EXISTS ix_documents_year   ON documents (year);
