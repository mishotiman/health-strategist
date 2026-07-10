-- Runs once on first `docker compose up` (empty volume) via the postgres init hook.
-- pgvector + the full PHS schema. voyage-3 embeddings = 1024 dimensions.

CREATE EXTENSION IF NOT EXISTS vector;

-- Users & profile ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id         BIGSERIAL PRIMARY KEY,
    email      TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS profiles (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id) ON DELETE CASCADE,
    goals       TEXT,
    constraints TEXT,
    injuries    TEXT,
    history     TEXT,
    sex         TEXT,
    birth_year  INT,
    height_cm   NUMERIC,
    weight_kg   NUMERIC,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- WHOOP OAuth tokens, one connection per user.
CREATE TABLE IF NOT EXISTS whoop_connections (
    user_id       BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    access_token  TEXT NOT NULL,
    refresh_token TEXT,
    expires_at    TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Normalized wearable / bloodwork metrics ("any source, one schema") ------
CREATE TABLE IF NOT EXISTS health_metrics (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,          -- whoop | garmin | bloodwork | manual
    metric_date DATE NOT NULL,
    metric_type TEXT NOT NULL,          -- hrv | rhr | sleep_hours | vo2max | ...
    value       DOUBLE PRECISION,
    unit        TEXT
);

-- Idempotent ingestion: one row per (user, source, date, metric_type).
CREATE UNIQUE INDEX IF NOT EXISTS uq_health_metrics
    ON health_metrics (user_id, source, metric_date, metric_type);

-- Knowledge corpus --------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id       BIGSERIAL PRIMARY KEY,
    title    TEXT,
    source   TEXT,                      -- journal / DOI / URL for citation
    authors  TEXT,
    year     INT,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT,
    content     TEXT NOT NULL,
    page        INT,
    embedding   vector(1024)
);

-- Conversation memory -----------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(id) ON DELETE CASCADE,
    role       TEXT,                    -- user | assistant | tool
    content    TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Note: add an ANN index (ivfflat/hnsw) on chunks.embedding once real data
-- is loaded — building it on an empty table is pointless.
