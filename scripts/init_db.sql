-- Runs once on first `docker compose up` (empty volume) via the postgres init hook.
-- pgvector + the full PHS schema. voyage-3.5 embeddings = 512 dimensions.

CREATE EXTENSION IF NOT EXISTS vector;

-- Users & profile ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    email         TEXT UNIQUE,
    display_name  TEXT,                      -- shown in the UI ("Demo User" / real name)
    is_demo       BOOLEAN NOT NULL DEFAULT false,  -- the read-only sample account (guest mode)
    password_hash TEXT,                      -- argon2id; NULL for OAuth-only accounts
    email_verified BOOLEAN NOT NULL DEFAULT false,
    onboarded_at  TIMESTAMPTZ,               -- set when the onboarding flow completes
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Login treats email case-insensitively, so uniqueness must too.
CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_lower ON users (lower(email));

-- Accounts & authentication (see scripts/migrate_auth.sql for the notes) ---
CREATE TABLE IF NOT EXISTS user_identities (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(id) ON DELETE CASCADE,
    provider   TEXT NOT NULL,            -- google | microsoft
    subject    TEXT NOT NULL,            -- the provider's stable user id ("sub")
    email      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_identities ON user_identities (provider, subject);

-- Revocable server-side sessions; only the token's hash is stored.
CREATE TABLE IF NOT EXISTS sessions (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sessions_token ON sessions (token_hash);
CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions (user_id);

-- One-time email tokens (verify address / reset password), hashed + expiring.
CREATE TABLE IF NOT EXISTS auth_tokens (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,            -- verify_email | reset_password
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at    TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_auth_tokens_hash ON auth_tokens (token_hash);
CREATE INDEX IF NOT EXISTS ix_auth_tokens_user ON auth_tokens (user_id, kind);

-- In-flight OIDC logins: CSRF state + replay nonce + PKCE verifier.
CREATE TABLE IF NOT EXISTS oauth_login_states (
    state         TEXT PRIMARY KEY,
    provider      TEXT NOT NULL,
    nonce         TEXT,
    code_verifier TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Login / password-reset throttling.
CREATE TABLE IF NOT EXISTS login_attempts (
    id           BIGSERIAL PRIMARY KEY,
    email        TEXT,
    ip           TEXT,
    success      BOOLEAN NOT NULL DEFAULT false,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_login_attempts_email ON login_attempts (email, attempted_at DESC);
CREATE INDEX IF NOT EXISTS ix_login_attempts_ip    ON login_attempts (ip, attempted_at DESC);

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

-- OAuth tokens for every connected data provider (WHOOP, Garmin, Oura, …).
-- One row per (user, provider); a provider account can own many devices, so
-- devices never need modelling here. See scripts/migrate_providers.sql.
CREATE TABLE IF NOT EXISTS provider_connections (
    id               BIGSERIAL PRIMARY KEY,
    user_id          BIGINT REFERENCES users(id) ON DELETE CASCADE,
    provider         TEXT NOT NULL,          -- whoop | garmin | oura | ...
    external_user_id TEXT,                   -- the provider's own account id
    access_token     TEXT NOT NULL,
    refresh_token    TEXT,
    expires_at       TIMESTAMPTZ,
    connected_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_synced_at   TIMESTAMPTZ,            -- last *successful* data pull (freshness)
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_provider_connections
    ON provider_connections (user_id, provider);

-- An authorization awaiting the user's decision because it belongs to a
-- different account at the provider than the one already connected.
CREATE TABLE IF NOT EXISTS pending_connections (
    user_id          BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider         TEXT NOT NULL,
    external_user_id TEXT,
    access_token     TEXT NOT NULL,
    refresh_token    TEXT,
    expires_at       TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, provider)
);

-- Pending OAuth CSRF tokens (one in-flight authorization per user). The token
-- is verified on /whoop/callback and deleted on use.
CREATE TABLE IF NOT EXISTS whoop_oauth_states (
    user_id     BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    state_token TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Uploaded bloodwork reports (metadata; the values land in health_metrics).
CREATE TABLE IF NOT EXISTS bloodwork_documents (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    report_date   DATE,
    metrics_count INT NOT NULL DEFAULT 0,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Normalized wearable / bloodwork metrics ("any source, one schema") ------
CREATE TABLE IF NOT EXISTS health_metrics (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,          -- whoop | garmin | bloodwork | manual
    metric_date DATE NOT NULL,
    metric_type TEXT NOT NULL,          -- hrv | rhr | sleep_hours | vo2max | ...
    value       DOUBLE PRECISION,       -- numeric result
    unit        TEXT,
    text_value  TEXT                    -- qualitative result (e.g. microbiology: negative/positive)
);

-- Idempotent ingestion: one row per (user, source, date, metric_type).
CREATE UNIQUE INDEX IF NOT EXISTS uq_health_metrics
    ON health_metrics (user_id, source, metric_date, metric_type);

-- Workouts: logged training activities (per-activity events, so NOT the
-- one-value-per-day health_metrics shape). See scripts/migrate_workouts.sql.
CREATE TABLE IF NOT EXISTS workouts (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT REFERENCES users(id) ON DELETE CASCADE,
    source        TEXT NOT NULL DEFAULT 'whoop',   -- whoop | garmin | manual | ...
    external_id   TEXT,                             -- the source's workout id (idempotency)
    sport         TEXT,                             -- e.g. running, weightlifting
    workout_date  DATE,
    start_time    TIMESTAMPTZ,
    end_time      TIMESTAMPTZ,
    duration_min  DOUBLE PRECISION,
    strain        DOUBLE PRECISION,                 -- WHOOP strain (0–21)
    avg_hr        INT,
    max_hr        INT,
    calories      DOUBLE PRECISION,                 -- kcal (converted from kilojoules)
    distance_m    DOUBLE PRECISION,                 -- cardio only; NULL for e.g. lifting
    tz_offset     TEXT,                             -- local UTC offset at the session, e.g. "+03:00"
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_workouts
    ON workouts (user_id, source, external_id);

-- Knowledge corpus --------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id       BIGSERIAL PRIMARY KEY,
    title    TEXT,
    source   TEXT,                      -- journal / DOI / URL for citation (natural key)
    authors  TEXT,
    year     INT,
    pmcid    TEXT,                      -- Europe PMC id (bulk corpus); NULL for hand-picked
    doi      TEXT,
    license  TEXT,                      -- e.g. "cc by" — from the fetch sidecar
    pillar   TEXT,                      -- corpus topic tag (see data/corpus_topics.yml)
    added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent corpus ingestion keys on `source` (always present, unlike pmcid);
-- the topic/year indexes support the deferred metadata-filtering retrieval pass.
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS ix_documents_pillar ON documents (pillar);
CREATE INDEX IF NOT EXISTS ix_documents_year   ON documents (year);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT,
    content     TEXT NOT NULL,
    page        INT,
    embedding   vector(512)
);

-- Conversation memory -----------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(id) ON DELETE CASCADE,
    role       TEXT,                    -- user | assistant | tool
    content    TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Note: the ANN index (HNSW) on chunks.embedding is built separately, after the
-- corpus is embedded — see scripts/index_corpus.sql (building it on an empty
-- table is pointless).
