-- Account identity for the demo-user / login feature.
ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT false;

-- Uploaded bloodwork reports (metadata; values live in health_metrics).
CREATE TABLE IF NOT EXISTS bloodwork_documents (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    report_date   DATE,
    metrics_count INT NOT NULL DEFAULT 0,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Durable observations the agent learns about the user over conversations (item 4).
CREATE TABLE IF NOT EXISTS profile_notes (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(id) ON DELETE CASCADE,
    note       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Pending WHOOP OAuth CSRF tokens — verified and consumed on /whoop/callback.
CREATE TABLE IF NOT EXISTS whoop_oauth_states (
    user_id     BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    state_token TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
