-- Week 3 migration for the existing (already-created) database.
-- Fresh installs get these from init_db.sql instead.

-- Physical profile fields used by onboarding.
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS sex        TEXT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS birth_year INT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS height_cm  NUMERIC;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS weight_kg  NUMERIC;

-- Idempotent ingestion: one row per (user, source, date, metric_type).
CREATE UNIQUE INDEX IF NOT EXISTS uq_health_metrics
    ON health_metrics (user_id, source, metric_date, metric_type);

-- WHOOP OAuth tokens, one connection per user.
CREATE TABLE IF NOT EXISTS whoop_connections (
    user_id       BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    access_token  TEXT NOT NULL,
    refresh_token TEXT,
    expires_at    TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
