-- Generalize the WHOOP-only connection table into one table for every data
-- provider (WHOOP, Garmin, Oura, …), and start recording *which account* at the
-- provider the tokens belong to. Idempotent — safe to re-run.

CREATE TABLE IF NOT EXISTS provider_connections (
    id               BIGSERIAL PRIMARY KEY,
    user_id          BIGINT REFERENCES users(id) ON DELETE CASCADE,
    provider         TEXT NOT NULL,          -- whoop | garmin | oura | ...
    external_user_id TEXT,                   -- the provider's own account id
    access_token     TEXT NOT NULL,
    refresh_token    TEXT,
    expires_at       TIMESTAMPTZ,
    connected_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One connection per provider per user (a WHOOP account can own many devices,
-- so devices never need modelling here).
CREATE UNIQUE INDEX IF NOT EXISTS uq_provider_connections
    ON provider_connections (user_id, provider);

-- A provider authorization parked awaiting the user's decision, because it
-- belongs to a DIFFERENT account at the provider than the one already connected.
-- Accepting it replaces the stored data for that provider; declining drops the row.
-- Kept apart from provider_connections so a half-decided authorization can never
-- be mistaken for a live connection.
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

-- Carry over any existing WHOOP connection, then retire the old table. Guarded
-- so a second run is a no-op once whoop_connections is gone.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'public' AND table_name = 'whoop_connections') THEN
        INSERT INTO provider_connections
            (user_id, provider, access_token, refresh_token, expires_at, updated_at)
        SELECT user_id, 'whoop', access_token, refresh_token, expires_at, updated_at
        FROM whoop_connections
        ON CONFLICT (user_id, provider) DO NOTHING;

        DROP TABLE whoop_connections;
    END IF;
END $$;
