-- Record when a provider's data was last pulled, so the UI can show
-- "Synced 3 minutes ago" under the Connect/Disconnect button.
--
-- This is deliberately NOT `updated_at`: that column only moves when tokens are
-- written (connect + token refresh), so it says nothing about data freshness.
-- Idempotent; safe to re-run. Fresh installs get it from init_db.sql instead.
ALTER TABLE provider_connections ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ;
