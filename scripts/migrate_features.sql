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
