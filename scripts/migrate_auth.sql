-- Real accounts: password credentials, external sign-in identities, revocable
-- server-side sessions, one-time email tokens, OIDC login state and login
-- throttling. Idempotent — safe to re-run. Fresh installs get this from
-- init_db.sql instead.

-- Password + verification state on the existing users table.
-- password_hash stays NULL for accounts that only sign in with Google/Microsoft.
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash  TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarded_at   TIMESTAMPTZ;

-- Login treats email case-insensitively, so uniqueness must too.
CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_lower ON users (lower(email));

-- External sign-in identities. One account can link several providers.
CREATE TABLE IF NOT EXISTS user_identities (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(id) ON DELETE CASCADE,
    provider   TEXT NOT NULL,            -- google | microsoft
    subject    TEXT NOT NULL,            -- the provider's stable user id ("sub")
    email      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_identities ON user_identities (provider, subject);

-- Server-side sessions, so a login can actually be revoked. The cookie carries a
-- random token; only its hash is stored, so a database leak yields no usable
-- sessions.
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

-- One-time, single-use email tokens (verify address / reset password), stored
-- hashed and with an expiry.
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

-- In-flight OIDC logins: CSRF state + replay nonce + PKCE verifier. Generalizes
-- the WHOOP-only whoop_oauth_states table to any provider.
CREATE TABLE IF NOT EXISTS oauth_login_states (
    state         TEXT PRIMARY KEY,
    provider      TEXT NOT NULL,         -- google | microsoft
    nonce         TEXT,
    code_verifier TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Login / password-reset throttling (brute-force resistance).
CREATE TABLE IF NOT EXISTS login_attempts (
    id           BIGSERIAL PRIMARY KEY,
    email        TEXT,
    ip           TEXT,
    success      BOOLEAN NOT NULL DEFAULT false,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_login_attempts_email ON login_attempts (email, attempted_at DESC);
CREATE INDEX IF NOT EXISTS ix_login_attempts_ip    ON login_attempts (ip, attempted_at DESC);
