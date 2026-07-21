"""One-time email tokens (verify address / reset password) must be single-use,
expiring, kind-specific, and stored only as a hash."""

from app import auth
from app.db import get_connection


def test_verification_token_roundtrip(temp_user):
    raw = auth.mint_token(temp_user, auth.VERIFY_EMAIL)
    assert auth.consume_token(raw, auth.VERIFY_EMAIL) == temp_user


def test_token_cannot_be_used_twice(temp_user):
    raw = auth.mint_token(temp_user, auth.VERIFY_EMAIL)
    assert auth.consume_token(raw, auth.VERIFY_EMAIL) == temp_user
    assert auth.consume_token(raw, auth.VERIFY_EMAIL) is None      # burned on first use


def test_a_verify_token_cannot_reset_a_password(temp_user):
    # otherwise an old verification link would be a password-reset link
    raw = auth.mint_token(temp_user, auth.VERIFY_EMAIL)
    assert auth.consume_token(raw, auth.RESET_PASSWORD) is None


def test_expired_token_is_rejected(temp_user):
    raw = auth.mint_token(temp_user, auth.RESET_PASSWORD)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE auth_tokens SET expires_at = now() - INTERVAL '1 hour' "
                    "WHERE user_id = %s", (temp_user,))
        conn.commit()
    assert auth.consume_token(raw, auth.RESET_PASSWORD) is None


def test_garbage_tokens_are_rejected(temp_user):
    assert auth.consume_token("", auth.VERIFY_EMAIL) is None
    assert auth.consume_token("nope", auth.VERIFY_EMAIL) is None


def test_only_a_hash_of_the_token_is_stored(temp_user):
    raw = auth.mint_token(temp_user, auth.VERIFY_EMAIL)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT token_hash FROM auth_tokens WHERE user_id = %s", (temp_user,))
        stored = cur.fetchone()[0]
    assert stored != raw        # the database never holds a working link
