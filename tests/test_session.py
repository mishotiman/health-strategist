"""Server-side sessions: a login must be revocable, expiring, and never stored
in a form that a database leak could replay."""

from app import session as sess
from app.db import get_connection


def test_create_then_read_returns_the_user(temp_user):
    token = sess.create(temp_user)
    assert sess.read(token) == temp_user


def test_missing_or_unknown_tokens_are_rejected(temp_user):
    assert sess.read(None) is None
    assert sess.read("") is None
    assert sess.read("not-a-real-token") is None


def test_revoke_ends_that_session(temp_user):
    token = sess.create(temp_user)
    sess.revoke(token)
    assert sess.read(token) is None            # the old cookie is dead server-side


def test_revoke_all_ends_every_session(temp_user):
    first, second = sess.create(temp_user), sess.create(temp_user)
    assert sess.revoke_all(temp_user) >= 2
    assert sess.read(first) is None
    assert sess.read(second) is None


def test_expired_session_is_rejected(temp_user):
    token = sess.create(temp_user)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE sessions SET expires_at = now() - INTERVAL '1 day' "
                    "WHERE user_id = %s", (temp_user,))
        conn.commit()
    assert sess.read(token) is None


def test_only_a_hash_of_the_token_is_stored(temp_user):
    token = sess.create(temp_user)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT token_hash FROM sessions WHERE user_id = %s", (temp_user,))
        stored = cur.fetchone()[0]
    assert stored != token           # a leaked database yields no usable cookies
