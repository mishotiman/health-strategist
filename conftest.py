"""Pytest bootstrap.

Some app modules construct API clients at import time (anthropic, voyage). The
unit tests here only exercise PURE functions and never make a network call, so
we set placeholder keys before collection to let those imports succeed offline.
"""

import os
import secrets

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


def _db_reachable() -> bool:
    try:
        from app.db import get_connection

        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - any failure means "no database here"
        return False


@pytest.fixture(scope="session")
def db():
    """Opt-in to a real database.

    Most of the suite is pure and runs offline. Sessions and one-time tokens,
    though, *are* database behaviour (single-use, expiry, revocation), so those
    tests take this fixture and are skipped when the stack isn't running.
    """
    if not _db_reachable():
        pytest.skip("needs Postgres — run `docker compose up -d db`")


@pytest.fixture
def temp_user(db):
    """A throwaway account, removed afterwards. Child rows (sessions,
    auth_tokens, identities) cascade on delete."""
    from app.db import get_connection

    email = f"pytest-{secrets.token_hex(6)}@example.invalid"
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO users (email) VALUES (%s) RETURNING id", (email,))
        user_id = cur.fetchone()[0]
        conn.commit()
    yield user_id
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
