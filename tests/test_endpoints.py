"""Endpoint-level tests through the real FastAPI app (TestClient).

These cover the security-critical glue the pure unit tests can't see: the auth
dependencies, guest write-rejection, per-user data scoping, chat thread
namespacing, and the spend guards. They need Postgres (the `db` fixture skips
when the stack isn't running) but no network — the agent and any email sending
are monkeypatched or no-op offline.
"""

from __future__ import annotations

import secrets

import pytest

from app import ratelimit
from app.ratelimit import SlidingWindowLimiter

STRONG_PW = "Str0ng!Passw0rd"


@pytest.fixture
def client(db):
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def second_client(db):
    """A separate browser: its own cookie jar, same app."""
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def cleanup_users(db):
    """Collect emails of accounts a test creates; remove them (and their
    metrics, which don't cascade) afterwards."""
    from app.db import get_connection
    emails: list[str] = []
    yield emails
    if not emails:
        return
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM health_metrics WHERE user_id IN "
            "(SELECT id FROM users WHERE email = ANY(%s))", (emails,))
        cur.execute("DELETE FROM users WHERE email = ANY(%s)", (emails,))
        conn.commit()


@pytest.fixture
def demo_id(db):
    """The demo account guests read; created (and removed) here if the local
    DB doesn't have one seeded."""
    from app.db import get_connection
    from app.deps import demo_user_id
    existing = demo_user_id()
    if existing is not None:
        yield existing
        return
    email = f"pytest-demo-{secrets.token_hex(4)}@example.invalid"
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO users (email, display_name, is_demo) "
                    "VALUES (%s, 'Test Demo', true) RETURNING id", (email,))
        uid = cur.fetchone()[0]
        conn.commit()
    yield uid
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (uid,))
        conn.commit()


def _register(client, cleanup_users, password: str = STRONG_PW):
    email = f"pytest-ep-{secrets.token_hex(5)}@example.invalid"
    cleanup_users.append(email)
    resp = client.post("/auth/register", json={"email": email, "password": password})
    return email, resp


@pytest.fixture
def fresh_limiter(monkeypatch):
    """Isolate rate-limit state per test — the real limiter is process-global."""
    monkeypatch.setattr(ratelimit, "_limiter", SlidingWindowLimiter())


# ---- auth gates ------------------------------------------------------------
def test_anonymous_is_rejected_everywhere_it_matters(client):
    assert client.get("/me").json() == {"kind": "anon"}
    assert client.post("/metrics", json={"source": "manual", "records": []}).status_code == 401
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/metrics/1").status_code == 401


def test_register_rejects_weak_password(client, cleanup_users):
    _, resp = _register(client, cleanup_users, password="short")
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "password_policy"
    assert "length" in detail["failed"]


def test_register_login_logout_flow(client, cleanup_users):
    email, resp = _register(client, cleanup_users)
    assert resp.status_code == 200
    me = client.get("/me").json()
    assert me["kind"] == "user" and me["email"] == email
    assert me["email_verified"] is False   # reminder, not a gate

    assert client.post("/auth/logout").json()["ok"] is True
    assert client.get("/me").json() == {"kind": "anon"}

    bad = client.post("/auth/login", json={"email": email, "password": "Wr0ng!Password"})
    assert bad.status_code == 401
    good = client.post("/auth/login", json={"email": email, "password": STRONG_PW})
    assert good.status_code == 200
    assert client.get("/me").json()["kind"] == "user"


def test_register_duplicate_email_conflicts(client, cleanup_users):
    email, first = _register(client, cleanup_users)
    assert first.status_code == 200
    dup = client.post("/auth/register", json={"email": email, "password": STRONG_PW})
    assert dup.status_code == 409


def test_change_password_revokes_other_sessions(client, second_client, cleanup_users):
    email, _ = _register(client, cleanup_users)
    second_client.post("/auth/login", json={"email": email, "password": STRONG_PW})
    assert second_client.get("/me").json()["kind"] == "user"

    resp = client.post("/auth/change-password",
                       json={"current_password": STRONG_PW,
                             "new_password": STRONG_PW + "2"})
    assert resp.status_code == 200
    # the other browser's stolen-able cookie is dead; this one was re-issued
    assert second_client.get("/me").json() == {"kind": "anon"}
    assert client.get("/me").json()["kind"] == "user"


def test_forgot_password_never_locks_login(client, cleanup_users, fresh_limiter):
    """Regression: reset requests used to write failed login_attempts, so a
    handful of them blocked the address from logging in for 15 minutes."""
    email, _ = _register(client, cleanup_users)
    client.post("/auth/logout")
    for _ in range(10):
        resp = client.post("/auth/forgot-password", json={"email": email})
        assert resp.status_code == 200 and resp.json() == {"ok": True}
    login = client.post("/auth/login", json={"email": email, "password": STRONG_PW})
    assert login.status_code == 200


# ---- data scoping ----------------------------------------------------------
def test_metrics_are_scoped_to_their_owner(client, second_client, cleanup_users, demo_id):
    _, resp = _register(client, cleanup_users)
    owner_id = resp.json()["user_id"]
    write = client.post("/metrics", json={
        "source": "manual",
        "records": [{"date": "2026-07-01", "metric_type": "resting_hr", "value": 55}],
    })
    assert write.status_code == 200 and write.json()["metrics_written"] == 1
    assert client.get(f"/metrics/{owner_id}").status_code == 200

    _register(second_client, cleanup_users)   # a different account
    other = second_client.get(f"/metrics/{owner_id}")
    assert other.status_code == 403           # not yours, not the sample account


def test_guest_reads_demo_but_cannot_write(client, demo_id):
    assert client.post("/auth/guest").status_code == 200
    me = client.get("/me").json()
    assert me["kind"] == "guest" and me["user_id"] == demo_id
    assert me["name"] is None and me["email"] is None   # not an account

    assert client.get(f"/metrics/{demo_id}").status_code == 200
    write = client.post("/metrics", json={
        "source": "manual",
        "records": [{"date": "2026-07-01", "metric_type": "resting_hr", "value": 55}],
    })
    assert write.status_code == 401


# ---- chat: thread namespacing + spend guard --------------------------------
@pytest.fixture
def captured_agent(monkeypatch):
    """Replace the real (Opus) agent with a recorder."""
    calls: list[tuple] = []

    def fake_run(user_id, message, thread_id=None):
        calls.append((user_id, message, thread_id))
        return {"answer": "ok", "thread_id": thread_id,
                "tools_used": [], "sources": []}

    monkeypatch.setattr("app.agent.run", fake_run)
    return calls


def test_chat_threads_are_namespaced_per_user(client, cleanup_users,
                                              captured_agent, fresh_limiter):
    _, resp = _register(client, cleanup_users)
    uid = resp.json()["user_id"]
    client.post("/chat", json={"message": "hi"})
    client.post("/chat", json={"message": "hi", "thread_id": "web-abc"})
    assert captured_agent[0][2] is None               # default: own thread
    assert captured_agent[1][2] == f"{uid}:web-abc"   # client thread, namespaced


def test_guest_chats_never_share_a_default_thread(client, second_client, demo_id,
                                                  captured_agent, fresh_limiter):
    """Every guest is the same demo user, so an omitted thread id must yield a
    unique thread — not one communal conversation."""
    for c in (client, second_client):
        c.post("/auth/guest")
        assert c.post("/chat", json={"message": "hi"}).status_code == 200
    t1, t2 = captured_agent[0][2], captured_agent[1][2]
    assert t1.startswith(f"{demo_id}:guest-") and t2.startswith(f"{demo_id}:guest-")
    assert t1 != t2


def test_guest_chat_is_rate_limited(client, demo_id, captured_agent,
                                    fresh_limiter, monkeypatch):
    monkeypatch.setattr(ratelimit, "CHAT_GUEST", (2, 60))
    client.post("/auth/guest")
    assert client.post("/chat", json={"message": "1"}).status_code == 200
    assert client.post("/chat", json={"message": "2"}).status_code == 200
    blocked = client.post("/chat", json={"message": "3"})
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert len(captured_agent) == 2   # the blocked turn never reached the agent
