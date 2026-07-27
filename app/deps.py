"""Request-level dependencies and helpers shared by every router.

Who is calling (Principal + the auth gates), the spend guard for the agent
endpoints, and the small helpers tied to the session cookie. Routers import
from here; nothing here imports a router, so there are no cycles.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, Response

from app import connections, ratelimit, session as sess, whoop
from app.config import settings
from app.db import get_connection

# Cookie hardening: Secure only over https, so local http dev still works.
COOKIE_SECURE = settings.app_base_url.startswith("https")


@dataclass
class Principal:
    """Who is making this request.

    user  — signed in with their own account.
    guest — the "Try Health Strategist now" visitor: reads the public sample
            account, may not write anything, and nothing is saved for them.
    anon  — no cookie at all; sent to the login page.
    """
    user_id: int | None
    kind: str  # "user" | "guest" | "anon"

    @property
    def is_user(self) -> bool:
        return self.kind == "user"


def demo_user_id() -> int | None:
    """The read-only sample account that guests see."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE is_demo = true ORDER BY id LIMIT 1")
        row = cur.fetchone()
    return row[0] if row else None


def current_principal(request: Request) -> Principal:
    """A live session wins; otherwise the guest marker; otherwise anonymous."""
    uid = sess.read(request.cookies.get(sess.COOKIE_NAME))
    if uid is not None:
        return Principal(uid, "user")
    if request.cookies.get(sess.GUEST_COOKIE_NAME):
        demo = demo_user_id()
        if demo is not None:
            return Principal(demo, "guest")
    return Principal(None, "anon")


def require_user(p: Principal = Depends(current_principal)) -> int:
    """For every endpoint that writes or touches a real account. Guests get 401."""
    if not p.is_user:
        raise HTTPException(status_code=401, detail="Login required.")
    return p.user_id


def readable_user(p: Principal = Depends(current_principal)) -> int:
    """For read-only endpoints: the signed-in account, or the sample account for
    guests. Anonymous callers are still rejected."""
    if p.user_id is None:
        raise HTTPException(status_code=401, detail="Login required.")
    return p.user_id


def client_ip(request: Request) -> str:
    """The caller's address, for per-IP rate limiting. Behind the Azure ingress
    the socket peer is the proxy, which appends the real client to
    X-Forwarded-For — the LAST entry is the one our proxy wrote (earlier ones
    are client-supplied and spoofable). No header (local dev) -> the socket."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def chat_principal(request: Request, p: Principal = Depends(current_principal)) -> Principal:
    """readable_user plus a spend guard for the agent endpoints — every /chat
    turn runs the Opus agent on the app's own key. Signed-in users get a
    per-account budget; guests (who all share the demo account and cost nothing
    to become) get a tighter one keyed by client address."""
    if p.user_id is None:
        raise HTTPException(status_code=401, detail="Login required.")
    if p.is_user:
        ratelimit.enforce(f"chat:user:{p.user_id}", ratelimit.CHAT_USER, "messages")
    else:
        ratelimit.enforce(f"chat:guest:{client_ip(request)}",
                          ratelimit.CHAT_GUEST, "messages")
    return p


def thread_for(p: Principal, thread_id: str | None) -> str | None:
    """The agent thread this conversation continues, namespaced per user so demo
    and real memory never mix. Guests all share the demo user id, so a guest who
    sends no thread id gets a random one-off thread — the per-user default would
    drop every such guest into one communal (and persisted) conversation."""
    if thread_id:
        return f"{p.user_id}:{thread_id}"
    if not p.is_user:
        return f"{p.user_id}:guest-{secrets.token_hex(8)}"
    return None  # agent.run defaults to the signed-in user's own thread


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        sess.COOKIE_NAME, token,
        httponly=True, samesite="lax", secure=COOKIE_SECURE,
        max_age=60 * 60 * 24 * sess.SESSION_DAYS,
    )


def user_info(user_id: int) -> dict:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, email, is_demo, email_verified, onboarded_at "
                    "FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    if not row:
        return {"user_id": user_id, "name": "Unknown", "is_demo": True}
    uid, name, email, is_demo, verified, onboarded = row
    connected = connections.providers_for(uid)
    whoop_connected = whoop.PROVIDER in connected
    last_synced = None
    if whoop_connected:
        # drives the "Synced 3 minutes ago" note under the WHOOP button
        row = connections.get(uid, whoop.PROVIDER)
        if row and row.get("last_synced_at"):
            last_synced = row["last_synced_at"].isoformat()
    return {"user_id": uid, "is_demo": is_demo, "email": email,
            "connected_providers": connected,
            "name": name or (email.split("@")[0] if email else "Your account"),
            "email_verified": verified, "onboarded": onboarded is not None,
            "whoop_connected": whoop_connected,
            "whoop_last_synced_at": last_synced}
