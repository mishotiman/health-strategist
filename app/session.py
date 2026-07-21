"""Server-side sessions — the login mechanism.

The browser holds a high-entropy random token in an HttpOnly cookie; the
database stores only its SHA-256 hash, so a database leak yields no usable
sessions. Because the session record lives server-side it can be **revoked**:
logging out, changing a password, or "log out everywhere" all invalidate it
immediately.

(The previous implementation signed the user id into the cookie as
``user_id.HMAC(user_id)``. That could never expire or be invalidated — logging
out only deleted the client's copy, and any leaked cookie was valid forever.)

A plain SHA-256 is the right hash here: unlike a password, the token is 32 bytes
of `secrets` randomness, so there is nothing to brute-force and no need for a
slow KDF.

Guests ("Try Health Strategist now") get a separate, unauthenticated marker
cookie. It grants read-only access to the public sample account only, so it
carries no secret and needs no signature.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets

from app.db import get_connection

COOKIE_NAME = "phs_session"
GUEST_COOKIE_NAME = "phs_guest"

SESSION_DAYS = 30


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create(user_id: int) -> str:
    """Open a session and return the raw token to put in the cookie.

    The raw token is returned once and never stored; only its hash is persisted.
    """
    token = secrets.token_urlsafe(32)
    expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=SESSION_DAYS)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (user_id, token_hash, expires_at) VALUES (%s, %s, %s)",
            (user_id, _hash(token), expires_at),
        )
        conn.commit()
    return token


def read(token: str | None) -> int | None:
    """The user id for a live session, or None if missing/expired/revoked."""
    if not token:
        return None
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM sessions "
            "WHERE token_hash = %s AND revoked_at IS NULL AND expires_at > now()",
            (_hash(token),),
        )
        row = cur.fetchone()
    return row[0] if row else None


def revoke(token: str | None) -> None:
    """End one session (logout on this device). Idempotent."""
    if not token:
        return
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE sessions SET revoked_at = now() "
            "WHERE token_hash = %s AND revoked_at IS NULL",
            (_hash(token),),
        )
        conn.commit()


def revoke_all(user_id: int) -> int:
    """End every session for a user — "log out everywhere". Also called after a
    password change or reset, so a stolen cookie stops working."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE sessions SET revoked_at = now() "
            "WHERE user_id = %s AND revoked_at IS NULL",
            (user_id,),
        )
        count = cur.rowcount
        conn.commit()
    return count
