"""Signed session cookie — the login mechanism.

When a user completes WHOOP OAuth we hand their browser a cookie containing
their user id plus an HMAC signature made with a server-only secret. On later
requests we recompute the signature; if it matches, we trust the id. A browser
can read the cookie but can't forge a valid signature without the secret, so it
can't impersonate another user. No cookie -> the caller is treated as the demo
user (see app.main.current_user_id).
"""

from __future__ import annotations

import hashlib
import hmac
import os

COOKIE_NAME = "phs_session"

# Must be a stable, secret value in production (set SESSION_SECRET). The dev
# fallback keeps local runs working; it only means sessions reset if the secret
# changes, never a security hole locally.
_SECRET = os.environ.get("SESSION_SECRET", "dev-insecure-secret-change-me").encode()


def _sign(user_id: int) -> str:
    return hmac.new(_SECRET, str(user_id).encode(), hashlib.sha256).hexdigest()


def make_token(user_id: int) -> str:
    """Cookie value binding a user id to its signature."""
    return f"{user_id}.{_sign(user_id)}"


def read_token(token: str | None) -> int | None:
    """Return the user id iff the token is present and the signature is valid."""
    if not token:
        return None
    user_part, _, sig = token.partition(".")
    if not user_part.isdigit() or not sig:
        return None
    expected = _sign(int(user_part))
    if not hmac.compare_digest(sig, expected):  # constant-time; resists tampering
        return None
    return int(user_part)
