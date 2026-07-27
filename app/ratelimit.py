"""Rate limiting for the endpoints that spend money on every call.

/chat runs the Opus agent and /ask, /search call Anthropic/Voyage — all on the
app's own keys, and /ask + /search are public by design (the production
smoke-eval and the pre-login demo rely on that). Without a ceiling, one script
hitting the live URL burns API credit without limit; these budgets cap the blast
radius while staying far above any honest use.

A sliding window over recent request timestamps, in process RAM. That is the
right weight here: the app runs as a single replica, and losing counters on a
restart merely resets short windows — an acceptable failure mode for abuse
protection, unlike losing conversations (which is why chat state lives in
Postgres and this doesn't).

Budgets come from app.config (RATE_* env vars) so production can be tuned
without a deploy.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

from fastapi import HTTPException

from app.config import settings

# (limit, window in seconds). Guests all share the demo account, so their chat
# budget is keyed by client address and kept tighter than a signed-in user's.
CHAT_USER = (settings.rate_chat_user, 5 * 60)
CHAT_GUEST = (settings.rate_chat_guest, 15 * 60)
ASK_IP = (settings.rate_ask_ip, 5 * 60)          # smoke-eval sends 5
SEARCH_IP = (settings.rate_search_ip, 5 * 60)

# Password-reset email budgets. These live here, NOT in login_attempts: counting
# reset requests as failed logins let 8 of them lock an address out of login
# for 15 minutes — a denial-of-service anyone could aim at any user.
RESET_EMAIL = (3, 60 * 60)   # per target address: an attacker can't drown an inbox
RESET_IP = (10, 60 * 60)     # per requester address

# Keys whose newest hit is older than this are dropped wholesale during pruning.
# Kept >= the longest window above so pruning can never forget live counts.
_PRUNE_AFTER = 60 * 60
_PRUNE_THRESHOLD = 4096  # only sweep once this many keys have accumulated


class SlidingWindowLimiter:
    """`limit` requests per `window` seconds, per key. Thread-safe."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def try_acquire(self, key: str, limit: int, window_s: float) -> float | None:
        """Record one request if the budget allows it.

        Returns None when allowed, else the seconds until a slot frees up
        (the request is NOT recorded — being told to wait costs nothing).
        """
        now = time.monotonic()
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                q = self._hits[key] = deque()
            cutoff = now - window_s
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                return q[0] + window_s - now
            q.append(now)
            if len(self._hits) > _PRUNE_THRESHOLD:
                self._prune(now)
            return None

    def _prune(self, now: float) -> None:
        """Drop keys idle past every window, so the dict can't grow unbounded."""
        stale = [k for k, q in self._hits.items()
                 if not q or q[-1] <= now - _PRUNE_AFTER]
        for k in stale:
            del self._hits[k]


_limiter = SlidingWindowLimiter()


def allow(key: str, budget: tuple[int, float]) -> bool:
    """Like enforce(), but reports the verdict instead of raising — for flows
    that must respond identically either way (e.g. forgot-password, where a 429
    would leak how often an address is being asked about)."""
    limit, window_s = budget
    return _limiter.try_acquire(key, limit, window_s) is None


def enforce(key: str, budget: tuple[int, float], what: str = "requests") -> None:
    """Raise 429 (with a Retry-After header) once `key` exhausts its budget."""
    limit, window_s = budget
    retry = _limiter.try_acquire(key, limit, window_s)
    if retry is not None:
        minutes = max(1, int(window_s // 60))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({limit} {what} per {minutes} min). "
                   "Please try again in a moment.",
            headers={"Retry-After": str(max(1, math.ceil(retry)))},
        )
