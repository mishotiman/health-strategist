"""The sliding-window rate limiter (app.ratelimit) — pure, no I/O.

Time is controlled by monkeypatching time.monotonic inside the module, so the
window behaviour is tested deterministically instead of with sleeps.
"""

import pytest
from fastapi import HTTPException

from app import ratelimit
from app.ratelimit import SlidingWindowLimiter


@pytest.fixture
def clock(monkeypatch):
    """A settable clock injected into app.ratelimit's view of time."""
    state = {"now": 1000.0}
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: state["now"])
    return state


def test_allows_up_to_limit(clock):
    lim = SlidingWindowLimiter()
    for _ in range(5):
        assert lim.try_acquire("k", 5, 60) is None


def test_blocks_over_limit_and_reports_wait(clock):
    lim = SlidingWindowLimiter()
    for _ in range(3):
        assert lim.try_acquire("k", 3, 60) is None
    retry = lim.try_acquire("k", 3, 60)
    assert retry == pytest.approx(60)  # first slot frees a full window from now


def test_rejected_request_costs_nothing(clock):
    """Being told to wait must not push the wait further out."""
    lim = SlidingWindowLimiter()
    lim.try_acquire("k", 1, 60)
    first = lim.try_acquire("k", 1, 60)
    clock["now"] += 30
    second = lim.try_acquire("k", 1, 60)
    assert first == pytest.approx(60)
    assert second == pytest.approx(30)  # shrank with time: the denial wasn't recorded


def test_window_expiry_frees_slots(clock):
    lim = SlidingWindowLimiter()
    for _ in range(3):
        lim.try_acquire("k", 3, 60)
    assert lim.try_acquire("k", 3, 60) is not None
    clock["now"] += 61
    assert lim.try_acquire("k", 3, 60) is None


def test_keys_are_independent(clock):
    lim = SlidingWindowLimiter()
    assert lim.try_acquire("a", 1, 60) is None
    assert lim.try_acquire("a", 1, 60) is not None
    assert lim.try_acquire("b", 1, 60) is None  # a's exhaustion doesn't touch b


def test_enforce_raises_429_with_retry_after(clock):
    key = "enforce-test"
    budget = (2, 60)
    ratelimit.enforce(key, budget)
    ratelimit.enforce(key, budget)
    with pytest.raises(HTTPException) as exc:
        ratelimit.enforce(key, budget)
    assert exc.value.status_code == 429
    assert int(exc.value.headers["Retry-After"]) >= 1
    assert "Rate limit" in exc.value.detail


def test_prune_never_drops_live_counts(clock):
    """Pruning removes only keys idle past every window — a busy key survives."""
    lim = SlidingWindowLimiter()
    lim.try_acquire("busy", 2, 60)
    lim._prune(clock["now"] + 30)   # mid-window sweep
    assert lim.try_acquire("busy", 2, 60) is None
    assert lim.try_acquire("busy", 2, 60) is not None  # count was preserved
