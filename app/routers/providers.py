"""Connected data providers — the WHOOP OAuth flow, sync, and the
account-mismatch decision endpoints."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from app import connections, whoop
from app.deps import require_user, user_info
from app.ingestion import count_for_source as ingestion_count
from app.workouts import count_for_source as workouts_count

router = APIRouter()


@router.get("/whoop/connect")
def whoop_connect(user_id: int = Depends(require_user)):
    """Connect WHOOP to the signed-in account.

    The user comes from the session. It used to be a `?user_id=` query parameter
    defaulting to a single owner, which let a caller bind their WHOOP account to
    someone else's user. Guests can't connect at all.
    """
    # state = "<user_id>.<random CSRF token>". The token is stored server-side
    # and verified on callback, so a forged callback can't bind someone else's
    # authorization code to this user.
    state = whoop.new_oauth_state(user_id)
    return RedirectResponse(whoop.authorize_url(state=state))


@router.get("/whoop/pending")
def whoop_pending(user_id: int = Depends(require_user)):
    """Details for the "different WHOOP account" prompt: which accounts are
    involved and exactly how much data replacing would delete."""
    pending = connections.get_pending(user_id, whoop.PROVIDER)
    if not pending:
        return {"pending": False}
    existing = connections.get(user_id, whoop.PROVIDER) or {}
    return {
        "pending": True,
        "connected_account": existing.get("external_user_id"),
        "new_account": pending.get("external_user_id"),
        "metrics_at_risk": ingestion_count(user_id, whoop.PROVIDER),
        "workouts_at_risk": workouts_count(user_id, whoop.PROVIDER),
    }


@router.post("/whoop/pending/replace")
def whoop_pending_replace(user_id: int = Depends(require_user)):
    """Replace: drop the old account's WHOOP data and sync the new account."""
    return whoop.replace_pending(user_id)


@router.post("/whoop/pending/cancel")
def whoop_pending_cancel(user_id: int = Depends(require_user)):
    """Cancel: keep the existing connection and data, discard the new authorization."""
    return {"ok": True, "discarded": whoop.discard_pending(user_id)}


@router.post("/whoop/disconnect")
def whoop_disconnect(user_id: int = Depends(require_user)):
    """Unlink WHOOP from this account. The synced metrics stay; only the tokens go."""
    return {"ok": True, "disconnected": connections.delete(user_id, whoop.PROVIDER)}


@router.post("/whoop/sync")
def whoop_sync(user_id: int = Depends(require_user)):
    """Pull the latest WHOOP data for the signed-in user (called on app open).
    POST, not GET: it writes data, and SameSite=lax cookies ride along on
    cross-site GETs. The sample account is pre-seeded, so there's nothing to sync."""
    if user_info(user_id)["is_demo"]:
        return {"demo": True, "detail": "Showing sample data."}
    try:
        return whoop.sync(user_id)
    except RuntimeError as e:            # not connected yet
        return {"connected": False, "detail": str(e)}
    except httpx.HTTPStatusError as e:   # token/refresh/API problem
        return {"connected": True, "error": e.response.status_code}


@router.get("/whoop/callback")
def whoop_callback(code: str | None = None, state: str | None = None,
                   error: str | None = None, error_description: str | None = None):
    # Surface WHOOP's own OAuth error rather than a generic 422.
    if error or not code:
        return {
            "whoop_oauth_error": error or "no authorization code in the request",
            "description": error_description,
            "hint": "Start at /whoop/connect while signed in — "
                    "don't open the callback URL directly.",
        }
    if not state:
        return {"error": "missing state (user id)"}
    user_id = whoop.verify_oauth_state(state)  # checks the CSRF token, one-time use
    if user_id is None:
        return {"error": "invalid or expired OAuth state — restart at /whoop/connect"}
    try:
        result = whoop.connect_and_sync(user_id, code)
    except httpx.HTTPStatusError as e:
        return {"whoop_api_error": e.response.status_code, "detail": e.response.text[:500]}
    # A different WHOOP account than the one already connected: nothing has been
    # changed yet, the app asks before replacing anyone's history.
    if result.get("status") == "account_mismatch":
        return RedirectResponse(url="/?whoop=mismatch", status_code=303)
    # The user was already signed in before starting /whoop/connect, so their
    # session cookie rides along on this redirect — connecting WHOOP no longer
    # doubles as the login mechanism.
    return RedirectResponse(url="/?whoop=connected", status_code=303)
