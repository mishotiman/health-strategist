"""WHOOP OAuth2 + data sync, normalized through the ingestion layer.

Flow:
  1. /whoop/connect redirects the user to WHOOP to authorize.
  2. WHOOP redirects back to /whoop/callback with a code.
  3. We exchange the code for tokens, store them, then sync recent data.
  4. WHOOP's JSON is mapped to canonical metric_types and upserted.

Tokens live in provider_connections (see app.connections, shared with every
other data provider); we refresh with the offline refresh_token when the access
token has expired.

The connection is to the WHOOP **account**, not a device: WHOOP's API is
account-scoped, so swapping straps keeps one continuous history and needs no
modelling here.
"""

from __future__ import annotations

import datetime as dt
import os
import secrets
from urllib.parse import urlencode

import httpx

from app import connections
from app.db import get_connection
from app.ingestion import delete_metrics, upsert_metrics
from app.workouts import delete_workouts, offset_to_tz, upsert_workouts

# Key for this provider in provider_connections / the `source` column.
PROVIDER = "whoop"

# 1 kilojoule in kilocalories (WHOOP reports workout energy in kilojoules).
_KJ_TO_KCAL = 0.239006

AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
API_BASE = "https://api.prod.whoop.com/developer/v2"
SCOPES = "offline read:recovery read:sleep read:cycles read:workout read:profile read:body_measurement"

CLIENT_ID = os.environ.get("WHOOP_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("WHOOP_CLIENT_SECRET", "")
REDIRECT_URI = os.environ.get("WHOOP_REDIRECT_URI", "http://localhost:8000/whoop/callback")


# --------------------------------------------------------------------------- #
# OAuth
# --------------------------------------------------------------------------- #
def authorize_url(state: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
    })


def new_oauth_state(user_id: int) -> str:
    """Mint an OAuth `state`: the user id plus a random CSRF token. The token is
    stored server-side (below) and verified on callback, so a forged callback
    with someone else's code can't bind it to this user."""
    token = secrets.token_urlsafe(24)
    _store_oauth_state(user_id, token)
    return f"{user_id}.{token}"


def _store_oauth_state(user_id: int, token: str) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO whoop_oauth_states (user_id, state_token, created_at)
            VALUES (%s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                state_token = EXCLUDED.state_token, created_at = now()
            """,
            (user_id, token),
        )
        conn.commit()


def verify_oauth_state(state: str) -> int | None:
    """Validate a callback `state` and return the user id if it matches the
    token we stored, else None. The token is single-use: consumed on success."""
    user_part, _, token = (state or "").partition(".")
    if not user_part.isdigit() or not token:
        return None
    user_id = int(user_part)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT state_token FROM whoop_oauth_states WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        if not row or not secrets.compare_digest(row[0], token):
            return None
        cur.execute("DELETE FROM whoop_oauth_states WHERE user_id = %s", (user_id,))
        conn.commit()
    return user_id


def _token_request(data: dict) -> dict:
    resp = httpx.post(TOKEN_URL, data={**data, "client_id": CLIENT_ID,
                                       "client_secret": CLIENT_SECRET}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def exchange_code(code: str) -> dict:
    return _token_request({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
    })


def _save_tokens(user_id: int, tok: dict, external_user_id: str | None = None) -> None:
    expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
        seconds=int(tok.get("expires_in", 3600)))
    connections.save_tokens(
        user_id, PROVIDER,
        access_token=tok["access_token"],
        refresh_token=tok.get("refresh_token"),
        expires_at=expires_at,
        external_user_id=external_user_id,
    )


def _valid_access_token(user_id: int) -> str:
    conn_row = connections.get(user_id, PROVIDER)
    if not conn_row:
        raise RuntimeError("This user has not connected WHOOP yet.")
    expires_at = conn_row["expires_at"]
    if expires_at and expires_at <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60):
        tok = _token_request({"grant_type": "refresh_token",
                              "refresh_token": conn_row["refresh_token"],
                              "scope": SCOPES})
        _save_tokens(user_id, tok)
        return tok["access_token"]
    return conn_row["access_token"]


# --------------------------------------------------------------------------- #
# Sync — map WHOOP JSON to canonical metrics
# --------------------------------------------------------------------------- #
# WHOOP score key -> (canonical metric_type, decimal places)
_RECOVERY_MAP = [
    ("recovery_score", "recovery_score", 0),
    ("hrv_rmssd_milli", "hrv_rmssd", 1),
    ("resting_heart_rate", "resting_hr", 0),
    ("spo2_percentage", "spo2", 1),
    ("skin_temp_celsius", "skin_temp", 1),
]
_SLEEP_STAGE_KEYS = (
    "total_light_sleep_time_milli",
    "total_slow_wave_sleep_time_milli",
    "total_rem_sleep_time_milli",
)


def parse_recovery(records: list[dict]) -> list[dict]:
    """Map WHOOP /recovery records to canonical metric rows. Pure: no I/O."""
    out: list[dict] = []
    for rec in records:
        score = rec.get("score") or {}
        date = (rec.get("created_at") or "")[:10]
        if not date:
            continue
        for whoop_key, canonical, rnd in _RECOVERY_MAP:
            if score.get(whoop_key) is not None:
                out.append({"date": date, "metric_type": canonical,
                            "value": round(float(score[whoop_key]), rnd)})
    return out


def parse_sleep(records: list[dict]) -> list[dict]:
    """Map WHOOP /activity/sleep records to canonical metric rows. Pure: no I/O.

    WHOOP returns overnight sleeps and naps (``nap: true``) in the same feed, and
    a nap can share a calendar date with that night's sleep. Since metrics are
    keyed by (date, metric_type), we keep them separate: the main sleep metrics
    (sleep_hours, sleep_efficiency, respiratory_rate) come only from overnight
    sessions, while naps are summed per day into their own nap_hours metric.
    Mixing them would let a short nap overwrite the real night's sleep.
    """
    out: list[dict] = []
    nap_ms_by_date: dict[str, float] = {}
    for s in records:
        score = s.get("score") or {}
        date = (s.get("start") or "")[:10]
        if not date:
            continue
        stages = score.get("stage_summary") or {}
        asleep_ms = sum(stages.get(k, 0) or 0 for k in _SLEEP_STAGE_KEYS)

        if s.get("nap"):
            # Accumulate; a day can have several naps.
            if asleep_ms:
                nap_ms_by_date[date] = nap_ms_by_date.get(date, 0) + asleep_ms
            continue

        if asleep_ms:
            out.append({"date": date, "metric_type": "sleep_hours",
                        "value": round(asleep_ms / 3_600_000, 2)})
        if score.get("sleep_efficiency_percentage") is not None:
            out.append({"date": date, "metric_type": "sleep_efficiency",
                        "value": round(float(score["sleep_efficiency_percentage"]), 1)})
        if score.get("respiratory_rate") is not None:
            out.append({"date": date, "metric_type": "respiratory_rate",
                        "value": round(float(score["respiratory_rate"]), 1)})

    for date, ms in nap_ms_by_date.items():
        out.append({"date": date, "metric_type": "nap_hours",
                    "value": round(ms / 3_600_000, 2)})
    return out


def _num(v, places: int):
    """Round to `places` dp, or None if the value is missing."""
    return round(float(v), places) if v is not None else None


def _hr(v):
    """Heart rate as a whole number, or None."""
    return int(round(float(v))) if v is not None else None


def _parse_iso(ts: str | None):
    """WHOOP ISO timestamp (trailing 'Z') -> aware datetime, or None."""
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_min(start: str, end: str | None):
    """Minutes between two ISO timestamps (offset-independent)."""
    s, e = _parse_iso(start), _parse_iso(end)
    return round((e - s).total_seconds() / 60, 1) if s and e else None


def parse_workout(records: list[dict]) -> list[dict]:
    """Map WHOOP /activity/workout records to workout rows. Pure: no I/O.

    WHOOP v2 names the activity in `sport_name`; older payloads carry a numeric
    `sport_id` instead (kept as a string fallback). Energy arrives as kilojoules
    and is converted to kcal. `start`/`end` are UTC and stored as-is (the true
    instant); WHOOP's per-activity `timezone_offset` is kept so the read layer
    can show local time, and the calendar date is taken in local time so a
    late-night session lands on the day the user actually trained. Records
    without an id or start time are skipped.
    """
    out: list[dict] = []
    for w in records:
        wid, start = w.get("id"), w.get("start")
        if wid is None or not start:
            continue
        score = w.get("score") or {}
        sport = w.get("sport_name")
        if not sport and w.get("sport_id") is not None:
            sport = str(w["sport_id"])
        kj = score.get("kilojoule")
        offset = w.get("timezone_offset")
        tz, started = offset_to_tz(offset), _parse_iso(start)
        local_date = (started.astimezone(tz).date().isoformat()
                      if started and tz else start[:10])
        out.append({
            "external_id": str(wid),
            "sport": sport,
            "workout_date": local_date,
            "start_time": start,
            "end_time": w.get("end"),
            "tz_offset": offset,
            "duration_min": _duration_min(start, w.get("end")),
            "strain": _num(score.get("strain"), 1),
            "avg_hr": _hr(score.get("average_heart_rate")),
            "max_hr": _hr(score.get("max_heart_rate")),
            "calories": round(kj * _KJ_TO_KCAL) if kj is not None else None,
            "distance_m": _num(score.get("distance_meter"), 1),
        })
    return out


def _get(path: str, token: str, params: dict) -> dict:
    resp = httpx.get(f"{API_BASE}{path}", headers={"Authorization": f"Bearer {token}"},
                     params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_profile(user_id: int) -> dict:
    """The connected WHOOP account's profile (read:profile scope): its account id,
    email and name."""
    return _get("/user/profile/basic", _valid_access_token(user_id), {})


def profile_name(profile: dict) -> str | None:
    name = " ".join(x for x in (profile.get("first_name"), profile.get("last_name")) if x)
    return name.strip() or None


def _store_display_name(user_id: int, name: str) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE users SET display_name = %s WHERE id = %s", (name, user_id))
        conn.commit()


def _expiry(tok: dict) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=int(tok.get("expires_in", 3600)))


def connect_and_sync(user_id: int, code: str) -> dict:
    """Called from the OAuth callback.

    Devices are irrelevant here — a WHOOP account keeps one continuous history
    across straps. A different **account** is not: its readings would land under
    the same `source='whoop'` and silently overwrite the existing ones day for
    day. So before storing anything we check who we just authenticated as, and
    if it isn't the account already connected we park the authorization and let
    the user decide (see replace_pending / discard_pending).
    """
    tok = exchange_code(code)

    # Identify the account with the fresh token, *before* touching stored state.
    profile, new_account_id = {}, None
    try:
        profile = _get("/user/profile/basic", tok["access_token"], {})
        if profile.get("user_id") is not None:
            new_account_id = str(profile["user_id"])
    except httpx.HTTPStatusError:
        pass  # can't identify it; fall through and treat as a normal connect

    existing = connections.get(user_id, PROVIDER)
    known_id = (existing or {}).get("external_user_id")
    if existing and known_id and new_account_id and known_id != new_account_id:
        connections.save_pending(user_id, PROVIDER, tok["access_token"],
                                 tok.get("refresh_token"), _expiry(tok), new_account_id)
        return {"status": "account_mismatch",
                "connected_account": known_id, "new_account": new_account_id}

    _save_tokens(user_id, tok, external_user_id=new_account_id)
    name = profile_name(profile)
    if name:
        _store_display_name(user_id, name)
    return sync(user_id)


def replace_pending(user_id: int) -> dict:
    """Accept a parked authorization: discard the previous account's data, make
    the new account live, and pull its history."""
    pending = connections.get_pending(user_id, PROVIDER)
    if not pending:
        return {"status": "no_pending"}

    metrics_removed = delete_metrics(user_id, PROVIDER)
    workouts_removed = delete_workouts(user_id, PROVIDER)
    connections.promote_pending(user_id, PROVIDER)
    try:
        name = profile_name(fetch_profile(user_id))
        if name:
            _store_display_name(user_id, name)
    except httpx.HTTPStatusError:
        pass

    result = sync(user_id)
    result.update({"status": "replaced", "metrics_removed": metrics_removed,
                   "workouts_removed": workouts_removed})
    return result


def discard_pending(user_id: int) -> bool:
    """Decline it: the existing connection and its data are left untouched."""
    return connections.drop_pending(user_id, PROVIDER)


def _backfill_account_id(user_id: int) -> None:
    """Connections made before we tracked the provider's account id get it filled
    in on their next sync, so nobody has to reconnect. One extra call, once."""
    row = connections.get(user_id, PROVIDER)
    if not row or row.get("external_user_id"):
        return
    try:
        profile = fetch_profile(user_id)
        if profile.get("user_id") is not None:
            connections.set_external_user_id(user_id, PROVIDER, str(profile["user_id"]))
    except httpx.HTTPStatusError:
        pass


def sync(user_id: int, limit: int = 25) -> dict:
    token = _valid_access_token(user_id)
    _backfill_account_id(user_id)
    records = parse_recovery(_get("/recovery", token, {"limit": limit}).get("records", []))
    records += parse_sleep(_get("/activity/sleep", token, {"limit": limit}).get("records", []))
    written = upsert_metrics(user_id, PROVIDER, records)

    # Logged training sessions land in their own table (per-activity events).
    workouts = parse_workout(_get("/activity/workout", token, {"limit": limit}).get("records", []))
    workouts_written = upsert_workouts(user_id, PROVIDER, workouts)

    # Only a sync that got this far counts: the calls above raise on a bad token
    # or an API error, so a failure can never leave the data looking fresh.
    synced_at = connections.mark_synced(user_id, PROVIDER)

    return {"source": PROVIDER, "metrics_written": written,
            "workouts_written": workouts_written,
            "last_synced_at": synced_at.isoformat() if synced_at else None,
            "distinct_dates": len({r["date"] for r in records})}
