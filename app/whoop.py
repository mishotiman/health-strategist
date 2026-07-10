"""WHOOP OAuth2 + data sync, normalized through the ingestion layer.

Flow:
  1. /whoop/connect redirects the user to WHOOP to authorize.
  2. WHOOP redirects back to /whoop/callback with a code.
  3. We exchange the code for tokens, store them, then sync recent data.
  4. WHOOP's JSON is mapped to canonical metric_types and upserted.

Tokens live in whoop_connections; we refresh with the offline refresh_token
when the access token has expired.
"""

from __future__ import annotations

import datetime as dt
import os
from urllib.parse import urlencode

import httpx

from app.db import get_connection
from app.ingestion import upsert_metrics

AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
API_BASE = "https://api.prod.whoop.com/developer/v2"
SCOPES = "offline read:recovery read:sleep read:cycles read:profile read:body_measurement"

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


def _save_tokens(user_id: int, tok: dict) -> None:
    expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
        seconds=int(tok.get("expires_in", 3600)))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO whoop_connections (user_id, access_token, refresh_token, expires_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                refresh_token = COALESCE(EXCLUDED.refresh_token, whoop_connections.refresh_token),
                expires_at = EXCLUDED.expires_at,
                updated_at = now()
            """,
            (user_id, tok["access_token"], tok.get("refresh_token"), expires_at),
        )
        conn.commit()


def _valid_access_token(user_id: int) -> str:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT access_token, refresh_token, expires_at FROM whoop_connections WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        raise RuntimeError("This user has not connected WHOOP yet.")
    access_token, refresh_token, expires_at = row
    if expires_at and expires_at <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60):
        tok = _token_request({"grant_type": "refresh_token", "refresh_token": refresh_token,
                              "scope": SCOPES})
        _save_tokens(user_id, tok)
        return tok["access_token"]
    return access_token


# --------------------------------------------------------------------------- #
# Sync — map WHOOP JSON to canonical metrics
# --------------------------------------------------------------------------- #
def _get(path: str, token: str, params: dict) -> dict:
    resp = httpx.get(f"{API_BASE}{path}", headers={"Authorization": f"Bearer {token}"},
                     params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def connect_and_sync(user_id: int, code: str) -> dict:
    """Called from the OAuth callback: store tokens, then pull recent data."""
    _save_tokens(user_id, exchange_code(code))
    return sync(user_id)


def sync(user_id: int, limit: int = 25) -> dict:
    token = _valid_access_token(user_id)
    records: list[dict] = []

    # Recovery: recovery score, HRV, resting HR, SpO2, skin temp
    for rec in _get("/recovery", token, {"limit": limit}).get("records", []):
        score = rec.get("score") or {}
        date = (rec.get("created_at") or "")[:10]
        if not date:
            continue
        for whoop_key, canonical, rnd in [
            ("recovery_score", "recovery_score", 0),
            ("hrv_rmssd_milli", "hrv_rmssd", 1),
            ("resting_heart_rate", "resting_hr", 0),
            ("spo2_percentage", "spo2", 1),
            ("skin_temp_celsius", "skin_temp", 1),
        ]:
            if score.get(whoop_key) is not None:
                records.append({"date": date, "metric_type": canonical,
                                "value": round(float(score[whoop_key]), rnd)})

    # Sleep: hours asleep, efficiency, respiratory rate
    for s in _get("/activity/sleep", token, {"limit": limit}).get("records", []):
        score = s.get("score") or {}
        date = (s.get("start") or "")[:10]
        if not date:
            continue
        stages = score.get("stage_summary") or {}
        asleep_ms = sum(stages.get(k, 0) or 0 for k in (
            "total_light_sleep_time_milli",
            "total_slow_wave_sleep_time_milli",
            "total_rem_sleep_time_milli",
        ))
        if asleep_ms:
            records.append({"date": date, "metric_type": "sleep_hours",
                            "value": round(asleep_ms / 3_600_000, 2)})
        if score.get("sleep_efficiency_percentage") is not None:
            records.append({"date": date, "metric_type": "sleep_efficiency",
                            "value": round(float(score["sleep_efficiency_percentage"]), 1)})
        if score.get("respiratory_rate") is not None:
            records.append({"date": date, "metric_type": "respiratory_rate",
                            "value": round(float(score["respiratory_rate"]), 1)})

    written = upsert_metrics(user_id, "whoop", records)
    return {"source": "whoop", "metrics_written": written,
            "distinct_dates": len({r["date"] for r in records})}
