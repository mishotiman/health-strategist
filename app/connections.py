"""Connected data providers — WHOOP today; Garmin, Oura and friends later.

One row per (user, provider) holding that provider's OAuth tokens, so adding a
new sensor means writing a mapper that emits canonical metric rows (see
app.ingestion) rather than another bespoke `*_connections` table with its own
copy of the refresh logic.

This mirrors `user_identities`, which already generalizes the *login* providers
(Google/Microsoft) by `(provider, subject)`. Same idea, applied to the providers
we pull data from.

`external_user_id` records **which account at the provider** the tokens belong
to. Without it we cannot tell a reconnect of the same account from a switch to a
different one — and the data rows, all tagged `source='whoop'`, would silently
merge.
"""

from __future__ import annotations

import datetime as dt

from app.db import get_connection


def save_tokens(user_id: int, provider: str, access_token: str,
                refresh_token: str | None, expires_at: dt.datetime,
                external_user_id: str | None = None) -> None:
    """Store (or refresh) a provider's tokens for this user.

    A refresh response often omits the refresh token; COALESCE keeps the one we
    already hold rather than nulling it. Same for external_user_id, which is only
    known after we fetch the provider's profile.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO provider_connections
                (user_id, provider, access_token, refresh_token, expires_at, external_user_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, provider) DO UPDATE SET
                access_token     = EXCLUDED.access_token,
                refresh_token    = COALESCE(EXCLUDED.refresh_token,
                                            provider_connections.refresh_token),
                expires_at       = EXCLUDED.expires_at,
                external_user_id = COALESCE(EXCLUDED.external_user_id,
                                            provider_connections.external_user_id),
                updated_at       = now()
            """,
            (user_id, provider, access_token, refresh_token, expires_at, external_user_id),
        )
        conn.commit()


def get(user_id: int, provider: str) -> dict | None:
    """The stored connection, or None if this user hasn't connected the provider."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, provider, external_user_id, access_token, refresh_token, "
            "       expires_at, connected_at, last_synced_at "
            "FROM provider_connections WHERE user_id = %s AND provider = %s",
            (user_id, provider),
        )
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
    return dict(zip(cols, row)) if row else None


def set_external_user_id(user_id: int, provider: str, external_user_id: str) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE provider_connections SET external_user_id = %s, updated_at = now() "
            "WHERE user_id = %s AND provider = %s",
            (str(external_user_id), user_id, provider),
        )
        conn.commit()


def mark_synced(user_id: int, provider: str) -> dt.datetime | None:
    """Record a **successful** data pull, and return the timestamp written.

    Distinct from `updated_at`, which only moves when tokens are stored, so it
    would report a token refresh as if it were fresh data. Callers hand the
    returned value straight back to the UI, which saves a follow-up read.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE provider_connections SET last_synced_at = now() "
            "WHERE user_id = %s AND provider = %s "
            "RETURNING last_synced_at",
            (user_id, provider),
        )
        row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def delete(user_id: int, provider: str) -> bool:
    """Disconnect a provider. Already-synced metrics are deliberately kept —
    only the tokens go."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM provider_connections WHERE user_id = %s AND provider = %s",
                    (user_id, provider))
        removed = cur.rowcount
        conn.commit()
    return bool(removed)


# --------------------------------------------------------------------------- #
# Pending authorizations (a different account at the provider)
# --------------------------------------------------------------------------- #
# How long a parked authorization stays actionable. Long enough to decide,
# short enough that a forgotten one doesn't nag on a later visit.
PENDING_TTL_MINUTES = 60


def save_pending(user_id: int, provider: str, access_token: str,
                 refresh_token: str | None, expires_at: dt.datetime,
                 external_user_id: str | None) -> None:
    """Park an authorization instead of applying it, pending the user's choice."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pending_connections
                (user_id, provider, external_user_id, access_token, refresh_token, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, provider) DO UPDATE SET
                external_user_id = EXCLUDED.external_user_id,
                access_token     = EXCLUDED.access_token,
                refresh_token    = EXCLUDED.refresh_token,
                expires_at       = EXCLUDED.expires_at,
                created_at       = now()
            """,
            (user_id, provider, external_user_id, access_token, refresh_token, expires_at),
        )
        conn.commit()


def get_pending(user_id: int, provider: str) -> dict | None:
    """A still-actionable parked authorization, or None. Stale ones are ignored
    so an abandoned decision doesn't resurface days later."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, provider, external_user_id, access_token, refresh_token, "
            "       expires_at, created_at "
            "FROM pending_connections "
            "WHERE user_id = %s AND provider = %s "
            "  AND created_at > now() - (%s * INTERVAL '1 minute')",
            (user_id, provider, PENDING_TTL_MINUTES),
        )
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
    return dict(zip(cols, row)) if row else None


def drop_pending(user_id: int, provider: str) -> bool:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM pending_connections WHERE user_id = %s AND provider = %s",
                    (user_id, provider))
        removed = cur.rowcount
        conn.commit()
    return bool(removed)


def promote_pending(user_id: int, provider: str) -> bool:
    """Make the parked authorization the live connection, replacing whatever was
    there. Done in one transaction so we can never end up with both or neither."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO provider_connections
                (user_id, provider, external_user_id, access_token, refresh_token, expires_at)
            SELECT user_id, provider, external_user_id, access_token, refresh_token, expires_at
            FROM pending_connections WHERE user_id = %s AND provider = %s
            ON CONFLICT (user_id, provider) DO UPDATE SET
                external_user_id = EXCLUDED.external_user_id,
                access_token     = EXCLUDED.access_token,
                refresh_token    = EXCLUDED.refresh_token,
                expires_at       = EXCLUDED.expires_at,
                connected_at     = now(),
                updated_at       = now()
            """,
            (user_id, provider),
        )
        promoted = cur.rowcount
        cur.execute("DELETE FROM pending_connections WHERE user_id = %s AND provider = %s",
                    (user_id, provider))
        conn.commit()
    return bool(promoted)


def providers_for(user_id: int) -> list[str]:
    """Which providers this user has connected — for the UI and /me."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT provider FROM provider_connections WHERE user_id = %s "
                    "ORDER BY provider", (user_id,))
        return [r[0] for r in cur.fetchall()]
