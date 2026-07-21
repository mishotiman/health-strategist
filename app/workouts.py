"""Workouts — logged training activities (WHOOP now; other sources later).

Unlike health_metrics ("any source, one schema" — a single scalar value per
day), a workout is a per-activity *event*: several can share a calendar date,
each with a sport, duration, strain, heart rate, calories and distance. So they
live in their own `workouts` table.

A source-specific mapper (e.g. app.whoop.parse_workout) produces rows in the
shape below; upsert_workouts persists them idempotently, and query_workouts
reads them back for the agent's `workouts` tool.

    row = {
        "external_id": <source's workout id>,   # required for idempotency
        "sport": "running", "workout_date": "YYYY-MM-DD",
        "start_time": <iso>, "end_time": <iso>,
        "duration_min": 60.0, "strain": 12.3,
        "avg_hr": 149, "max_hr": 181,
        "calories": 478.0, "distance_m": 5000.0,   # distance is cardio-only
    }
"""

from __future__ import annotations

import datetime as dt

from app.db import get_connection


def offset_to_tz(offset: str | None):
    """Parse a WHOOP `timezone_offset` into a tzinfo, or None if absent/unparseable.

    Accepts the shapes WHOOP emits — "+03:00", "-05:00", "+0000", "Z". The offset
    is the user's local UTC offset at the time of the activity, so applying it to
    the stored UTC instant recovers the local wall-clock time they experienced.
    """
    if not offset:
        return None
    s = offset.strip()
    if s in ("Z", "z"):
        return dt.timezone.utc
    sign = 1
    if s[:1] in ("+", "-"):
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    s = s.replace(":", "")
    if len(s) < 2 or not s[:2].isdigit():
        return None
    minutes = int(s[2:4]) if len(s) >= 4 and s[2:4].isdigit() else 0
    return dt.timezone(sign * dt.timedelta(hours=int(s[:2]), minutes=minutes))


def upsert_workouts(user_id: int, source: str, rows: list[dict]) -> int:
    """Idempotently write workout events. Re-running with the same
    (user_id, source, external_id) updates the row instead of duplicating."""
    written = 0
    with get_connection() as conn, conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """
                INSERT INTO workouts
                    (user_id, source, external_id, sport, workout_date, start_time,
                     end_time, duration_min, strain, avg_hr, max_hr, calories,
                     distance_m, tz_offset)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (user_id, source, external_id) DO UPDATE SET
                    sport = EXCLUDED.sport, workout_date = EXCLUDED.workout_date,
                    start_time = EXCLUDED.start_time, end_time = EXCLUDED.end_time,
                    duration_min = EXCLUDED.duration_min, strain = EXCLUDED.strain,
                    avg_hr = EXCLUDED.avg_hr, max_hr = EXCLUDED.max_hr,
                    calories = EXCLUDED.calories, distance_m = EXCLUDED.distance_m,
                    tz_offset = EXCLUDED.tz_offset
                """,
                (user_id, source, r.get("external_id"), r.get("sport"),
                 r.get("workout_date"), r.get("start_time"), r.get("end_time"),
                 r.get("duration_min"), r.get("strain"), r.get("avg_hr"),
                 r.get("max_hr"), r.get("calories"), r.get("distance_m"),
                 r.get("tz_offset")),
            )
            written += 1
        conn.commit()
    return written


def delete_workouts(user_id: int, source: str) -> int:
    """Drop every workout a given source contributed for this user (see
    app.ingestion.delete_metrics for why)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM workouts WHERE user_id = %s AND source = %s",
                    (user_id, source))
        removed = cur.rowcount
        conn.commit()
    return removed


def count_for_source(user_id: int, source: str) -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM workouts WHERE user_id = %s AND source = %s",
                    (user_id, source))
        return cur.fetchone()[0] or 0


def query_workouts(user_id: int, limit: int = 25) -> list[dict]:
    """Recent workouts for a user, newest first. Timestamps come back as ISO
    strings and NULL columns are dropped, so the payload stays tidy for the LLM."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, sport, workout_date, start_time, end_time, duration_min,
                   strain, avg_hr, max_hr, calories, distance_m, tz_offset
            FROM workouts WHERE user_id = %s
            ORDER BY start_time DESC NULLS LAST, workout_date DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    for row in rows:
        # start_time/end_time are stored as UTC instants; render them in the
        # user's local time (from the session's own offset) so "9:12 AM" reads
        # as the wall-clock time they actually trained.
        tz = offset_to_tz(row.pop("tz_offset", None))
        for k in ("start_time", "end_time"):
            v = row.get(k)
            if v is not None:
                row[k] = (v.astimezone(tz) if tz else v).isoformat()
        if row.get("workout_date") is not None:
            row["workout_date"] = row["workout_date"].isoformat()
        for k in list(row):          # drop NULLs (e.g. distance for lifting)
            if row[k] is None:
                row.pop(k)
    return rows
