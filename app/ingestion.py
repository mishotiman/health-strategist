"""The normalized ingestion layer — "any source, one schema".

Every source (WHOOP, bloodwork PDF, manual) funnels through upsert_metrics(),
which writes rows in the single health_metrics shape:
    (user_id, source, metric_date, metric_type, value, unit)

CANONICAL_UNITS is the vocabulary: the metric_type names the rest of the app
(and the agent) can rely on, each with a standard unit. A source-specific
mapper's only job is to produce records that use these names.
"""

from __future__ import annotations

from app.db import get_connection

# canonical metric_type -> standard unit. WHOOP units live here; the bloodwork
# panel is large and self-describing, so app.bloodwork registers its own units
# into this dict at import (keeping one source of truth in app.bloodwork.MARKERS).
CANONICAL_UNITS: dict[str, str] = {
    # wearable (WHOOP)
    "recovery_score": "%",
    "hrv_rmssd": "ms",
    "resting_hr": "bpm",
    "spo2": "%",
    "skin_temp": "celsius",
    "sleep_hours": "h",
    "nap_hours": "h",
    "sleep_efficiency": "%",
    "respiratory_rate": "rpm",
}


def upsert_metrics(user_id: int, source: str, records: list[dict]) -> int:
    """Idempotently write normalized metric rows.

    records: list of {"date": "YYYY-MM-DD", "metric_type": <canonical>,
                      "value": <number>, "unit": <optional>}
    A qualitative record carries "text_value" (e.g. "negative") instead of "value".
    Re-running with the same (user_id, source, date, metric_type) updates the
    row instead of inserting a duplicate.
    """
    if not records:
        return 0
    params = [
        (user_id, source, r["date"], r["metric_type"], r.get("value"),
         r.get("unit") or CANONICAL_UNITS.get(r["metric_type"]), r.get("text_value"))
        for r in records
    ]
    with get_connection() as conn, conn.cursor() as cur:
        # executemany pipelines the whole batch in one round-trip set
        cur.executemany(
            """
            INSERT INTO health_metrics
                (user_id, source, metric_date, metric_type, value, unit, text_value)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, source, metric_date, metric_type)
            DO UPDATE SET value = EXCLUDED.value, unit = EXCLUDED.unit,
                          text_value = EXCLUDED.text_value
            """,
            params,
        )
        conn.commit()
    return len(records)


def count_for_source(user_id: int, source: str) -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM health_metrics WHERE user_id = %s AND source = %s",
                    (user_id, source))
        return cur.fetchone()[0] or 0


def delete_metrics(user_id: int, source: str) -> int:
    """Drop every metric a given source contributed for this user.

    Used when a user replaces a connected account with a different one at the
    same provider — the old account's readings aren't theirs to keep mixed in.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM health_metrics WHERE user_id = %s AND source = %s",
                    (user_id, source))
        removed = cur.rowcount
        conn.commit()
    return removed


def tracked_metric_types(user_id: int) -> list[str]:
    """Every metric_type this user actually has data for.

    Exists so a lookup that finds nothing can say WHICH metrics are tracked
    instead of only that this one is missing. An agent told merely "no data"
    has been observed to fill the gap with a plausible number; an agent told
    "vo2max is not tracked; here is what is" can answer honestly.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT metric_type FROM health_metrics WHERE user_id = %s "
            "ORDER BY metric_type",
            (user_id,),
        )
        return [r[0] for r in cur.fetchall()]


def query_metrics(user_id: int, metric_type: str | None = None, limit: int = 100) -> list[dict]:
    sql = (
        "SELECT source, metric_date, metric_type, value, unit, text_value "
        "FROM health_metrics WHERE user_id = %s"
    )
    params: list = [user_id]
    if metric_type:
        sql += " AND metric_type = %s"
        params.append(metric_type)
    sql += " ORDER BY metric_date DESC, metric_type LIMIT %s"
    params.append(limit)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    for row in rows:
        row["metric_date"] = row["metric_date"].isoformat()  # -> string for JSON
        # keep the payload tidy: numeric rows have no text_value, qualitative no value/unit
        for k in ("value", "unit", "text_value"):
            if row.get(k) is None:
                row.pop(k, None)
    return rows
