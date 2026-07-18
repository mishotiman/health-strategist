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

# canonical metric_type -> standard unit
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
    # bloodwork (labs)
    "vitamin_d": "ng/mL",
    "ferritin": "ng/mL",
    "glucose": "mg/dL",
    "hdl": "mg/dL",
    "ldl": "mg/dL",
    "triglycerides": "mg/dL",
    "total_cholesterol": "mg/dL",
    "hba1c": "%",
    "tsh": "mIU/L",
    "testosterone": "ng/dL",
    "crp": "mg/L",
}


def upsert_metrics(user_id: int, source: str, records: list[dict]) -> int:
    """Idempotently write normalized metric rows.

    records: list of {"date": "YYYY-MM-DD", "metric_type": <canonical>,
                      "value": <number>, "unit": <optional>}
    Re-running with the same (user_id, source, date, metric_type) updates the
    value instead of inserting a duplicate.
    """
    written = 0
    with get_connection() as conn, conn.cursor() as cur:
        for r in records:
            metric_type = r["metric_type"]
            unit = r.get("unit") or CANONICAL_UNITS.get(metric_type)
            cur.execute(
                """
                INSERT INTO health_metrics
                    (user_id, source, metric_date, metric_type, value, unit)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, source, metric_date, metric_type)
                DO UPDATE SET value = EXCLUDED.value, unit = EXCLUDED.unit
                """,
                (user_id, source, r["date"], metric_type, r["value"], unit),
            )
            written += 1
        conn.commit()
    return written


def query_metrics(user_id: int, metric_type: str | None = None, limit: int = 100) -> list[dict]:
    sql = (
        "SELECT source, metric_date, metric_type, value, unit "
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
    for row in rows:  # dates -> strings for JSON
        row["metric_date"] = row["metric_date"].isoformat()
    return rows
