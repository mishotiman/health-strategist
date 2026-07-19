"""Create (or refresh) the public "Demo User" — a synthetic account so the live
app can be shown to anyone without exposing real health data. Idempotent.

    docker compose exec -T api python scripts/seed_demo_user.py
"""

from __future__ import annotations

import datetime as dt
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # make `app` importable

from app.db import get_connection
from app.ingestion import upsert_metrics

DEMO_EMAIL = "demo@phs.local"
DEMO_NAME = "Demo User"
DAYS = 30


def _get_or_create_demo_user() -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (email, display_name, is_demo)
            VALUES (%s, %s, true)
            ON CONFLICT (email) DO UPDATE SET display_name = EXCLUDED.display_name,
                                             is_demo = true
            RETURNING id
            """,
            (DEMO_EMAIL, DEMO_NAME),
        )
        user_id = cur.fetchone()[0]
        # Fresh single profile (profiles has no unique key on user_id).
        cur.execute("DELETE FROM profiles WHERE user_id = %s", (user_id,))
        cur.execute(
            """
            INSERT INTO profiles (user_id, goals, sex, birth_year, height_cm, weight_kg, injuries)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, "Build strength and improve sleep consistency", "male",
             1992, 180, 78, "occasional lower-back tightness"),
        )
        conn.commit()
    return user_id


def _synthetic_metrics() -> list[dict]:
    """Realistic WHOOP-shaped daily metrics with gentle day-to-day variation."""
    rnd = random.Random(42)  # deterministic so the demo is stable across reseeds
    today = dt.date.today()
    rows: list[dict] = []
    for d in range(DAYS):
        day = (today - dt.timedelta(days=d)).isoformat()
        rows += [
            {"date": day, "metric_type": "recovery_score", "value": rnd.randint(45, 92)},
            {"date": day, "metric_type": "hrv_rmssd", "value": round(rnd.uniform(48, 95), 1)},
            {"date": day, "metric_type": "resting_hr", "value": rnd.randint(46, 58)},
            {"date": day, "metric_type": "sleep_hours", "value": round(rnd.uniform(6.1, 8.4), 2)},
            {"date": day, "metric_type": "sleep_efficiency", "value": round(rnd.uniform(82, 95), 1)},
            {"date": day, "metric_type": "respiratory_rate", "value": round(rnd.uniform(13.5, 16.2), 1)},
            {"date": day, "metric_type": "spo2", "value": round(rnd.uniform(95.0, 98.5), 1)},
        ]
        if d % 6 == 3:  # an occasional daytime nap, to exercise the nap_hours metric
            rows.append({"date": day, "metric_type": "nap_hours", "value": round(rnd.uniform(0.3, 1.1), 2)})
    return rows


def main() -> None:
    user_id = _get_or_create_demo_user()
    written = upsert_metrics(user_id, "whoop", _synthetic_metrics())
    print(f"Demo User ready: id={user_id}, name='{DEMO_NAME}', {written} metrics over {DAYS} days.")


if __name__ == "__main__":
    main()
