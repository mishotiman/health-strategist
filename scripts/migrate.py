#!/usr/bin/env python3
"""Apply database migrations in order, tracking what has been applied.

Replaces the "run the right migrate_*.sql files by hand, in the right order"
workflow: a `schema_migrations` table records every applied file, so this can
be run on any database — fresh, local, or the Azure one — and it applies only
what's missing, in the one canonical order below.

    python scripts/migrate.py            # apply pending migrations
    python scripts/migrate.py --dry-run  # show what would run, change nothing

Uses DATABASE_URL via app.config (same value the app itself uses). Every file
in the list is idempotent (IF NOT EXISTS / guarded DO blocks), so the first
run against an existing database that predates the tracking table simply
no-ops through the files and records them.

Adding a migration: create the .sql next to this script and APPEND it to
MIGRATIONS — never reorder or edit entries that may already be applied
somewhere. (scripts/index_corpus.sql stays out of the list on purpose: the
ANN index build runs after embedding, not at schema time.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import app.*

from app.db import get_connection  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent

# Canonical order (chronological; sync_time alters the table providers creates).
MIGRATIONS = [
    "init_db.sql",           # the full schema; a fresh DB needs nothing else
    "migrate_week3.sql",
    "migrate_features.sql",
    "migrate_auth.sql",
    "migrate_providers.sql",
    "migrate_sync_time.sql",
    "migrate_workouts.sql",
    "migrate_corpus.sql",
]


def applied_migrations(cur) -> set[str]:
    cur.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  name TEXT PRIMARY KEY,"
        "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    cur.execute("SELECT name FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply pending DB migrations in order.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be applied without touching the database.")
    args = ap.parse_args()

    with get_connection() as conn, conn.cursor() as cur:
        done = applied_migrations(cur)
        conn.commit()
        pending = [name for name in MIGRATIONS if name not in done]

        if not pending:
            print(f"Up to date — {len(done)} migration(s) applied, nothing pending.")
            return 0
        if args.dry_run:
            print("Would apply, in order:")
            for name in pending:
                print(f"  {name}")
            return 0

        for name in pending:
            sql = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            print(f"applying {name} ...", end=" ", flush=True)
            # The file and its tracking row commit together: a failure rolls
            # both back, so a migration can never be recorded but not applied.
            cur.execute(sql)
            cur.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
            conn.commit()
            print("ok")

    print(f"Applied {len(pending)} migration(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
