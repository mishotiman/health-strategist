#!/usr/bin/env python
"""One-off: encrypt provider OAuth tokens that predate TOKEN_ENCRYPTION_KEY.

    docker compose exec -T api python scripts/encrypt_tokens.py         # report
    docker compose exec -T api python scripts/encrypt_tokens.py --apply

app/crypto.py encrypts on write and reads legacy plaintext transparently, so the
application keeps working either way; this closes the gap by converting rows that
were written before the key existed. Safe to re-run: rows already encrypted are
detected by their Fernet prefix and skipped, so it never double-encrypts.

Run it against the deployed database from inside the container, which already
holds DATABASE_URL and sits inside the DB firewall:

    az containerapp exec -g phs-rg -n phs-api \
        --command "python scripts/encrypt_tokens.py --apply" < /dev/null
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.crypto import encrypt, looks_encrypted
from app.db import get_connection

TABLES = ("provider_connections", "pending_connections")
FIELDS = ("access_token", "refresh_token")


def main() -> None:
    ap = argparse.ArgumentParser(description="Encrypt legacy plaintext OAuth tokens")
    ap.add_argument("--apply", action="store_true", help="write (default: report only)")
    args = ap.parse_args()

    if not settings.token_encryption_key:
        sys.exit("TOKEN_ENCRYPTION_KEY is not set - nothing to migrate to. "
                 "Set it first (see .env.example), then re-run.")

    total_plain = 0
    for table in TABLES:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT user_id, provider, access_token, refresh_token FROM {table}")
            rows = cur.fetchall()

            plain = [r for r in rows
                     if any(r[i + 2] and not looks_encrypted(r[i + 2]) for i in range(len(FIELDS)))]
            total_plain += len(plain)
            print(f"{table}: {len(rows)} row(s), {len(plain)} with plaintext token(s)")

            if args.apply and plain:
                for user_id, provider, access, refresh in plain:
                    cur.execute(
                        f"UPDATE {table} SET access_token = %s, refresh_token = %s "
                        "WHERE user_id = %s AND provider = %s",
                        (encrypt(access) if access and not looks_encrypted(access) else access,
                         encrypt(refresh) if refresh and not looks_encrypted(refresh) else refresh,
                         user_id, provider),
                    )
                conn.commit()
                print(f"{table}: encrypted {len(plain)} row(s)")

    if not args.apply:
        print(f"\nReport only - {total_plain} row(s) would be encrypted. Re-run with --apply.")


if __name__ == "__main__":
    main()
