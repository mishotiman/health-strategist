"""Accounts & authentication.

Password hashing (argon2id), the password policy, single-use email tokens
(address verification / password reset), login throttling, and the
account-linking rule for external sign-in.

The pure logic — policy checks, hashing, the linking decision — is deliberately
kept free of database access so it unit-tests offline, like the rest of the
suite (see conftest.py).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.db import get_connection

# --------------------------------------------------------------------------- #
# Password policy
# --------------------------------------------------------------------------- #
PASSWORD_MIN_LENGTH = 10

# key -> the requirement, phrased for the signup checklist in the UI.
PASSWORD_RULES: tuple[tuple[str, str], ...] = (
    ("length", f"At least {PASSWORD_MIN_LENGTH} characters"),
    ("capital", "At least one capital letter"),
    ("number", "At least one number"),
    ("special", "At least one special character"),
)


def password_failures(password: str) -> list[str]:
    """Keys of every rule the password fails; an empty list means it's acceptable.

    Returning the individual failures (rather than a bool) lets the signup form
    render a live checklist. This is the authoritative check — the UI mirrors it
    for feedback but never replaces it, since a client can be bypassed.
    """
    p = password or ""
    failed: list[str] = []
    if len(p) < PASSWORD_MIN_LENGTH:
        failed.append("length")
    if not any(c.isupper() for c in p):
        failed.append("capital")
    if not any(c.isdigit() for c in p):
        failed.append("number")
    # "special" = punctuation/symbol; a space alone shouldn't satisfy the rule.
    if not any((not c.isalnum()) and (not c.isspace()) for c in p):
        failed.append("special")
    return failed


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
_hasher = PasswordHasher()  # argon2id with the library's current defaults

# Verified against when an account doesn't exist, so a missing user and a wrong
# password take the same amount of time (no enumeration by stopwatch).
_DUMMY_HASH = _hasher.hash("not-a-real-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored_hash: str | None, password: str) -> bool:
    """Check a password. False for OAuth-only accounts, which have no password
    set — callers should still call waste_time() so the timing doesn't differ."""
    if not stored_hash:
        waste_time()
        return False
    try:
        return _hasher.verify(stored_hash, password or "")
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def waste_time() -> None:
    """Burn the same work as a real verify, for logins against unknown emails."""
    try:
        _hasher.verify(_DUMMY_HASH, "wrong")
    except Exception:  # noqa: BLE001 - always mismatches; we only want the delay
        pass


def needs_rehash(stored_hash: str) -> bool:
    """True when argon2 parameters have moved on and the hash should be upgraded."""
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except (InvalidHashError, VerificationError):
        return False


# --------------------------------------------------------------------------- #
# Single-use email tokens
# --------------------------------------------------------------------------- #
VERIFY_EMAIL = "verify_email"
RESET_PASSWORD = "reset_password"

_TOKEN_TTL = {
    VERIFY_EMAIL: dt.timedelta(days=3),
    RESET_PASSWORD: dt.timedelta(hours=1),   # short: it's a credential reset
}


def _token_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def mint_token(user_id: int, kind: str) -> str:
    """Issue a one-time token and return the raw value for the emailed link.
    Only the hash is stored, so the database never holds a usable link."""
    raw = secrets.token_urlsafe(32)
    expires_at = dt.datetime.now(dt.timezone.utc) + _TOKEN_TTL[kind]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO auth_tokens (user_id, kind, token_hash, expires_at) "
            "VALUES (%s, %s, %s, %s)",
            (user_id, kind, _token_hash(raw), expires_at),
        )
        conn.commit()
    return raw


def consume_token(raw: str, kind: str) -> int | None:
    """Validate and burn a token, returning its user id (or None if invalid,
    expired or already used). The UPDATE ... RETURNING makes it genuinely
    single-use even if the link is clicked twice at once."""
    if not raw:
        return None
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE auth_tokens SET used_at = now() "
            "WHERE token_hash = %s AND kind = %s AND used_at IS NULL "
            "  AND expires_at > now() "
            "RETURNING user_id",
            (_token_hash(raw), kind),
        )
        row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


# --------------------------------------------------------------------------- #
# Throttling
# --------------------------------------------------------------------------- #
MAX_ATTEMPTS = 8
ATTEMPT_WINDOW_MINUTES = 15


def record_attempt(email: str | None, ip: str | None, success: bool) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO login_attempts (email, ip, success) VALUES (%s, %s, %s)",
            (normalize_email(email) if email else None, ip, success),
        )
        conn.commit()


def too_many_attempts(email: str | None, ip: str | None) -> bool:
    """True once failures from this email *or* IP exceed the limit in the window."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM login_attempts "
            "WHERE success = false "
            "  AND attempted_at > now() - (%s * INTERVAL '1 minute') "
            "  AND (email = %s OR ip = %s)",
            (ATTEMPT_WINDOW_MINUTES, normalize_email(email) if email else None, ip),
        )
        return (cur.fetchone()[0] or 0) >= MAX_ATTEMPTS


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def find_by_email(email: str) -> dict | None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, display_name, password_hash, email_verified, onboarded_at "
            "FROM users WHERE lower(email) = %s",
            (normalize_email(email),),
        )
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
    return dict(zip(cols, row)) if row else None


def get_user(user_id: int) -> dict | None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, display_name, password_hash, email_verified, onboarded_at, "
            "       is_demo "
            "FROM users WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
    return dict(zip(cols, row)) if row else None


def create_account(email: str, password_hash: str | None = None,
                   display_name: str | None = None,
                   email_verified: bool = False) -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (email, display_name, password_hash, email_verified) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (normalize_email(email), display_name, password_hash, email_verified),
        )
        user_id = cur.fetchone()[0]
        conn.commit()
    return user_id


def set_password(user_id: int, password_hash: str) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE users SET password_hash = %s WHERE id = %s",
                    (password_hash, user_id))
        conn.commit()


def mark_email_verified(user_id: int) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE users SET email_verified = true WHERE id = %s", (user_id,))
        conn.commit()


def mark_onboarded(user_id: int) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE users SET onboarded_at = now() WHERE id = %s "
                    "AND onboarded_at IS NULL", (user_id,))
        conn.commit()


# --------------------------------------------------------------------------- #
# External identities
# --------------------------------------------------------------------------- #
def may_auto_link(provider_email_verified: bool, local_email_verified: bool) -> bool:
    """Whether a Google/Microsoft identity may be attached to an existing local
    account automatically.

    Only when **both** sides have a verified address. Auto-linking on an
    unverified provider email is an account-takeover route: anyone able to
    register that address at the provider could otherwise claim the account.
    When this returns False the user must sign in with their password first and
    link the provider from account settings.
    """
    return bool(provider_email_verified) and bool(local_email_verified)


def find_identity(provider: str, subject: str) -> int | None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM user_identities WHERE provider = %s AND subject = %s",
            (provider, subject),
        )
        row = cur.fetchone()
    return row[0] if row else None


def link_identity(user_id: int, provider: str, subject: str, email: str | None) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO user_identities (user_id, provider, subject, email) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (provider, subject) DO NOTHING",
            (user_id, provider, subject, normalize_email(email) if email else None),
        )
        conn.commit()
