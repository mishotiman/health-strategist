"""Encryption at rest for third-party OAuth tokens.

`provider_connections` holds live WHOOP access and refresh tokens. Those are
bearer credentials to somebody's health data, so a database leak — a backup left
somewhere, a read-only replica, a support query pasted into a chat — should not
hand them over in the clear. Postgres access control alone is one layer; this is
the second.

Key: TOKEN_ENCRYPTION_KEY, a urlsafe-base64 32-byte Fernet key. Generate one with

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

and store it beside SESSION_SECRET in the Container Apps secret store. Losing the
key does not lose accounts — it invalidates stored tokens, and users reconnect.

Rollout is deliberately gradual, so this can ship without a maintenance window:

  * With no key set, both functions pass values through unchanged. Dev, tests and
    CI work with no extra setup, and deploying the code before setting the secret
    is a no-op rather than an outage.
  * `decrypt` recognises legacy plaintext and returns it as-is, so rows written
    before the key existed keep working while `scripts/encrypt_tokens.py`
    migrates them.
"""

from __future__ import annotations

import logging

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings

log = logging.getLogger(__name__)

# Every Fernet token begins with a 0x80 version byte followed by a big-endian
# timestamp, which base64s to this prefix for any realistic date. It is how we
# tell "already encrypted" from "written before the key existed" without a
# schema column to track it.
_FERNET_PREFIX = "gAAAAA"

_cipher: Fernet | None = None
_warned = False


def _get_cipher() -> Fernet | None:
    global _cipher
    if _cipher is None and settings.token_encryption_key:
        _cipher = Fernet(settings.token_encryption_key.encode())
    return _cipher


def _warn_once() -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning("TOKEN_ENCRYPTION_KEY is not set - provider OAuth tokens are "
                    "being stored in plaintext.")


def looks_encrypted(value: str | None) -> bool:
    return bool(value) and value.startswith(_FERNET_PREFIX)


def encrypt(value: str | None) -> str | None:
    """Ciphertext for storage, or the value unchanged when no key is configured."""
    if value is None:
        return None
    cipher = _get_cipher()
    if cipher is None:
        _warn_once()
        return value
    return cipher.encrypt(value.encode()).decode()


def decrypt(value: str | None) -> str | None:
    """Plaintext for use. Handles three cases: a value this key encrypted, a
    legacy plaintext row, and a value some OTHER key encrypted.

    That last case returns the ciphertext unchanged with an error logged, rather
    than raising. The caller is about to use it as a bearer token, so the failure
    surfaces as a 401 and a prompt to reconnect the provider - a bad key degrades
    to "reconnect WHOOP" instead of 500ing every request, and the log says which
    it was. Raising here would also turn the (unlikely) plaintext token that
    happens to start with the Fernet prefix into a hard outage for that user.
    """
    if value is None:
        return None
    cipher = _get_cipher()
    if cipher is None or not looks_encrypted(value):
        return value
    try:
        return cipher.decrypt(value.encode()).decode()
    except InvalidToken:
        log.error("Stored provider token could not be decrypted with the current "
                  "TOKEN_ENCRYPTION_KEY (rotated or mismatched key?). The user "
                  "will need to reconnect the provider.")
        return value
