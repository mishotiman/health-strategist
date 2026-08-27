"""Encryption at rest for provider OAuth tokens (app/crypto.py).

Pure and offline: no database, no network. The three behaviours that matter are
the round trip, the no-key pass-through that keeps dev/CI working, and the
legacy-plaintext read that makes the rollout gradual.
"""

import pytest
from cryptography.fernet import Fernet


@pytest.fixture
def crypto_with_key(monkeypatch):
    """app.crypto bound to a fresh key, with its module-level cipher reset."""
    import app.crypto as crypto
    from app.config import settings

    monkeypatch.setattr(settings, "token_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(crypto, "_cipher", None)
    return crypto


@pytest.fixture
def crypto_no_key(monkeypatch):
    import app.crypto as crypto
    from app.config import settings

    monkeypatch.setattr(settings, "token_encryption_key", "")
    monkeypatch.setattr(crypto, "_cipher", None)
    return crypto


def test_round_trip(crypto_with_key):
    token = "whoop-access-token-abc123"
    stored = crypto_with_key.encrypt(token)
    assert stored != token, "value must not be stored in the clear"
    assert crypto_with_key.decrypt(stored) == token


def test_ciphertext_differs_each_time(crypto_with_key):
    """Fernet includes a random IV, so the same token encrypts differently every
    time — two users with the same token don't produce matching rows."""
    a = crypto_with_key.encrypt("same-token")
    b = crypto_with_key.encrypt("same-token")
    assert a != b
    assert crypto_with_key.decrypt(a) == crypto_with_key.decrypt(b) == "same-token"


def test_none_passes_through(crypto_with_key):
    """A refresh response often omits refresh_token; NULL must stay NULL."""
    assert crypto_with_key.encrypt(None) is None
    assert crypto_with_key.decrypt(None) is None


def test_no_key_is_a_passthrough(crypto_no_key):
    """Dev and CI run without a key: behaviour is the pre-existing plaintext."""
    assert crypto_no_key.encrypt("plain") == "plain"
    assert crypto_no_key.decrypt("plain") == "plain"


def test_legacy_plaintext_is_readable(crypto_with_key):
    """Rows written before the key existed must keep working, so the key can be
    introduced without a maintenance window."""
    assert crypto_with_key.decrypt("legacy-plaintext-token") == "legacy-plaintext-token"


def test_wrong_key_degrades_instead_of_raising(monkeypatch, crypto_with_key):
    """A rotated/mismatched key must not 500 every request. The caller uses the
    value as a bearer token, so it surfaces as a 401 and a reconnect prompt."""
    stored = crypto_with_key.encrypt("token")

    from app.config import settings
    monkeypatch.setattr(settings, "token_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(crypto_with_key, "_cipher", None)

    assert crypto_with_key.decrypt(stored) == stored  # unchanged, error logged


def test_looks_encrypted_detects_fernet(crypto_with_key):
    assert crypto_with_key.looks_encrypted(crypto_with_key.encrypt("x"))
    assert not crypto_with_key.looks_encrypted("plain-token")
    assert not crypto_with_key.looks_encrypted(None)
