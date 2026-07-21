"""Password policy, hashing, and the account-linking rule. Pure — no database."""

import pytest

from app import auth


# --- policy: 10+ characters, a capital, a number, a special character --------
def test_a_compliant_password_passes():
    assert auth.password_failures("Str0ng!Passw0rd") == []


@pytest.mark.parametrize("password, missing", [
    ("Ab1!x",          "length"),    # only 5 characters
    ("lowercase1!x",   "capital"),
    ("NoDigitsHere!x", "number"),
    ("NoSpecials1abc", "special"),
])
def test_each_rule_is_enforced(password, missing):
    assert missing in auth.password_failures(password)


def test_every_failing_rule_is_reported_at_once():
    # the signup form renders these as a checklist, so it needs all of them
    assert set(auth.password_failures("abc")) == {"length", "capital", "number", "special"}


def test_empty_password_fails_everything():
    assert set(auth.password_failures("")) == {"length", "capital", "number", "special"}


def test_whitespace_does_not_count_as_a_special_character():
    # long enough, has a capital and a digit, but the only non-alphanumeric is a space
    assert "special" in auth.password_failures("Abcdefgh 1")


def test_rule_keys_are_exposed_for_the_ui_checklist():
    assert [key for key, _ in auth.PASSWORD_RULES] == ["length", "capital", "number", "special"]


# --- hashing ----------------------------------------------------------------
def test_hash_and_verify_roundtrip():
    stored = auth.hash_password("Str0ng!Passw0rd")
    assert "Str0ng!Passw0rd" not in stored          # never recoverable from the hash
    assert auth.verify_password(stored, "Str0ng!Passw0rd") is True
    assert auth.verify_password(stored, "Str0ng!Passw0rE") is False


def test_same_password_hashes_differently_each_time():
    # per-hash salt: two accounts with the same password must not look identical
    assert auth.hash_password("Str0ng!Passw0rd") != auth.hash_password("Str0ng!Passw0rd")


def test_account_without_a_password_never_verifies():
    # OAuth-only accounts have password_hash = NULL; nothing should log in as them
    assert auth.verify_password(None, "anything") is False
    assert auth.verify_password("", "anything") is False


# --- external identity linking ----------------------------------------------
@pytest.mark.parametrize("provider_verified, local_verified, may_link", [
    (True,  True,  True),    # both sides proven -> safe to link
    (True,  False, False),
    (False, True,  False),   # provider hasn't proven the address -> takeover risk
    (False, False, False),
])
def test_auto_link_requires_both_sides_verified(provider_verified, local_verified, may_link):
    assert auth.may_auto_link(provider_verified, local_verified) is may_link
