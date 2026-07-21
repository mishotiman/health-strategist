"""Google/Microsoft sign-in: the parts that must not be got wrong — PKCE,
single-use state, and issuer pinning. No network: discovery is stubbed."""

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import pytest

from app import oidc
from app.db import get_connection


def test_pkce_challenge_is_the_sha256_of_the_verifier():
    verifier, challenge = oidc._pkce_pair()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    assert challenge == expected
    assert "=" not in challenge          # base64url, unpadded
    assert verifier != challenge         # the verifier never leaves the server


def test_each_login_gets_a_fresh_verifier():
    assert oidc._pkce_pair()[0] != oidc._pkce_pair()[0]


def test_provider_without_credentials_is_not_offered(monkeypatch):
    monkeypatch.setitem(oidc.PROVIDERS["google"], "client_id", "")
    monkeypatch.setitem(oidc.PROVIDERS["google"], "client_secret", "")
    assert oidc.is_configured("google") is False
    assert all(p["provider"] != "google" for p in oidc.available())


# --- issuer pinning ---------------------------------------------------------
def test_issuer_must_match_discovery(monkeypatch):
    monkeypatch.setitem(oidc._discovery_cache, "google",
                        {"issuer": "https://accounts.google.com"})
    oidc._check_issuer("google", {"iss": "https://accounts.google.com"})   # no raise
    with pytest.raises(oidc.OIDCError):
        oidc._check_issuer("google", {"iss": "https://evil.example"})


def test_microsoft_tenant_issuer_is_resolved_from_the_tid_claim(monkeypatch):
    # the `common` endpoint issues per-tenant issuers, so the placeholder has to
    # be resolved against the token's own tid — and must actually agree with it
    monkeypatch.setitem(oidc._discovery_cache, "microsoft",
                        {"issuer": "https://login.microsoftonline.com/{tenantid}/v2.0"})
    oidc._check_issuer("microsoft", {
        "iss": "https://login.microsoftonline.com/abc-123/v2.0", "tid": "abc-123"})
    with pytest.raises(oidc.OIDCError):
        oidc._check_issuer("microsoft", {
            "iss": "https://login.microsoftonline.com/abc-123/v2.0", "tid": "someone-else"})
    with pytest.raises(oidc.OIDCError):
        oidc._check_issuer("microsoft", {
            "iss": "https://login.microsoftonline.com/abc-123/v2.0"})   # no tid at all


# --- is the provider's email actually proven? -------------------------------
def test_google_states_verification_outright():
    assert oidc.email_is_verified("google", {"email_verified": True}) is True
    assert oidc.email_is_verified("google", {"email_verified": False}) is False
    assert oidc.email_is_verified("google", {}) is False


def test_personal_microsoft_account_is_verified_by_construction():
    # you can't register an MSA on an address you don't control
    assert oidc.email_is_verified("microsoft", {"tid": oidc.MSA_TENANT_ID}) is True


def test_work_account_without_domain_proof_is_not_verified():
    # a tenant admin can set `email` to anything, so this must stay blocked
    assert oidc.email_is_verified("microsoft", {"tid": "some-company-tenant"}) is False
    assert oidc.email_is_verified("microsoft", {}) is False


def test_work_account_with_domain_owner_verification_is_trusted():
    assert oidc.email_is_verified(
        "microsoft", {"tid": "some-company-tenant", "xms_edov": True}) is True


def test_msa_tenant_does_not_leak_to_other_providers():
    # the MSA shortcut must be Microsoft-only, never a bypass for another provider
    assert oidc.email_is_verified("google", {"tid": oidc.MSA_TENANT_ID}) is False


# --- the authorization request ----------------------------------------------
def test_authorize_url_carries_state_nonce_pkce_and_account_chooser(db, monkeypatch):
    monkeypatch.setitem(oidc._discovery_cache, "google",
                        {"authorization_endpoint": "https://accounts.example/auth"})
    monkeypatch.setitem(oidc.PROVIDERS["google"], "client_id", "test-client")

    url = oidc.authorize_url("google")
    q = parse_qs(urlparse(url).query)

    assert url.startswith("https://accounts.example/auth?")
    assert q["response_type"] == ["code"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["state"] and q["nonce"] and q["code_challenge"]
    # without this the provider silently reuses the browser's signed-in account
    assert q["prompt"] == ["select_account"]

    # the verifier is kept server-side and must never appear in the redirect
    assert "code_verifier" not in q
    stored = oidc._take_state(q["state"][0], "google")
    assert stored is not None and stored[0] == q["nonce"][0]


# --- login state ------------------------------------------------------------
def _put_state(state, provider, nonce="n", verifier="v"):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO oauth_login_states (state, provider, nonce, code_verifier) "
                    "VALUES (%s, %s, %s, %s)", (state, provider, nonce, verifier))
        conn.commit()


def test_state_can_only_be_used_once(db):
    _put_state("state-once", "google", nonce="abc", verifier="xyz")
    assert oidc._take_state("state-once", "google") == ("abc", "xyz")
    assert oidc._take_state("state-once", "google") is None   # replay rejected


def test_state_is_scoped_to_its_provider(db):
    _put_state("state-scoped", "google")
    assert oidc._take_state("state-scoped", "microsoft") is None
    assert oidc._take_state("state-scoped", "google") is not None


def test_unknown_state_is_rejected(db):
    assert oidc._take_state("never-issued", "google") is None
    assert oidc._take_state("", "google") is None
