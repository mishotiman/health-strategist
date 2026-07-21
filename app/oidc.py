"""Google / Microsoft sign-in (OpenID Connect).

The authorization-code flow with PKCE, mirroring the structure of the WHOOP
OAuth code in app.whoop but generalized over providers:

  1. /auth/oauth/{provider}/start mints a `state` (CSRF), a `nonce` (replay) and
     a PKCE `code_verifier`, stores them server-side, and redirects to the
     provider.
  2. The provider redirects back with a code; `complete()` checks the state,
     exchanges the code, and *verifies the id_token* — signature against the
     provider's JWKS, plus issuer, audience, expiry and nonce.

Nothing is trusted from the query string beyond the opaque state lookup, and
each state is single-use.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from urllib.parse import urlencode

import httpx
from authlib.jose import JsonWebKey, jwt

from app.db import get_connection

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8000").rstrip("/")

# "common" lets both personal and work/school Microsoft accounts sign in.
_MS_TENANT = os.environ.get("MICROSOFT_TENANT", "common")

PROVIDERS: dict[str, dict] = {
    "google": {
        "label": "Google",
        "discovery": "https://accounts.google.com/.well-known/openid-configuration",
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
    },
    "microsoft": {
        "label": "Microsoft",
        "discovery": (f"https://login.microsoftonline.com/{_MS_TENANT}"
                      "/v2.0/.well-known/openid-configuration"),
        "client_id": os.environ.get("MICROSOFT_CLIENT_ID", ""),
        "client_secret": os.environ.get("MICROSOFT_CLIENT_SECRET", ""),
    },
}

SCOPE = "openid email profile"
_CLOCK_LEEWAY = 120  # seconds of tolerance for clock drift

# Every *personal* Microsoft account lives in this one well-known tenant.
MSA_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad"


def email_is_verified(provider: str, claims: dict) -> bool:
    """Whether the provider has actually proven the user owns this address.

    This gates account linking, so it must not be generous:

    - Google states it outright via `email_verified`.
    - Microsoft **personal** accounts (the MSA tenant above) are verified by
      construction — you can't register an MSA on an address you don't control,
      Microsoft mails a code during signup.
    - Microsoft **work/school** accounts are not: a tenant admin can set the
      `email` attribute to any string, including someone else's address. Only
      the optional `xms_edov` claim ("email domain owner verified") settles it,
      so without it we treat the address as unverified.
    """
    if claims.get("email_verified") or claims.get("xms_edov") is True:
        return True
    if provider == "microsoft" and claims.get("tid") == MSA_TENANT_ID:
        return True
    return False

_discovery_cache: dict[str, dict] = {}
_jwks_cache: dict[str, object] = {}


def is_configured(provider: str) -> bool:
    """Whether this provider has credentials — the UI hides buttons that aren't set up."""
    cfg = PROVIDERS.get(provider)
    return bool(cfg and cfg["client_id"] and cfg["client_secret"])


def available() -> list[dict]:
    return [{"provider": k, "label": v["label"]}
            for k, v in PROVIDERS.items() if is_configured(k)]


def redirect_uri(provider: str) -> str:
    return f"{APP_BASE_URL}/auth/oauth/{provider}/callback"


def _config(provider: str) -> dict:
    """The provider's OIDC discovery document (cached for the process lifetime)."""
    if provider not in _discovery_cache:
        resp = httpx.get(PROVIDERS[provider]["discovery"], timeout=15)
        resp.raise_for_status()
        _discovery_cache[provider] = resp.json()
    return _discovery_cache[provider]


def _jwks(provider: str):
    if provider not in _jwks_cache:
        resp = httpx.get(_config(provider)["jwks_uri"], timeout=15)
        resp.raise_for_status()
        _jwks_cache[provider] = JsonWebKey.import_key_set(resp.json())
    return _jwks_cache[provider]


# --------------------------------------------------------------------------- #
# In-flight login state (CSRF + replay + PKCE)
# --------------------------------------------------------------------------- #
def _pkce_pair() -> tuple[str, str]:
    """(verifier kept server-side, S256 challenge sent to the provider)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def authorize_url(provider: str) -> str:
    """Start a login: persist state/nonce/verifier, return the provider URL."""
    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    verifier, challenge = _pkce_pair()
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO oauth_login_states (state, provider, nonce, code_verifier) "
            "VALUES (%s, %s, %s, %s)",
            (state, provider, nonce, verifier),
        )
        conn.commit()
    params = {
        "client_id": PROVIDERS[provider]["client_id"],
        "redirect_uri": redirect_uri(provider),
        "response_type": "code",
        "scope": SCOPE,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # Always show the account chooser. Without this the provider silently
        # reuses whichever account is already signed in *in the browser*, so a
        # user with a second account can never pick it — and on a shared machine
        # someone else's session would sign them straight into this app.
        "prompt": "select_account",
    }
    return _config(provider)["authorization_endpoint"] + "?" + urlencode(params)


def _take_state(state: str, provider: str) -> tuple[str, str] | None:
    """Consume a state exactly once, returning (nonce, code_verifier)."""
    if not state:
        return None
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM oauth_login_states WHERE state = %s AND provider = %s "
            "RETURNING nonce, code_verifier",
            (state, provider),
        )
        row = cur.fetchone()
        conn.commit()
    return (row[0], row[1]) if row else None


def purge_stale_states(older_than_minutes: int = 30) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM oauth_login_states "
                    "WHERE created_at < now() - (%s * INTERVAL '1 minute')",
                    (older_than_minutes,))
        conn.commit()


# --------------------------------------------------------------------------- #
# Callback
# --------------------------------------------------------------------------- #
class OIDCError(Exception):
    """The login could not be trusted; the caller should send the user back to /login."""


def _check_issuer(provider: str, claims: dict) -> None:
    """Microsoft's `common` endpoint issues per-tenant issuers, so the discovery
    issuer contains a {tenantid} placeholder that must be resolved against the
    token's own `tid` claim. Google's is a fixed string."""
    expected = _config(provider).get("issuer", "")
    got = claims.get("iss", "")
    if "{tenantid}" in expected:
        tid = claims.get("tid")
        if not tid or got != expected.replace("{tenantid}", tid):
            raise OIDCError("id_token issuer mismatch")
        return
    if got != expected:
        raise OIDCError("id_token issuer mismatch")


def complete(provider: str, code: str, state: str) -> dict:
    """Finish a login. Returns the verified identity, or raises OIDCError."""
    if not is_configured(provider):
        raise OIDCError(f"{provider} sign-in is not configured")
    taken = _take_state(state, provider)
    if taken is None:
        raise OIDCError("invalid or expired login state")
    nonce, verifier = taken
    if not code:
        raise OIDCError("no authorization code")

    cfg = PROVIDERS[provider]
    resp = httpx.post(
        _config(provider)["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri(provider),
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "code_verifier": verifier,
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise OIDCError(f"token exchange failed ({resp.status_code})")
    id_token = resp.json().get("id_token")
    if not id_token:
        raise OIDCError("no id_token returned")

    # Signature is verified here; the rest of the claims are checked explicitly.
    try:
        claims = jwt.decode(id_token, _jwks(provider))
    except Exception as exc:  # noqa: BLE001 - any failure means "don't trust it"
        raise OIDCError("id_token signature is not valid") from exc

    aud = claims.get("aud")
    audiences = aud if isinstance(aud, list) else [aud]
    if cfg["client_id"] not in audiences:
        raise OIDCError("id_token audience mismatch")
    _check_issuer(provider, claims)
    if float(claims.get("exp", 0)) < time.time() - _CLOCK_LEEWAY:
        raise OIDCError("id_token has expired")
    if claims.get("nonce") != nonce:
        raise OIDCError("id_token nonce mismatch")   # replayed token
    subject = claims.get("sub")
    if not subject:
        raise OIDCError("id_token has no subject")

    return {
        "provider": provider,
        "subject": str(subject),
        "email": (claims.get("email") or "").strip().lower() or None,
        "email_verified": email_is_verified(provider, claims),
        "name": claims.get("name") or None,
    }
