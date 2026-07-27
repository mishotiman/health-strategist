"""Accounts: register/login, sessions, external sign-in, email verification,
password reset, and the profile the onboarding flow collects."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app import auth, emailer, oidc, ratelimit, session as sess
from app.config import settings
from app.db import get_connection
from app.deps import (COOKIE_SECURE, client_ip, demo_user_id, require_user,
                      set_session_cookie)

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class EmailRequest(BaseModel):
    email: str


class ResetRequest(BaseModel):
    token: str
    new_password: str


class ProfileRequest(BaseModel):
    goals: str | None = None
    sex: str | None = None
    birth_year: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    injuries: str | None = None


def _password_or_400(password: str) -> None:
    """Reject a password that fails the policy, telling the UI *which* rules failed."""
    failures = auth.password_failures(password)
    if failures:
        raise HTTPException(
            status_code=400,
            detail={"error": "password_policy", "failed": failures,
                    "rules": [{"key": k, "label": lbl} for k, lbl in auth.PASSWORD_RULES]},
        )


def _send_verification(user_id: int, email: str) -> bool:
    token = auth.mint_token(user_id, auth.VERIFY_EMAIL)
    return emailer.send_verification(
        email, f"{settings.app_base_url}/auth/verify?token={token}")


@router.get("/auth/password-rules")
def password_rules():
    """The signup form renders this as a live checklist (the server still enforces it)."""
    return {"rules": [{"key": k, "label": lbl} for k, lbl in auth.PASSWORD_RULES]}


@router.post("/auth/register")
def register(req: RegisterRequest, request: Request, response: Response):
    """Create an account. The address starts unverified; verification is a
    reminder banner, not a gate (see index()), but it still governs account
    linking."""
    email = auth.normalize_email(req.email)
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    _password_or_400(req.password)

    if auth.find_by_email(email):
        # Don't confirm which addresses exist; point them at the login page.
        raise HTTPException(status_code=409,
                            detail="That email can't be registered. Try logging in instead.")

    user_id = auth.create_account(email, password_hash=auth.hash_password(req.password))
    _send_verification(user_id, email)
    # Session starts immediately, but /me reports email_verified=false so the UI
    # holds them on the "check your email" screen.
    set_session_cookie(response, sess.create(user_id))
    response.delete_cookie(sess.GUEST_COOKIE_NAME)
    return {"ok": True, "user_id": user_id, "email_verified": False}


@router.post("/auth/login")
def login(req: LoginRequest, request: Request, response: Response):
    email = auth.normalize_email(req.email)
    ip = request.client.host if request.client else None
    if auth.too_many_attempts(email, ip):
        raise HTTPException(status_code=429,
                            detail="Too many attempts. Please wait a few minutes.")

    user = auth.find_by_email(email)
    if not user or not auth.verify_password(user.get("password_hash"), req.password):
        if not user:
            auth.waste_time()          # same cost as a real check: no enumeration
        auth.record_attempt(email, ip, success=False)
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    auth.record_attempt(email, ip, success=True)
    if auth.needs_rehash(user["password_hash"]):     # argon2 params moved on
        auth.set_password(user["id"], auth.hash_password(req.password))

    set_session_cookie(response, sess.create(user["id"]))
    response.delete_cookie(sess.GUEST_COOKIE_NAME)
    return {"ok": True, "email_verified": user["email_verified"],
            "onboarded": user["onboarded_at"] is not None}


@router.post("/auth/logout")
def logout(request: Request, response: Response):
    """End this session server-side, so the cookie is dead even if it's copied."""
    sess.revoke(request.cookies.get(sess.COOKIE_NAME))
    response.delete_cookie(sess.COOKIE_NAME)
    response.delete_cookie(sess.GUEST_COOKIE_NAME)
    return {"ok": True}


@router.post("/auth/logout-all")
def logout_all(response: Response, user_id: int = Depends(require_user)):
    """Sign out everywhere — revokes every session for this account."""
    count = sess.revoke_all(user_id)
    response.delete_cookie(sess.COOKIE_NAME)
    return {"ok": True, "sessions_revoked": count}


@router.post("/auth/change-password")
def change_password(req: ChangePasswordRequest, response: Response,
                    user_id: int = Depends(require_user)):
    user = auth.get_user(user_id)
    if not user or not auth.verify_password(user.get("password_hash"), req.current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    _password_or_400(req.new_password)

    auth.set_password(user_id, auth.hash_password(req.new_password))
    sess.revoke_all(user_id)                 # a stolen cookie stops working now
    set_session_cookie(response, sess.create(user_id))   # keep *this* browser signed in
    return {"ok": True}


@router.post("/auth/guest")
def start_guest(response: Response):
    """'Try Health Strategist now' — read-only sample data, nothing saved."""
    if demo_user_id() is None:
        raise HTTPException(status_code=503, detail="Sample account is not available.")
    response.set_cookie(sess.GUEST_COOKIE_NAME, "1", httponly=True,
                        samesite="lax", secure=COOKIE_SECURE, max_age=60 * 60 * 12)
    return {"ok": True}


@router.post("/auth/onboarding/complete")
def complete_onboarding(user_id: int = Depends(require_user)):
    auth.mark_onboarded(user_id)
    return {"ok": True}


# ---- Google / Microsoft sign-in --------------------------------------------
@router.get("/auth/providers")
def auth_providers():
    """What sign-in options the login page should offer.

    `passkeys` is the WebAuthn capability flag — false until that lands, so the
    login page hides the passkey entry point rather than showing a dead button.
    """
    return {"providers": oidc.available(), "passkeys": False}


@router.get("/auth/oauth/{provider}/start")
def oauth_start(provider: str):
    if not oidc.is_configured(provider):
        raise HTTPException(status_code=404, detail=f"{provider} sign-in is not configured.")
    return RedirectResponse(oidc.authorize_url(provider), status_code=303)


@router.get("/auth/oauth/{provider}/callback")
def oauth_callback(provider: str, code: str | None = None, state: str | None = None,
                   error: str | None = None):
    """Return leg of the OIDC flow: verify, then find/link/create the account."""
    if error:
        return RedirectResponse(url=f"/login?oauth={error}", status_code=303)
    try:
        info = oidc.complete(provider, code or "", state or "")
    except oidc.OIDCError:
        return RedirectResponse(url="/login?oauth=failed", status_code=303)

    user_id = auth.find_identity(provider, info["subject"])
    if user_id is None:
        existing = auth.find_by_email(info["email"]) if info["email"] else None
        if existing:
            # Only merge into an existing account when both sides have proven the
            # address; otherwise this would be an account-takeover route.
            if not auth.may_auto_link(info["email_verified"], existing["email_verified"]):
                return RedirectResponse(url="/login?link=password_required", status_code=303)
            user_id = existing["id"]
            auth.link_identity(user_id, provider, info["subject"], info["email"])
        else:
            if not info["email"]:
                return RedirectResponse(url="/login?oauth=no_email", status_code=303)
            user_id = auth.create_account(
                info["email"], password_hash=None, display_name=info["name"],
                email_verified=info["email_verified"],
            )
            auth.link_identity(user_id, provider, info["subject"], info["email"])

    resp = RedirectResponse(url="/?welcome=1", status_code=303)
    set_session_cookie(resp, sess.create(user_id))
    resp.delete_cookie(sess.GUEST_COOKIE_NAME)
    return resp


# ---- email verification + password reset -----------------------------------
@router.get("/auth/verify")
def verify_email(token: str = ""):
    """Target of the emailed link. Consumes the one-time token and unblocks the
    account, then bounces the browser back into the app."""
    user_id = auth.consume_token(token, auth.VERIFY_EMAIL)
    if user_id is None:
        return RedirectResponse(url="/login?verify=invalid", status_code=303)
    auth.mark_email_verified(user_id)
    return RedirectResponse(url="/?verified=1", status_code=303)


@router.post("/auth/resend-verification")
def resend_verification(user_id: int = Depends(require_user)):
    user = auth.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")
    if user["email_verified"]:
        return {"ok": True, "already_verified": True}
    sent = _send_verification(user_id, user["email"])
    return {"ok": True, "sent": sent}


@router.post("/auth/forgot-password")
def forgot_password(req: EmailRequest, request: Request):
    """Always reports success — revealing whether an address has an account would
    let anyone enumerate the user list.

    Reset traffic gets its own throttle buckets (never login_attempts: failed
    rows there once let 8 reset requests lock the address out of login). Over
    budget, the mail is silently skipped and the response stays identical."""
    email = auth.normalize_email(req.email)
    ip = client_ip(request)
    if (ratelimit.allow(f"reset-email:{email}", ratelimit.RESET_EMAIL)
            and ratelimit.allow(f"reset-ip:{ip}", ratelimit.RESET_IP)):
        user = auth.find_by_email(email)
        if user:
            token = auth.mint_token(user["id"], auth.RESET_PASSWORD)
            emailer.send_password_reset(
                email, f"{settings.app_base_url}/login?reset={token}")
    return {"ok": True}


@router.post("/auth/reset-password")
def reset_password(req: ResetRequest, response: Response):
    _password_or_400(req.new_password)
    user_id = auth.consume_token(req.token, auth.RESET_PASSWORD)
    if user_id is None:
        raise HTTPException(status_code=400,
                            detail="That reset link is invalid or has expired.")
    auth.set_password(user_id, auth.hash_password(req.new_password))
    # Whoever triggered the reset may have had a stolen session — kill them all.
    sess.revoke_all(user_id)
    auth.mark_email_verified(user_id)   # they proved control of the address
    set_session_cookie(response, sess.create(user_id))
    return {"ok": True}


@router.post("/profile")
def update_profile(req: ProfileRequest, user_id: int = Depends(require_user)):
    """Save the profile the onboarding prompt collects, for the current user.
    The agent's `memory` tool reads it to personalize responses."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE profiles SET goals=%s, sex=%s, birth_year=%s, height_cm=%s, "
            "weight_kg=%s, injuries=%s, updated_at=now() WHERE user_id=%s",
            (req.goals, req.sex, req.birth_year, req.height_cm,
             req.weight_kg, req.injuries, user_id),
        )
        if cur.rowcount == 0:
            cur.execute(
                "INSERT INTO profiles (user_id, goals, sex, birth_year, height_cm, weight_kg, injuries)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (user_id, req.goals, req.sex, req.birth_year,
                 req.height_cm, req.weight_kg, req.injuries),
            )
        conn.commit()
    return {"ok": True}
