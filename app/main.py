"""Personal Health Strategist — FastAPI app.

Endpoints:
  GET  /health              — liveness
  GET  /db-check            — DB reachable + pgvector present
  POST /search              — retrieval only (no LLM)
  POST /ask                 — retrieval + grounded, cited Claude answer
  POST /users               — onboard: create a user + profile
  POST /metrics             — ingest normalized health metrics (any source)
  GET  /metrics/{user_id}   — query a user's metrics
  GET  /whoop/connect       — start WHOOP OAuth (redirects to WHOOP)
  GET  /whoop/callback      — OAuth return: store tokens + sync data
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import (agent, auth, bloodwork, connections, emailer, oidc,
                 session as sess, whoop)
from app.db import get_connection
from app.ingestion import count_for_source as ingestion_count
from app.ingestion import query_metrics, upsert_metrics
from app.workouts import count_for_source as workouts_count
from app.qa import answer_question
from app.rag import retrieve

app = FastAPI(title="Personal Health Strategist")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Public origin, used for cookie hardening and the links in emails.
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8000").rstrip("/")
_COOKIE_SECURE = APP_BASE_URL.startswith("https")


@dataclass
class Principal:
    """Who is making this request.

    user  — signed in with their own account.
    guest — the "Try Health Strategist now" visitor: reads the public sample
            account, may not write anything, and nothing is saved for them.
    anon  — no cookie at all; sent to the login page.
    """
    user_id: int | None
    kind: str  # "user" | "guest" | "anon"

    @property
    def is_user(self) -> bool:
        return self.kind == "user"


def _demo_user_id() -> int | None:
    """The read-only sample account that guests see."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE is_demo = true ORDER BY id LIMIT 1")
        row = cur.fetchone()
    return row[0] if row else None


def current_principal(request: Request) -> Principal:
    """A live session wins; otherwise the guest marker; otherwise anonymous."""
    uid = sess.read(request.cookies.get(sess.COOKIE_NAME))
    if uid is not None:
        return Principal(uid, "user")
    if request.cookies.get(sess.GUEST_COOKIE_NAME):
        demo = _demo_user_id()
        if demo is not None:
            return Principal(demo, "guest")
    return Principal(None, "anon")


def require_user(p: Principal = Depends(current_principal)) -> int:
    """For every endpoint that writes or touches a real account. Guests get 401."""
    if not p.is_user:
        raise HTTPException(status_code=401, detail="Login required.")
    return p.user_id


def readable_user(p: Principal = Depends(current_principal)) -> int:
    """For read-only endpoints: the signed-in account, or the sample account for
    guests. Anonymous callers are still rejected."""
    if p.user_id is None:
        raise HTTPException(status_code=401, detail="Login required.")
    return p.user_id


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        sess.COOKIE_NAME, token,
        httponly=True, samesite="lax", secure=_COOKIE_SECURE,
        max_age=60 * 60 * 24 * sess.SESSION_DAYS,
    )


def _user_info(user_id: int) -> dict:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, email, is_demo, email_verified, onboarded_at "
                    "FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    if not row:
        return {"user_id": user_id, "name": "Unknown", "is_demo": True}
    uid, name, email, is_demo, verified, onboarded = row
    connected = connections.providers_for(uid)
    whoop_connected = whoop.PROVIDER in connected
    last_synced = None
    if whoop_connected:
        # drives the "Synced 3 minutes ago" note under the WHOOP button
        row = connections.get(uid, whoop.PROVIDER)
        if row and row.get("last_synced_at"):
            last_synced = row["last_synced_at"].isoformat()
    return {"user_id": uid, "is_demo": is_demo, "email": email,
            "connected_providers": connected,
            "name": name or (email.split("@")[0] if email else "Your account"),
            "email_verified": verified, "onboarded": onboarded is not None,
            "whoop_connected": whoop_connected,
            "whoop_last_synced_at": last_synced}


def _page(name: str) -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, name))


@app.get("/")
def index(p: Principal = Depends(current_principal)):
    """The chat UI — but only once the visitor is actually allowed to see it.

    anon      -> login page
    unverified-> the "check your email" screen (verification blocks access)
    not yet   -> the onboarding flow
    onboarded -> chat.  Guests skip straight to chat with sample data.
    """
    if p.kind == "anon":
        return RedirectResponse(url="/login", status_code=303)
    if p.kind == "user":
        info = _user_info(p.user_id)
        if not info["email_verified"]:
            return RedirectResponse(url="/login?verify=pending", status_code=303)
        if not info["onboarded"]:
            return RedirectResponse(url="/onboarding", status_code=303)
    return _page("index.html")


@app.get("/login")
def login_page():
    return _page("login.html")


@app.get("/onboarding")
def onboarding_page(user_id: int = Depends(require_user)):
    return _page("onboarding.html")


# ---- request models --------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    k: int = 6


class MetricRecord(BaseModel):
    date: str            # YYYY-MM-DD
    metric_type: str     # canonical (see app.ingestion.CANONICAL_UNITS)
    value: float
    unit: str | None = None


class MetricsRequest(BaseModel):
    # No user_id: metrics always land on the authenticated caller's own account.
    source: str
    records: list[MetricRecord]


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


# ---- basics ----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        has_vector = cur.fetchone() is not None
    return {"db": "reachable", "pgvector": has_vector}


# ---- session ("who is connected") ------------------------------------------
@app.get("/me")
def me(p: Principal = Depends(current_principal)):
    """Who this browser is. The UI reads this to decide which screen to show and
    whether to render the 'User: …' label (signed-in) or a Login button (guest)."""
    if p.kind == "anon":
        return {"kind": "anon"}
    info = _user_info(p.user_id)
    info["kind"] = p.kind
    if p.kind == "guest":  # a guest is browsing sample data, not an account
        info["name"] = None
        info["email"] = None
    return info


# ---- accounts --------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def _password_or_400(password: str) -> None:
    """Reject a password that fails the policy, telling the UI *which* rules failed."""
    failures = auth.password_failures(password)
    if failures:
        raise HTTPException(
            status_code=400,
            detail={"error": "password_policy", "failed": failures,
                    "rules": [{"key": k, "label": lbl} for k, lbl in auth.PASSWORD_RULES]},
        )


@app.get("/auth/password-rules")
def password_rules():
    """The signup form renders this as a live checklist (the server still enforces it)."""
    return {"rules": [{"key": k, "label": lbl} for k, lbl in auth.PASSWORD_RULES]}


@app.post("/auth/register")
def register(req: RegisterRequest, request: Request, response: Response):
    """Create an account. The address starts unverified and stays blocked from
    the app until the emailed link is used."""
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
    _set_session_cookie(response, sess.create(user_id))
    response.delete_cookie(sess.GUEST_COOKIE_NAME)
    return {"ok": True, "user_id": user_id, "email_verified": False}


@app.post("/auth/login")
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

    _set_session_cookie(response, sess.create(user["id"]))
    response.delete_cookie(sess.GUEST_COOKIE_NAME)
    return {"ok": True, "email_verified": user["email_verified"],
            "onboarded": user["onboarded_at"] is not None}


@app.post("/auth/logout")
def logout(request: Request, response: Response):
    """End this session server-side, so the cookie is dead even if it's copied."""
    sess.revoke(request.cookies.get(sess.COOKIE_NAME))
    response.delete_cookie(sess.COOKIE_NAME)
    response.delete_cookie(sess.GUEST_COOKIE_NAME)
    return {"ok": True}


@app.post("/auth/logout-all")
def logout_all(response: Response, user_id: int = Depends(require_user)):
    """Sign out everywhere — revokes every session for this account."""
    count = sess.revoke_all(user_id)
    response.delete_cookie(sess.COOKIE_NAME)
    return {"ok": True, "sessions_revoked": count}


@app.post("/auth/change-password")
def change_password(req: ChangePasswordRequest, response: Response,
                    user_id: int = Depends(require_user)):
    user = auth.get_user(user_id)
    if not user or not auth.verify_password(user.get("password_hash"), req.current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    _password_or_400(req.new_password)

    auth.set_password(user_id, auth.hash_password(req.new_password))
    sess.revoke_all(user_id)                 # a stolen cookie stops working now
    _set_session_cookie(response, sess.create(user_id))   # keep *this* browser signed in
    return {"ok": True}


@app.post("/auth/guest")
def start_guest(response: Response):
    """'Try Health Strategist now' — read-only sample data, nothing saved."""
    if _demo_user_id() is None:
        raise HTTPException(status_code=503, detail="Sample account is not available.")
    response.set_cookie(sess.GUEST_COOKIE_NAME, "1", httponly=True,
                        samesite="lax", secure=_COOKIE_SECURE, max_age=60 * 60 * 12)
    return {"ok": True}


@app.post("/auth/onboarding/complete")
def complete_onboarding(user_id: int = Depends(require_user)):
    auth.mark_onboarded(user_id)
    return {"ok": True}


# ---- Google / Microsoft sign-in --------------------------------------------
@app.get("/auth/providers")
def auth_providers():
    """What sign-in options the login page should offer.

    `passkeys` is the WebAuthn capability flag — false until that lands, so the
    login page hides the passkey entry point rather than showing a dead button.
    """
    return {"providers": oidc.available(), "passkeys": False}


@app.get("/auth/oauth/{provider}/start")
def oauth_start(provider: str):
    if not oidc.is_configured(provider):
        raise HTTPException(status_code=404, detail=f"{provider} sign-in is not configured.")
    return RedirectResponse(oidc.authorize_url(provider), status_code=303)


@app.get("/auth/oauth/{provider}/callback")
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
    _set_session_cookie(resp, sess.create(user_id))
    resp.delete_cookie(sess.GUEST_COOKIE_NAME)
    return resp


# ---- email verification + password reset -----------------------------------
class EmailRequest(BaseModel):
    email: str


class ResetRequest(BaseModel):
    token: str
    new_password: str


def _send_verification(user_id: int, email: str) -> bool:
    token = auth.mint_token(user_id, auth.VERIFY_EMAIL)
    return emailer.send_verification(email, f"{APP_BASE_URL}/auth/verify?token={token}")


@app.get("/auth/verify")
def verify_email(token: str = ""):
    """Target of the emailed link. Consumes the one-time token and unblocks the
    account, then bounces the browser back into the app."""
    user_id = auth.consume_token(token, auth.VERIFY_EMAIL)
    if user_id is None:
        return RedirectResponse(url="/login?verify=invalid", status_code=303)
    auth.mark_email_verified(user_id)
    return RedirectResponse(url="/?verified=1", status_code=303)


@app.post("/auth/resend-verification")
def resend_verification(user_id: int = Depends(require_user)):
    user = auth.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")
    if user["email_verified"]:
        return {"ok": True, "already_verified": True}
    sent = _send_verification(user_id, user["email"])
    return {"ok": True, "sent": sent}


@app.post("/auth/forgot-password")
def forgot_password(req: EmailRequest, request: Request):
    """Always reports success — revealing whether an address has an account would
    let anyone enumerate the user list."""
    email = auth.normalize_email(req.email)
    ip = request.client.host if request.client else None
    if not auth.too_many_attempts(email, ip):
        user = auth.find_by_email(email)
        if user:
            token = auth.mint_token(user["id"], auth.RESET_PASSWORD)
            emailer.send_password_reset(
                email, f"{APP_BASE_URL}/login?reset={token}")
        auth.record_attempt(email, ip, success=False)
    return {"ok": True}


@app.post("/auth/reset-password")
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
    _set_session_cookie(response, sess.create(user_id))
    return {"ok": True}


class ProfileRequest(BaseModel):
    goals: str | None = None
    sex: str | None = None
    birth_year: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    injuries: str | None = None


@app.post("/profile")
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


# ---- RAG -------------------------------------------------------------------
@app.post("/search")
def search(req: AskRequest):
    chunks = retrieve(req.question, req.k)
    return {
        "question": req.question,
        "results": [
            {
                "n": i + 1,
                "similarity": round(c["similarity"], 3),
                "title": c["title"],
                "source": c["source"],
                "preview": " ".join(c["content"].split())[:200],
            }
            for i, c in enumerate(chunks)
        ],
    }


@app.post("/ask")
def ask(req: AskRequest):
    result = answer_question(req.question, req.k)
    sources = [
        {"n": i + 1, "title": c["title"], "authors": c["authors"],
         "year": c["year"], "source": c["source"], "page": c["page"]}
        for i, c in enumerate(result["chunks"])
    ]
    return {"question": req.question, "answer": result["answer"], "sources": sources}


# ---- onboarding + ingestion ------------------------------------------------
@app.post("/metrics")
def ingest_metrics(req: MetricsRequest, user_id: int = Depends(require_user)):
    """Ingest normalized metrics for the caller's own account.

    The user id comes from the session, never the request body — accepting it
    from the body previously let anyone write metrics into any account.
    """
    written = upsert_metrics(user_id, req.source, [r.model_dump() for r in req.records])
    return {"user_id": user_id, "source": req.source, "metrics_written": written}


@app.get("/metrics/{user_id}")
def get_metrics(user_id: int, type: str | None = None, limit: int = 100,
                session_user: int = Depends(readable_user)):
    # Only your own data (via login session) or the public sample account's.
    if user_id != session_user and user_id != _demo_user_id():
        return JSONResponse(status_code=403,
                            content={"error": "not authorized to read this user's metrics"})
    return {"user_id": user_id, "metrics": query_metrics(user_id, type, limit)}


@app.post("/upload/bloodwork")
async def upload_bloodwork(file: UploadFile = File(...),
                           user_id: int = Depends(require_user)):
    """Upload a lab-report PDF; Claude classifies it and, if it's bloodwork,
    extracts values into normalized metrics for the current user."""
    is_pdf = (file.filename or "").lower().endswith(".pdf") or file.content_type == "application/pdf"
    if not is_pdf:
        return JSONResponse(status_code=400,
                            content={"status": "bad_file", "metrics_written": 0,
                                     "message": "Please upload a PDF lab report."})
    try:
        result = bloodwork.ingest_pdf(user_id, await file.read())
        if result.get("status") == "ok":
            name = os.path.splitext(file.filename or "")[0].strip() or "Bloodwork report"
            with get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO bloodwork_documents (user_id, name, report_date, metrics_count) "
                    "VALUES (%s, %s, %s, %s) RETURNING id, uploaded_at",
                    (user_id, name, result.get("report_date"), result.get("metrics_written", 0)),
                )
                doc_id, uploaded_at = cur.fetchone()
                conn.commit()
            result["document"] = {"id": doc_id, "name": name,
                                  "report_date": result.get("report_date"),
                                  "metrics_count": result.get("metrics_written", 0),
                                  "uploaded_at": uploaded_at.isoformat()}
        return result
    except Exception:  # never surface a bare 500 to the uploader
        return JSONResponse(status_code=200,
                            content={"status": "error", "metrics_written": 0,
                                     "message": "Something went wrong processing that file — "
                                                "please try again."})


@app.get("/bloodwork/documents")
def list_bloodwork_documents(user_id: int = Depends(readable_user)):
    """Bloodwork reports the current user has uploaded (for the upload manager)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, report_date, metrics_count, uploaded_at "
            "FROM bloodwork_documents WHERE user_id = %s ORDER BY uploaded_at DESC",
            (user_id,),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["report_date"] = r["report_date"].isoformat() if r["report_date"] else None
        r["uploaded_at"] = r["uploaded_at"].isoformat()
    return {"documents": rows}


class RenameRequest(BaseModel):
    name: str


@app.patch("/bloodwork/documents/{doc_id}")
def rename_bloodwork_document(doc_id: int, req: RenameRequest,
                              user_id: int = Depends(require_user)):
    name = req.name.strip()
    if not name:
        return JSONResponse(status_code=400, content={"error": "name cannot be empty"})
    with get_connection() as conn, conn.cursor() as cur:
        # scoped to the current user, so you can only rename your own documents
        cur.execute("UPDATE bloodwork_documents SET name = %s WHERE id = %s AND user_id = %s",
                    (name, doc_id, user_id))
        updated = cur.rowcount
        conn.commit()
    if not updated:
        return JSONResponse(status_code=404, content={"error": "document not found"})
    return {"ok": True, "id": doc_id, "name": name}


@app.delete("/bloodwork/documents/{doc_id}")
def delete_bloodwork_document(doc_id: int, user_id: int = Depends(require_user)):
    """Remove an uploaded report and the lab values it contributed. (Values are
    matched by report_date; a rare second report on the same date would share them.)"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT report_date FROM bloodwork_documents WHERE id = %s AND user_id = %s",
                    (doc_id, user_id))
        row = cur.fetchone()
        if row is None:
            return JSONResponse(status_code=404, content={"error": "document not found"})
        report_date = row[0]
        cur.execute("DELETE FROM bloodwork_documents WHERE id = %s AND user_id = %s",
                    (doc_id, user_id))
        if report_date is not None:
            cur.execute("DELETE FROM health_metrics WHERE user_id = %s AND source = 'bloodwork' "
                        "AND metric_date = %s", (user_id, report_date))
        conn.commit()
    return {"ok": True, "id": doc_id}


# ---- agent -----------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest, user_id: int = Depends(readable_user)):
    """Talk to the health-strategist agent as the logged-in user (or demo user).
    The thread is namespaced per user so demo and real memory never mix."""
    thread = f"{user_id}:{req.thread_id}" if req.thread_id else None
    return agent.run(user_id, req.message, thread)


# ---- WHOOP OAuth -----------------------------------------------------------
@app.get("/whoop/connect")
def whoop_connect(user_id: int = Depends(require_user)):
    """Connect WHOOP to the signed-in account.

    The user comes from the session. It used to be a `?user_id=` query parameter
    defaulting to a single owner, which let a caller bind their WHOOP account to
    someone else's user. Guests can't connect at all.
    """
    # state = "<user_id>.<random CSRF token>". The token is stored server-side
    # and verified on callback, so a forged callback can't bind someone else's
    # authorization code to this user.
    state = whoop.new_oauth_state(user_id)
    return RedirectResponse(whoop.authorize_url(state=state))


@app.get("/whoop/pending")
def whoop_pending(user_id: int = Depends(require_user)):
    """Details for the "different WHOOP account" prompt: which accounts are
    involved and exactly how much data replacing would delete."""
    pending = connections.get_pending(user_id, whoop.PROVIDER)
    if not pending:
        return {"pending": False}
    existing = connections.get(user_id, whoop.PROVIDER) or {}
    return {
        "pending": True,
        "connected_account": existing.get("external_user_id"),
        "new_account": pending.get("external_user_id"),
        "metrics_at_risk": ingestion_count(user_id, whoop.PROVIDER),
        "workouts_at_risk": workouts_count(user_id, whoop.PROVIDER),
    }


@app.post("/whoop/pending/replace")
def whoop_pending_replace(user_id: int = Depends(require_user)):
    """Replace: drop the old account's WHOOP data and sync the new account."""
    return whoop.replace_pending(user_id)


@app.post("/whoop/pending/cancel")
def whoop_pending_cancel(user_id: int = Depends(require_user)):
    """Cancel: keep the existing connection and data, discard the new authorization."""
    return {"ok": True, "discarded": whoop.discard_pending(user_id)}


@app.post("/whoop/disconnect")
def whoop_disconnect(user_id: int = Depends(require_user)):
    """Unlink WHOOP from this account. The synced metrics stay; only the tokens go."""
    return {"ok": True, "disconnected": connections.delete(user_id, whoop.PROVIDER)}


@app.get("/whoop/sync")
def whoop_sync(user_id: int = Depends(require_user)):
    """Pull the latest WHOOP data for the signed-in user (called on app open).
    The sample account is pre-seeded, so there's nothing to sync."""
    if _user_info(user_id)["is_demo"]:
        return {"demo": True, "detail": "Showing sample data."}
    try:
        return whoop.sync(user_id)
    except RuntimeError as e:            # not connected yet
        return {"connected": False, "detail": str(e)}
    except httpx.HTTPStatusError as e:   # token/refresh/API problem
        return {"connected": True, "error": e.response.status_code}


@app.get("/whoop/callback")
def whoop_callback(code: str | None = None, state: str | None = None,
                   error: str | None = None, error_description: str | None = None):
    # Surface WHOOP's own OAuth error rather than a generic 422.
    if error or not code:
        return {
            "whoop_oauth_error": error or "no authorization code in the request",
            "description": error_description,
            "hint": "Start at /whoop/connect while signed in — "
                    "don't open the callback URL directly.",
        }
    if not state:
        return {"error": "missing state (user id)"}
    user_id = whoop.verify_oauth_state(state)  # checks the CSRF token, one-time use
    if user_id is None:
        return {"error": "invalid or expired OAuth state — restart at /whoop/connect"}
    try:
        result = whoop.connect_and_sync(user_id, code)
    except httpx.HTTPStatusError as e:
        return {"whoop_api_error": e.response.status_code, "detail": e.response.text[:500]}
    # A different WHOOP account than the one already connected: nothing has been
    # changed yet, the app asks before replacing anyone's history.
    if result.get("status") == "account_mismatch":
        return RedirectResponse(url="/?whoop=mismatch", status_code=303)
    # The user was already signed in before starting /whoop/connect, so their
    # session cookie rides along on this redirect — connecting WHOOP no longer
    # doubles as the login mechanism.
    return RedirectResponse(url="/?whoop=connected", status_code=303)
