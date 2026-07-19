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

import httpx
from fastapi import Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import agent, bloodwork, session as sess, whoop
from app.db import get_connection
from app.ingestion import query_metrics, upsert_metrics
from app.qa import answer_question
from app.rag import retrieve

app = FastAPI(title="Personal Health Strategist")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# The real account WHOOP "log in" connects (single-owner app).
OWNER_USER_ID = int(os.environ.get("OWNER_USER_ID", "1"))


def _demo_user_id() -> int:
    """The public sample account. Anyone without a login session is treated as this user."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE is_demo = true ORDER BY id LIMIT 1")
        row = cur.fetchone()
    return row[0] if row else OWNER_USER_ID


def current_user_id(request: Request) -> int:
    """Who is this request? A valid signed cookie -> that user; otherwise the demo user."""
    uid = sess.read_token(request.cookies.get(sess.COOKIE_NAME))
    return uid if uid is not None else _demo_user_id()


def _user_info(user_id: int) -> dict:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, is_demo FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    if not row:
        return {"user_id": user_id, "name": "Unknown", "is_demo": True}
    uid, name, is_demo = row
    return {"user_id": uid, "is_demo": is_demo,
            "name": name or ("Demo User" if is_demo else "Your account")}


@app.get("/")
def index():
    """Serve the thin chat UI."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


# ---- request models --------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    k: int = 6


class UserRequest(BaseModel):
    email: str
    goals: str | None = None
    sex: str | None = None
    birth_year: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    injuries: str | None = None


class MetricRecord(BaseModel):
    date: str            # YYYY-MM-DD
    metric_type: str     # canonical (see app.ingestion.CANONICAL_UNITS)
    value: float
    unit: str | None = None


class MetricsRequest(BaseModel):
    user_id: int
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
def me(user_id: int = Depends(current_user_id)):
    """Which WHOOP user this browser is acting as: the demo user, or the real
    logged-in account. The UI reads this to render the 'Connected: …' label."""
    return _user_info(user_id)


@app.post("/auth/logout")
def logout(response: Response):
    """Drop the session cookie — back to the demo user."""
    response.delete_cookie(sess.COOKIE_NAME)
    return {"ok": True}


class ProfileRequest(BaseModel):
    goals: str | None = None
    sex: str | None = None
    birth_year: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    injuries: str | None = None


@app.post("/profile")
def update_profile(req: ProfileRequest, user_id: int = Depends(current_user_id)):
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
@app.post("/users")
def create_user(req: UserRequest):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (email) VALUES (%s) "
            "ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email RETURNING id",
            (req.email,),
        )
        user_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO profiles (user_id, goals, sex, birth_year, height_cm, weight_kg, injuries)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, req.goals, req.sex, req.birth_year,
             req.height_cm, req.weight_kg, req.injuries),
        )
        conn.commit()
    return {"user_id": user_id}


@app.post("/metrics")
def ingest_metrics(req: MetricsRequest):
    written = upsert_metrics(
        req.user_id, req.source,
        [r.model_dump() for r in req.records],
    )
    return {"user_id": req.user_id, "source": req.source, "metrics_written": written}


@app.get("/metrics/{user_id}")
def get_metrics(user_id: int, type: str | None = None, limit: int = 100,
                session_user: int = Depends(current_user_id)):
    # Only your own data (via login session) or the public demo user's.
    if user_id != session_user and user_id != _demo_user_id():
        return JSONResponse(status_code=403,
                            content={"error": "not authorized to read this user's metrics"})
    return {"user_id": user_id, "metrics": query_metrics(user_id, type, limit)}


@app.post("/upload/bloodwork")
async def upload_bloodwork(file: UploadFile = File(...),
                           user_id: int = Depends(current_user_id)):
    """Upload a lab-report PDF; Claude classifies it and, if it's bloodwork,
    extracts values into normalized metrics for the current user."""
    is_pdf = (file.filename or "").lower().endswith(".pdf") or file.content_type == "application/pdf"
    if not is_pdf:
        return JSONResponse(status_code=400,
                            content={"status": "bad_file", "metrics_written": 0,
                                     "message": "Please upload a PDF lab report."})
    return bloodwork.ingest_pdf(user_id, await file.read())


# ---- agent -----------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest, user_id: int = Depends(current_user_id)):
    """Talk to the health-strategist agent as the logged-in user (or demo user).
    The thread is namespaced per user so demo and real memory never mix."""
    thread = f"{user_id}:{req.thread_id}" if req.thread_id else None
    return agent.run(user_id, req.message, thread)


# ---- WHOOP OAuth -----------------------------------------------------------
@app.get("/whoop/connect")
def whoop_connect(user_id: int = OWNER_USER_ID):
    """Start 'Log in with WHOOP' for the owner account (defaults to OWNER_USER_ID)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE id = %s", (user_id,))
        if cur.fetchone() is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"No user with id {user_id} exists yet.",
                         "hint": "Onboard first: POST /users with an email."},
            )
    # state = "<user_id>.<random CSRF token>". The token is stored server-side
    # and verified on callback, so a forged callback can't bind someone else's
    # authorization code to this user.
    state = whoop.new_oauth_state(user_id)
    return RedirectResponse(whoop.authorize_url(state=state))


@app.get("/whoop/sync")
def whoop_sync(user_id: int = Depends(current_user_id)):
    """Pull the latest WHOOP data for the logged-in user (called on app open).
    The demo user has pre-seeded sample data, so there's nothing to sync."""
    if _user_info(user_id)["is_demo"]:
        return {"demo": True, "detail": "Showing sample data for Demo User."}
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
            "hint": "Start at http://localhost:8000/whoop/connect?user_id=1 — "
                    "don't open the callback URL directly.",
        }
    if not state:
        return {"error": "missing state (user id)"}
    user_id = whoop.verify_oauth_state(state)  # checks the CSRF token, one-time use
    if user_id is None:
        return {"error": "invalid or expired OAuth state — restart at /whoop/connect"}
    try:
        whoop.connect_and_sync(user_id, code)
    except httpx.HTTPStatusError as e:
        return {"whoop_api_error": e.response.status_code, "detail": e.response.text[:500]}
    # Log this browser in as the connected user, then return to the app.
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(
        sess.COOKIE_NAME, sess.make_token(user_id),
        httponly=True, samesite="lax",
        secure=whoop.REDIRECT_URI.startswith("https"),  # over HTTPS in prod
        max_age=60 * 60 * 24 * 30,
    )
    return resp
