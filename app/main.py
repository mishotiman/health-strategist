"""Personal Health Strategist — FastAPI app assembly.

The endpoints live in domain routers (app/routers/): accounts (auth, sessions,
external sign-in, profile), knowledge (/search, /ask, corpus stats), chat (the
agent), health_data (/metrics + bloodwork uploads), and providers (WHOOP).
Shared request dependencies — the Principal, auth gates, and spend guards —
live in app/deps.py. This module keeps only the pages, the liveness probes,
and /me.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.db import get_connection
from app.deps import Principal, current_principal, require_user, user_info
from app.routers import accounts, chat, health_data, knowledge, providers

app = FastAPI(title="Personal Health Strategist")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(accounts.router)
app.include_router(knowledge.router)
app.include_router(chat.router)
app.include_router(health_data.router)
app.include_router(providers.router)


def _page(name: str) -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, name))


@app.get("/")
def index(p: Principal = Depends(current_principal)):
    """The chat UI — but only once the visitor is actually allowed to see it.

    anon      -> login page
    not yet   -> the onboarding flow
    onboarded -> chat.  Guests skip straight to chat with sample data.

    An unverified address does **not** block access. Verification is a reminder
    (a dismissible banner in the chat UI), not a gate: mail lands in spam, DNS
    lapses and free tiers throttle, and none of those should be able to lock a
    user out of their own account. The flag is still tracked and still governs
    account linking, which is where it actually carries weight.
    """
    if p.kind == "anon":
        return RedirectResponse(url="/login", status_code=303)
    if p.kind == "user" and not user_info(p.user_id)["onboarded"]:
        return RedirectResponse(url="/onboarding", status_code=303)
    return _page("index.html")


@app.get("/login")
def login_page():
    return _page("login.html")


@app.get("/onboarding")
def onboarding_page(user_id: int = Depends(require_user)):
    return _page("onboarding.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        has_vector = cur.fetchone() is not None
    return {"db": "reachable", "pgvector": has_vector}


@app.get("/me")
def me(p: Principal = Depends(current_principal)):
    """Who this browser is. The UI reads this to decide which screen to show and
    whether to render the 'User: …' label (signed-in) or a Login button (guest)."""
    if p.kind == "anon":
        return {"kind": "anon"}
    info = user_info(p.user_id)
    info["kind"] = p.kind
    if p.kind == "guest":  # a guest is browsing sample data, not an account
        info["name"] = None
        info["email"] = None
    return info
