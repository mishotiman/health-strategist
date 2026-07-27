"""The agent endpoints — the conversational core of the product."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import agent
from app.deps import Principal, chat_principal, thread_for

log = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@router.post("/chat")
def chat(req: ChatRequest, p: Principal = Depends(chat_principal)):
    """Talk to the health-strategist agent as the logged-in user (or demo user).
    Non-streaming; kept as a fallback for /chat/stream."""
    return agent.run(p.user_id, req.message, thread_for(p, req.thread_id))


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, p: Principal = Depends(chat_principal)):
    """The streaming counterpart to /chat: newline-delimited JSON events
    (token…token…done) so the UI can reveal the answer as it's written. Sync
    generator — FastAPI runs it in a worker thread, and the whole agent stack
    is sync, so no async conversion is needed."""
    user_id = p.user_id
    thread = thread_for(p, req.thread_id)

    def events():
        try:
            for ev in agent.stream_run(user_id, req.message, thread):
                yield json.dumps(ev, default=str) + "\n"
        except Exception:  # noqa: BLE001 - surface any failure to the client
            log.exception("chat stream failed for user %s", user_id)
            yield json.dumps({"type": "error",
                              "message": "Something went wrong mid-response."}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
