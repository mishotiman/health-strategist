"""Opt-in on-disk cache for Anthropic Messages calls.

Enabled only when LLM_CACHE_DIR is set (unset in production). Meant for eval and
dev iteration: re-running a harness while you tweak scoring/aggregation replays
identical generations and judge verdicts from disk instead of re-paying for them.
Cache key is a hash of the full request (model + system + messages + params), so
any change to the prompt or model misses and regenerates.

A side benefit for evals: cached runs are deterministic, so score changes reflect
your harness edits, not model sampling noise.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _cache_dir() -> str | None:
    return os.environ.get("LLM_CACHE_DIR") or None


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def complete_text(client, *, model: str, max_tokens: int, messages: list,
                  system: str | None = None, **kwargs) -> str:
    """Anthropic Messages call that returns concatenated text, with optional
    on-disk caching. Behaves like a normal call when LLM_CACHE_DIR is unset."""
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages,
               "system": system, **kwargs}
    directory = _cache_dir()
    path = Path(directory) / f"{_key(payload)}.json" if directory else None

    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]

    call_kwargs = dict(model=model, max_tokens=max_tokens, messages=messages, **kwargs)
    if system is not None:
        call_kwargs["system"] = system
    resp = client.messages.create(**call_kwargs)
    text = "".join(b.text for b in resp.content if b.type == "text")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text, "request": payload}, default=str),
                        encoding="utf-8")
    return text
