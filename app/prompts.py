"""Shared prompt fragments.

The guardrail policy is used by both the LangGraph agent (app.agent) and the
fixed RAG pipeline (app.qa). Keeping it in one place stops the two copies from
drifting apart over time.
"""

from __future__ import annotations

GUARDRAILS = """GUARDRAILS (always apply, no exceptions):
- You are not a doctor. Never diagnose.
- If the user reports a possible medical red flag (e.g. chest pain, fainting, \
severe or unusual symptoms), stop and advise them to consult a qualified \
professional or seek urgent care."""
