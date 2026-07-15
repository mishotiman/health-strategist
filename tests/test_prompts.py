"""The guardrail policy is shared, so it can't drift between the agent and the
fixed RAG pipeline."""

from app.prompts import GUARDRAILS


def test_guardrails_cover_the_essentials():
    text = GUARDRAILS.lower()
    assert "never diagnose" in text
    assert "red flag" in text
