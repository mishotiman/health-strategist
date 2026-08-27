"""Shared question-answering logic used by both the /ask endpoint and the eval
harness, so they exercise exactly the same retrieval + generation path."""

from __future__ import annotations

import anthropic

from app.config import settings
from app.llm_cache import complete_text
from app.prompts import GUARDRAILS
from app.rag import retrieve

_claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

# Sonnet by default: near-Opus quality on grounded Q&A at ~1/2 the cost. The
# flagship agent (/chat) stays on Opus; override with RAG_GEN_MODEL.
GEN_MODEL = settings.rag_gen_model

SYSTEM_PROMPT = f"""You are a science-grounded health strategist.

Answer ONLY using the numbered research passages provided in the user message.
- Cite every claim with [n], where n is the passage number you drew it from.
- If the passages do not cover the question, say so plainly. Do not answer from
  outside knowledge.

{GUARDRAILS}
Be concise, practical, and honest about uncertainty in the evidence."""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {c['content']}\n"
        f"(Source: {c['title']}; {c['authors']}; {c['year']}. {c['source']})"
        for i, c in enumerate(chunks)
    )


def answer_question(question: str, k: int = 6, model: str | None = None) -> dict:
    """Retrieve passages and generate a grounded, cited answer.

    Returns the answer plus the retrieved chunks and the exact context string,
    so callers (the API, the eval harness) can inspect what the model saw.
    `model` overrides GEN_MODEL for this call (the eval harness uses it).
    """
    chunks = retrieve(question, k)
    context = build_context(chunks)
    answer = complete_text(
        _claude,
        model=model or GEN_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f"Research passages:\n\n{context}\n\nQuestion: {question}"}],
    )
    return {"answer": answer, "chunks": chunks, "context": context}

# question → retrieve(6 chunks) → build_context() → Claude → cited answer
#                                       ↓
#                         "[1] <passage text>
#                          (Source: title; authors; year. DOI)
#                          [2] <passage text> ..."