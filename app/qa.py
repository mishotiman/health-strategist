"""Shared question-answering logic used by both the /ask endpoint and the eval
harness, so they exercise exactly the same retrieval + generation path."""

from __future__ import annotations

import anthropic

from app.rag import retrieve

_claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

GEN_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a science-grounded health strategist.

Answer ONLY using the numbered research passages provided in the user message.
- Cite every claim with [n], where n is the passage number you drew it from.
- If the passages do not cover the question, say so plainly. Do not answer from
  outside knowledge.
- You are not a doctor: never diagnose, and defer to a qualified professional
  for anything that looks like a medical red flag.
Be concise, practical, and honest about uncertainty in the evidence."""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {c['content']}\n"
        f"(Source: {c['title']}; {c['authors']}; {c['year']}. {c['source']})"
        for i, c in enumerate(chunks)
    )


def answer_question(question: str, k: int = 6) -> dict:
    """Retrieve passages and generate a grounded, cited answer.

    Returns the answer plus the retrieved chunks and the exact context string,
    so callers (the API, the eval harness) can inspect what the model saw.
    """
    chunks = retrieve(question, k)
    context = build_context(chunks)
    message = _claude.messages.create(
        model=GEN_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f"Research passages:\n\n{context}\n\nQuestion: {question}"}],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")
    return {"answer": answer, "chunks": chunks, "context": context}
