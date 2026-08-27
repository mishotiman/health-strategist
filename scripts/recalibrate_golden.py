#!/usr/bin/env python
"""Recalibrate the RAG golden set's `expected_source` for the current corpus.

    docker compose exec -T api python scripts/recalibrate_golden.py          # report
    docker compose exec -T api python scripts/recalibrate_golden.py --apply  # rewrite

Why this exists: `expected_source` was one hand-picked DOI per question, chosen
when the corpus was 13 papers. At 4,200 papers several papers legitimately answer
a given question, so `recall@k` scores 0 whenever retrieval surfaces a
different-but-equally-valid one — under-reporting retrieval quality.

Method: POOLING, as IR benchmarks (TREC) have built relevance judgments for
decades. The lazy alternative — "accept whatever retrieval returns" — would make
recall@k 1.00 by construction and measure nothing, so instead:

  1. Pool candidates per question from TWO independent retrieval methods: dense
     vector search (top POOL_VECTOR) and lexical ILIKE matching on the question's
     content words (top POOL_LEXICAL). Two methods, so the pool is not merely a
     mirror of what one embedder prefers.
  2. Collapse the pool to documents. recall@k scores whole DOCUMENTS, so the
     ground-truth question is whether a paper contains ANY supporting passage,
     not whether its single best-matching chunk does. Judging one chunk
     rejected papers whose relevant section simply was not the closest match
     to the question - a meta-analysis of cold water and hypertrophy was
     rejected for 'not mentioning hypertrophy'. So each candidate is judged
     over its top CHUNKS_PER_DOC chunks, short-circuiting on the first hit.
  3. Judge on CONTENT ALONE - does this passage support the case's
     expected_facts? - with rank deliberately withheld from the judge.

A paper is admitted because it answers the question, never because retrieval
ranked it highly, so recall@k keeps asking something the retriever can fail:
out of the papers that genuinely answer this, does the top-k surface one?

The originally pinned DOI is always retained (it was hand-verified when the case
was written), but it is judged too, and a WARN is printed when it no longer looks
supporting - that is a signal about the case, not something to paper over.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic

from app.db import get_connection
from app.llm_cache import complete_text
from app.rag import embed_query

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "golden_set.jsonl")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "claude-haiku-4-5")

POOL_VECTOR = 40     # dense candidates per question
POOL_LEXICAL = 20    # lexical candidates per question
MAX_CANDIDATES = 12  # documents actually sent to the judge, per question
CHUNKS_PER_DOC = 3   # passages examined per candidate document (see step 2)

judge = anthropic.Anthropic()

_STOP = {"the", "a", "an", "of", "for", "to", "and", "or", "is", "are", "do", "does",
         "how", "what", "which", "much", "many", "my", "i", "me", "should", "can",
         "in", "on", "at", "per", "with", "be", "it", "that", "this", "there", "if",
         "best", "good", "any", "more", "most", "day", "week", "have", "has", "you"}


def content_terms(question: str, limit: int = 4) -> list[str]:
    """The question's distinctive words, longest first - longer tokens carry more
    topical signal than short ones ("creatine" beats "take")."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", question.lower())
    ranked = sorted({w for w in words if w not in _STOP}, key=len, reverse=True)
    return ranked[:limit]


def pool_candidates(question: str) -> tuple[dict[str, dict], str]:
    """Candidate documents from dense + lexical retrieval, keyed by DOI, plus the
    question's embedding (reused to pick each candidate's passages).

    `score` orders the report only; it never reaches the judge.
    """
    pool: dict[str, dict] = {}
    qvec = str(embed_query(question))
    terms = content_terms(question)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.source, d.title, 1 - (c.embedding <=> %s::vector)
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector LIMIT %s
            """,
            (qvec, qvec, POOL_VECTOR),
        )
        for doc_id, source, title, sim in cur.fetchall():
            prev = pool.get(source)
            if prev is None or sim > prev["score"]:
                pool[source] = {"title": title, "doc_id": doc_id,
                                "score": float(sim), "via": "dense"}

        # Lexical arm: an independent view of the corpus, so the pool is not just
        # a mirror of the embedder. Unindexed ILIKE means a seq scan, but this
        # runs once per question in an offline script - correctness over speed.
        if terms:
            clause = " AND ".join(["c.content ILIKE %s"] * len(terms))
            cur.execute(
                f"""
                SELECT DISTINCT d.id, d.source, d.title
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE {clause} LIMIT %s
                """,
                (*[f"%{t}%" for t in terms], POOL_LEXICAL),
            )
            for doc_id, source, title in cur.fetchall():
                if source not in pool:
                    pool[source] = {"title": title, "doc_id": doc_id,
                                    "score": 0.0, "via": "lexical"}
                elif pool[source]["via"] == "dense":
                    pool[source]["via"] = "both"
    return pool, qvec


def best_chunks(doc_id: int, qvec: str, n: int = CHUNKS_PER_DOC) -> list[str]:
    """This document's n passages closest to the question. Filtered to one
    document, so it bypasses the HNSW index - fine, since a paper holds only
    tens of chunks."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT content FROM chunks
            WHERE document_id = %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector LIMIT %s
            """,
            (doc_id, qvec, n),
        )
        return [row[0] for row in cur.fetchall()]


_RUBRIC = """You are building ground truth for a retrieval benchmark.

Given a QUESTION, the FACTS a correct answer must convey, and one PASSAGE from a
research paper, decide whether that passage genuinely supports those facts - i.e.
whether a system that retrieved this passage could answer the question correctly.

Score 1 when the passage states or directly evidences the facts. Paraphrase is
fine, and it need not cover every fact: substantial support for the core claim is
enough.
Score 0 when the passage is merely on the same broad topic, mentions the subject
in passing, or discusses something adjacent without evidencing the facts.

Be strict. This list becomes the answer key, so a passage that only shares
vocabulary with the question is a 0."""


def judge_candidate(question: str, facts: list[str], title: str,
                    passage: str) -> bool | None:
    """True/False if the judge ruled; None when it produced no parseable verdict."""
    prompt = (
        f"{_RUBRIC}\n\n"
        f"QUESTION: {question}\n\n"
        f"FACTS A CORRECT ANSWER MUST CONVEY:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"PASSAGE (from '{title}'):\n{passage[:2500]}\n\n"
        "Think briefly, then finish with a line of exactly 'VERDICT: 1' or "
        "'VERDICT: 0'. That line is mandatory and must be last."
    )
    text = complete_text(judge, model=JUDGE_MODEL, max_tokens=512,
                         messages=[{"role": "user", "content": prompt}])
    m = re.search(r"VERDICT:\s*([01])", text)
    return None if not m else m.group(1) == "1"


def main() -> None:
    ap = argparse.ArgumentParser(description="Recalibrate golden-set expected sources")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite golden_set.jsonl (default: report only)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(GOLDEN, encoding="utf-8") if line.strip()]
    widened = unscored = warned = 0

    for r in rows:
        pinned = r.get("expected_source") or (r.get("expected_sources") or [None])[0]
        if not pinned:
            continue  # guardrail / out-of-scope cases carry no answer key

        pool, qvec = pool_candidates(r["question"])
        ranked = sorted(pool.items(), key=lambda kv: -kv[1]["score"])[:MAX_CANDIDATES]
        if pinned in pool and pinned not in {s for s, _ in ranked}:
            ranked.append((pinned, pool[pinned]))  # always judge the pinned paper

        accepted: list[str] = []
        pinned_ok: bool | None = None
        for source, meta in ranked:
            supported = ruled = False
            for passage in best_chunks(meta["doc_id"], qvec):
                verdict = judge_candidate(r["question"], r["expected_facts"],
                                          meta["title"], passage)
                if verdict is None:
                    unscored += 1
                    continue
                ruled = True
                if verdict:
                    supported = True
                    break  # one supporting passage is enough to admit the paper
            if not ruled:
                continue  # no parseable verdict on any passage - leave it out
            if supported:
                accepted.append(source)
            if source == pinned:
                pinned_ok = supported

        # The pinned DOI stays regardless: it was hand-verified, and dropping it
        # on the strength of one judge call would be a worse error than keeping a
        # slightly weak entry. The WARN surfaces it for human review instead.
        sources = [pinned] + [s for s in accepted if s != pinned]
        r["expected_sources"] = sources
        r.pop("expected_source", None)

        flag = ""
        if pinned_ok is False:
            flag = "   WARN pinned DOI judged NOT supporting"
            warned += 1
        print(f"{r['id']:28} {len(sources):2} source(s)  "
              f"(pool {len(pool):3}, judged {len(ranked):2}){flag}")
        if len(sources) > 1:
            widened += 1

    print(f"\n{widened} cases gained additional valid sources, "
          f"{warned} pinned DOIs flagged, {unscored} judge calls unparseable.")

    if args.apply:
        with open(GOLDEN, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {GOLDEN}.")
        print("Delete the LangSmith dataset (phs-golden-v2) so it re-seeds from this file.")
    else:
        print("Report only - re-run with --apply to write.")


if __name__ == "__main__":
    main()
