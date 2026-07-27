#!/usr/bin/env python3
"""Production smoke-eval — grade the **deployed** app over HTTP, on a schedule.

This is the black-box counterpart to the LangSmith golden-set harness
(`scripts/eval.py` / `scripts/eval_agent.py`, which grade the pipeline *locally*).
Here we hit the live URL and ask a blunter question: is production actually up,
retrieving relevant evidence, and returning grounded, cited answers *right now*?

It is deliberately standalone — **standard library only, no `app` imports** — so
it runs anywhere (a bare GitHub Actions runner, your laptop) with nothing
installed, and never drags in the agent, the model SDKs, or a database.

Three checks, cheapest first:

  health      GET  /health   — liveness. Free.
  retrieval   POST /search   — does a known topic return relevant passages?
                               Voyage embeddings only, so **no Anthropic spend**.
  answers     POST /ask      — is the RAG answer non-empty, grounded (mentions
                               the topic) and *validly cited* (every [n] points
                               at a real source)? Costs Anthropic on the
                               deployed key, so it's opt-in via --with-answers.

Exit code is 0 only if every check passed, so CI turns a regression red.

    python scripts/prod_smoke_eval.py                       # free tier
    python scripts/prod_smoke_eval.py --with-answers        # + the /ask checks
    python scripts/prod_smoke_eval.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "https://phs-api.orangehill-97462476.polandcentral.azurecontainerapps.io"

# A tiny, corpus-robust probe set kept separate from data/eval/* on purpose:
# these are load-bearing production canaries, not the quality golden set. Each
# `expect` term is one the topic's papers are overwhelmingly likely to contain,
# so the check stays green across ordinary corpus churn (a smoke test, not a
# regression test on exact wording).
PROBES = [
    {"topic": "protein",   "question": "How much protein per day maximizes muscle strength gains?",
     "expect": ["protein"]},
    {"topic": "creatine",  "question": "Is creatine effective for strength and how should it be dosed?",
     "expect": ["creatine"]},
    {"topic": "caffeine",  "question": "Does caffeine before exercise improve performance?",
     "expect": ["caffeine"]},
    {"topic": "sleep",     "question": "How does sleep affect athletic performance and recovery?",
     "expect": ["sleep"]},
    {"topic": "vitamin_d", "question": "Does vitamin D supplementation improve muscle strength?",
     "expect": ["vitamin d", "vitamin-d"]},
]

CITATION = re.compile(r"\[(\d+)\]")


class Reporter:
    """Collects pass/fail lines and prints an aligned report."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def record(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((ok, name, detail))
        return ok

    def failed(self) -> int:
        return sum(1 for ok, _, _ in self.rows if not ok)

    def print(self) -> None:
        width = max((len(n) for _, n, _ in self.rows), default=0)
        for ok, name, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            line = f"  [{mark}] {name.ljust(width)}"
            print(line + (f"  - {detail}" if detail else ""))


def _post(url: str, payload: dict, timeout: float) -> tuple[int, dict | None]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    return _send(req, timeout)


def _get(url: str, timeout: float) -> tuple[int, dict | None]:
    return _send(urllib.request.Request(url, method="GET"), timeout)


def _send(req: urllib.request.Request, timeout: float) -> tuple[int, dict | None]:
    """(status, parsed-json-or-None). Network/HTTP errors return a status and
    None rather than raising, so one dead endpoint can't abort the whole run."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"    (network error: {e})", file=sys.stderr)
        return 0, None


def check_health(base: str, rep: Reporter, timeout: float) -> None:
    status, data = _get(f"{base}/health", timeout)
    ok = status == 200 and isinstance(data, dict) and data.get("status") == "ok"
    rep.record(ok, "health", f"HTTP {status}" + ("" if ok else " (expected 200 + status:ok)"))


def check_retrieval(base: str, probe: dict, k: int, rep: Reporter, timeout: float) -> None:
    status, data = _post(f"{base}/search", {"question": probe["question"], "k": k}, timeout)
    results = (data or {}).get("results") or []
    if status != 200 or not results:
        rep.record(False, f"retrieval:{probe['topic']}", f"HTTP {status}, {len(results)} results")
        return
    # is the topic actually represented in what came back?
    hay = " ".join((r.get("title", "") + " " + r.get("preview", "")) for r in results).lower()
    grounded = any(term in hay for term in probe["expect"])
    rep.record(grounded, f"retrieval:{probe['topic']}",
               f"{len(results)} results" + ("" if grounded else f", none mention {probe['expect']}"))


def check_answer(base: str, probe: dict, k: int, rep: Reporter, timeout: float) -> None:
    status, data = _post(f"{base}/ask", {"question": probe["question"], "k": k}, timeout)
    answer = (data or {}).get("answer") or ""
    sources = (data or {}).get("sources") or []
    if status != 200 or not answer:
        rep.record(False, f"answer:{probe['topic']}", f"HTTP {status}, empty={not answer}")
        return
    cites = [int(n) for n in CITATION.findall(answer)]
    # every [n] must point at a real returned source — the citation-validity
    # check, run deterministically without a judge model
    valid_cites = bool(cites) and all(1 <= n <= len(sources) for n in cites)
    grounded = any(term in answer.lower() for term in probe["expect"])
    ok = valid_cites and grounded
    detail = f"{len(sources)} sources, {len(cites)} citations"
    if not cites:
        detail += ", no [n] citations"
    elif not valid_cites:
        detail += ", a citation points nowhere"
    if not grounded:
        detail += f", doesn't mention {probe['expect']}"
    rep.record(ok, f"answer:{probe['topic']}", detail)


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-eval the deployed PHS app.")
    ap.add_argument("--url", default=os.environ.get("APP_BASE_URL", DEFAULT_URL),
                    help="Base URL of the deployed app (or set APP_BASE_URL).")
    ap.add_argument("--with-answers", action="store_true",
                    help="Also grade /ask answers (uses the deployed Anthropic key).")
    ap.add_argument("--k", type=int, default=6, help="Passages to retrieve per query.")
    ap.add_argument("--timeout", type=float, default=45.0, help="Per-request timeout (s).")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    rep = Reporter()
    started = time.time()

    tier = "health + retrieval + answers" if args.with_answers else "health + retrieval (free)"
    print(f"PHS production smoke-eval -> {base}")
    print(f"tier: {tier}\n")

    check_health(base, rep, args.timeout)
    for probe in PROBES:
        check_retrieval(base, probe, args.k, rep, args.timeout)
    if args.with_answers:
        for probe in PROBES:
            check_answer(base, probe, args.k, rep, args.timeout)

    rep.print()
    failed = rep.failed()
    total = len(rep.rows)
    print(f"\n{total - failed}/{total} checks passed in {time.time() - started:.1f}s")
    if failed:
        print(f"FAILED: {failed} check(s) regressed.")
        return 1
    print("All production checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
