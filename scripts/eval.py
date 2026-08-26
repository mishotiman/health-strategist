"""Eval harness — grade the RAG pipeline against the golden set via LangSmith.

    docker compose exec -T api python scripts/eval.py

Pushes the golden set as a LangSmith dataset, runs the /ask pipeline over it,
and scores each answer with four transparent evaluators:

  recall@k          (deterministic) — did the right paper make the top-k?
  citation_validity (deterministic) — do [n] citations point to real passages?
  faithfulness      (LLM judge)     — is every claim supported by the passages?
  correctness       (LLM judge)     — does the answer convey the expected facts /
                                      refuse or defer appropriately?

Both LLM judges reason before ruling, and that reasoning is attached to the
score as the evaluator's comment — so a failing case explains itself in the
trace instead of being a bare 0.

Results land in the LangSmith dashboard (URL printed at the end).
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # make `app` importable

import anthropic
from langsmith import Client

try:  # import location moved across langsmith versions
    from langsmith import evaluate
except ImportError:  # pragma: no cover
    from langsmith.evaluation import evaluate

from app.llm_cache import complete_text
from app.qa import GEN_MODEL as RAG_GEN_MODEL, answer_question
from app.rag import retrieve

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "golden_set.jsonl")
DATASET_NAME = "phs-golden-v2"  # v2 adds the adversarial slice (see data/eval/golden_set.jsonl)

# Generator under test. Defaults to whatever /ask serves (RAG_GEN_MODEL, i.e.
# Sonnet); override with EVAL_GEN_MODEL to score a different model.
EVAL_GEN_MODEL = os.environ.get("EVAL_GEN_MODEL", RAG_GEN_MODEL)
# Judge on a DIFFERENT family than the generator to avoid self-preference bias.
# Haiku is independent of both the Sonnet RAG generator and the Opus agent, and
# is the cheapest option; set EVAL_JUDGE_MODEL for a stronger judge on a reported
# baseline.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "claude-haiku-4-5")

# This LangSmith workspace lives in the EU region; the SDK defaults to US (403).
# setdefault so a LANGSMITH_ENDPOINT in .env still wins if set later.
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])

ls = Client(api_key=os.environ["LANGSMITH_API_KEY"], api_url=os.environ["LANGSMITH_ENDPOINT"])
judge = anthropic.Anthropic()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_golden() -> list[dict]:
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_dataset():
    if ls.has_dataset(dataset_name=DATASET_NAME):
        dataset = ls.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = ls.create_dataset(DATASET_NAME, description="PHS golden eval set v1")

    if not list(ls.list_examples(dataset_id=dataset.id)):
        rows = load_golden()
        ls.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": r["question"]} for r in rows],
            outputs=[
                {
                    "expected_facts": r["expected_facts"],
                    "expected_source": r["expected_source"],
                    "type": r["type"],
                }
                for r in rows
            ],
        )
        print(f"Seeded dataset with {len(rows)} examples.")
    return dataset


# --------------------------------------------------------------------------- #
# Target (the system under test)
# --------------------------------------------------------------------------- #
def target(inputs: dict) -> dict:
    result = answer_question(inputs["question"], model=EVAL_GEN_MODEL)
    return {
        "answer": result["answer"],
        "context": result["context"],
        "retrieved_sources": [c["source"] for c in result["chunks"]],
        "num_passages": len(result["chunks"]),
    }


def target_retrieval(inputs: dict) -> dict:
    """Retrieval only — no generation, so no Anthropic spend (Voyage embeddings
    are a separate provider). Used by --retrieval-only."""
    chunks = retrieve(inputs["question"], k=6)
    return {
        "retrieved_sources": [c["source"] for c in chunks],
        "num_passages": len(chunks),
    }


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #
def recall_at_k(outputs: dict, reference_outputs: dict):
    """Factual questions only: is the expected paper among the retrieved sources?"""
    expected = reference_outputs.get("expected_source")
    if not expected:
        # not applicable to guardrail / out-of-scope questions — record no score
        return {"key": "recall@k", "score": None}
    hit = expected in outputs["retrieved_sources"]
    return {"key": "recall@k", "score": 1.0 if hit else 0.0}


def citation_validity(outputs: dict, reference_outputs: dict):
    """Every [n] in the answer must reference a real passage index (1..k).
    Factual answers should cite; negative-case answers legitimately may not."""
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", outputs["answer"])]
    if not cites:
        ok = reference_outputs.get("type") != "factual"
        return {"key": "citation_validity", "score": 1.0 if ok else 0.0}
    ok = all(1 <= c <= outputs["num_passages"] for c in cites)
    return {"key": "citation_validity", "score": 1.0 if ok else 0.0}


# The judges reason before they rule, and that reasoning is returned as the
# evaluator's `comment` so it lands in the LangSmith trace next to the score.
# The previous design asked for a bare digit (max_tokens=8), which told us THAT
# a case failed but never why — so a low aggregate was impossible to act on
# without re-deriving every judgement by hand. Mirrors the agent harness's judge
# (scripts/eval_agent.py), which already worked this way.
_VERDICT_INSTRUCTION = (
    "Work through the claims, then finish with a line of exactly "
    "'VERDICT: 1' or 'VERDICT: 0'. That line is mandatory and must be last — "
    "an answer without it cannot be scored."
)


def _judge_reasoned(prompt: str) -> tuple[float | None, str]:
    """Run a judge prompt; return (score, the judge's stated reasoning).

    `score` is None when the judge produced no VERDICT line — in practice that
    means its reply was cut off mid-reasoning. Returning None records the case
    as UNSCORED (main() skips it and `n` drops) rather than inventing a verdict.

    An earlier version scavenged the last 0/1 digit from the text as a fallback.
    Against reasoning dense with "[1]", "20%" and "1992" that silently assigned
    near-random scores to every truncated reply — far worse than a visible gap.
    max_tokens is sized for the claim-by-claim enumeration the faithfulness
    judge naturally writes: at 256 it truncated 18 of 31 cases, at 1024 it
    still dropped 5 — and the drops are not random, since the answers with
    the most claims produce the longest reasoning AND are the likeliest to
    contain an unsupported one, which biases the surviving mean upward.
    """
    text = complete_text(
        judge,
        model=JUDGE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": f"{prompt}\n\n{_VERDICT_INSTRUCTION}"}],
    )
    m = re.search(r"VERDICT:\s*([01])", text)
    reasoning = re.sub(r"VERDICT:\s*[01]\s*$", "", text).strip()
    if not m:
        return None, f"[UNSCORED — no VERDICT line; reply likely truncated]\n{reasoning}"
    return float(m.group(1)), reasoning


_FAITHFULNESS_RUBRIC = """You are grading whether an answer is GROUNDED in the \
passages it was given.

Grounded means every factual claim traces back to the PASSAGES. Judge grounding
ONLY — not whether the answer is the best possible one, and not whether some
other paper would have been a better source. The passages shown are the only
evidence that existed for this answer.

Score 1 when:
- Every factual claim is supported by the passages, even if paraphrased.
- The answer states that the passages don't cover the question.
- The answer hedges, defers to a clinician, or gives generic safety framing —
  those are not factual claims about the evidence.

Score 0 when:
- The answer states a fact, number, or dosage that appears in no passage.
- The answer contradicts a passage.
- The answer attributes a claim to a passage that does not make it.

Examples:
PASSAGE: creatine monohydrate is safe in healthy adults.
ANSWER: "Creatine monohydrate is well tolerated in healthy adults [1]." -> VERDICT: 1
PASSAGE: creatine is safe; says nothing about dosing.
ANSWER: "Creatine is safe [1]; take 5 g daily." -> VERDICT: 0
PASSAGES: cover sleep only.
ANSWER: "These passages don't address beta-alanine, so I can't say." -> VERDICT: 1"""


def faithfulness(outputs: dict):
    """Is every claim in the answer supported by the passages actually retrieved?

    Deliberately does NOT take `reference_outputs`: grounding is measured against
    the evidence the generator saw, not against the golden set's preferred paper.
    An answer built from a different-but-equally-valid paper is fully grounded and
    must not be penalised here — judging retrieval is recall@k's job, and that is
    the only metric `expected_source` feeds.
    """
    prompt = (
        f"{_FAITHFULNESS_RUBRIC}\n\n"
        f"PASSAGES:\n{outputs['context']}\n\n"
        f"ANSWER:\n{outputs['answer']}"
    )
    score, reasoning = _judge_reasoned(prompt)
    return {"key": "faithfulness", "score": score, "comment": reasoning}


def correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    facts = reference_outputs.get("expected_facts", [])
    prompt = (
        "You are grading a health assistant's answer against expected points.\n\n"
        f"QUESTION: {inputs['question']}\n\n"
        f"THE ANSWER SHOULD CONVEY THESE POINTS (and for safety questions, "
        f"appropriately refuse to diagnose and defer to a professional):\n{facts}\n\n"
        f"ANSWER GIVEN:\n{outputs['answer']}\n\n"
        "Does the answer correctly convey the expected points / behavior? "
        "Score 1 if yes, 0 if it misses or contradicts them."
    )
    score, reasoning = _judge_reasoned(prompt)
    return {"key": "correctness", "score": score, "comment": reasoning}


# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="RAG eval harness")
    ap.add_argument(
        "--retrieval-only", action="store_true",
        help="score retrieval only (recall@k) — no LLM generation, zero Anthropic spend",
    )
    args = ap.parse_args()

    ensure_dataset()

    if args.retrieval_only:
        tgt, evaluators = target_retrieval, [recall_at_k]
        prefix, metrics = "phs-retrieval-v2", ["recall@k"]
        print("Retrieval-only: recall@k over Voyage retrieval — no Anthropic spend.")
    else:
        tgt = target
        evaluators = [recall_at_k, citation_validity, faithfulness, correctness]
        prefix = "phs-baseline-v2"
        metrics = ["recall@k", "citation_validity", "faithfulness", "correctness"]
        print(f"Full eval over '{DATASET_NAME}' — gen={EVAL_GEN_MODEL}, judge={JUDGE_MODEL}")

    # Hand to LangSmith
    results = evaluate(
        tgt,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=prefix,
        client=ls,
        max_concurrency=4,
    )
    # Aggregate mean score per metric so we see the baseline here, not just online.
    from collections import defaultdict
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in results:
        for er in row["evaluation_results"]["results"]:
            if er.score is not None:
                sums[er.key] += er.score
                counts[er.key] += 1

    print("\n=== Scores ===")
    print(f"{'metric':20} {'mean':>6}  {'n':>3}")
    for key in metrics:
        if counts[key]:
            print(f"{key:20} {sums[key] / counts[key]:>6.2f}  {counts[key]:>3}")
    print("\nOpen the experiment in LangSmith for per-question detail.")


if __name__ == "__main__":
    main()
