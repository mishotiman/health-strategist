# Personal Health Strategist

An **agentic RAG** application that turns your own body data — WHOOP recovery/sleep,
bloodwork labs — into a science-grounded, personalized health strategy. Every answer
is grounded in peer-reviewed sports-science research **with citations**, and a
LangGraph agent decides which tools to use per question.

> Portfolio project demonstrating production-shaped AI engineering: RAG done properly,
> agent orchestration, a normalized data-ingestion layer, and **measured** quality
> (retrieval + agent evals in LangSmith).

---

## Architecture

```mermaid
flowchart TD
    U([User]) -->|POST /chat| A{{"LangGraph Agent<br/>(Claude Opus 4.8)"}}
    A -->|knowledge_search| R["RAG · pgvector<br/>curated OA corpus"]
    A -->|health_data| H[("health_metrics<br/>normalized")]
    A -->|memory| P[("profile · goals")]
    A -. always-on guardrail<br/>no diagnosis / defer .-> A
    A -->|grounded, cited answer| U

    subgraph ING["Ingestion — any source, one schema"]
        direction LR
        W["WHOOP OAuth2 (v2)"] --> N[Normalizer]
        B["Bloodwork PDF<br/>Sonnet extraction + unit conversion"] --> N
        N --> H
        DOC["Open-access papers<br/>Europe PMC"] --> EX["extract → chunk → voyage-3.5 embed"] --> R
    end

    subgraph EVAL["Eval — LangSmith"]
        direction LR
        E1["RAG: recall · faithfulness · citations"]
        E2["Agent: tool routing · behavior · guardrails"]
    end
```

## What it does

- **Agentic RAG.** A LangGraph agent (built on LangGraph v1's `create_agent`) orchestrates
  four tools — `knowledge_search` (RAG over the research corpus), `health_data` (your
  normalized WHOOP + bloodwork metrics), `workouts` (your logged WHOOP training sessions),
  and `memory` (your profile/goals). It decides which to call per question and loops until
  it can answer, with conversation state persisted in Postgres so threads survive restarts
  and scale-to-zero. A **guardrail policy** (never diagnose, defer red flags) wraps every
  response.
- **"Any source, one schema" ingestion.** WHOOP (live OAuth2) and bloodwork PDFs both
  normalize into a single `health_metrics` shape (`source · date · metric_type · value ·
  unit`). Bloodwork values are extracted from messy lab PDFs by Claude with **unit
  conversion** to canonical units.
- **Grounded answers with citations.** Retrieval over a curated open-access sports-science
  & physical-health corpus (**~4,200 papers / 275k passages** from Europe PMC, voyage-3.5
  embeddings in pgvector); answers cite their sources and admit when the corpus doesn't
  cover a topic.
- **Measured quality.** Golden-set evals in LangSmith for both the RAG pipeline and the
  agent's behavior.

## Eval baselines (LangSmith)

The corpus grew from a 13-paper hand-picked sample to **~4,200 open-access papers
(275k passages)**, and the judges were rebuilt after the v2 numbers turned out to be
partly fabricated (below). Current baselines, all `n=31` unless noted:

| RAG metric (`scripts/eval.py`) | v1 · 13 papers<br>Opus, digit-judge | v2 · 4.2k<br>Sonnet, digit-judge | **v3 · 4.2k<br>Sonnet, reasoned judge** | v3 · 4.2k<br>Opus, reasoned judge |
|---|---|---|---|---|
| recall@k | 1.00 | 0.75 | **0.75** (n=24) | 0.75 (n=24) |
| citation_validity | 1.00 | 1.00 | **1.00** | 1.00 |
| faithfulness | 0.92 | 0.52 | **0.97** | 1.00 |
| correctness | 0.88 | 0.81 | **0.87** | 0.77 |

Sonnet is the bolded column because it's what `/ask` actually serves; the Opus column
holds the generator fixed at v1's model to isolate it. `recall@k` scores only the 24
factual cases — guardrail and out-of-corpus questions have no `expected_source`, so
they're recorded unscored rather than silently counted.

**Agent baseline** (`scripts/eval_agent.py`, first run against the 4.2k corpus):
tool routing **1.00** (n=14) · behavior correctness **0.79** (n=19).

### The v2 numbers were wrong, and finding out was the useful part

`faithfulness` 0.52 was not a quality signal — it was a **broken evaluator**. The judge
was asked for a bare digit at `max_tokens=8`; when it instead reasoned through the claims
it ran out of budget mid-sentence and never wrote its verdict, and the parser then
scavenged the last `0` or `1` out of prose dense with `[1]`, `20%` and `1992`. At 256
tokens, **18 of 31 faithfulness scores were assigned that way** — near-random. Three
things came out of the fix:

- **The judges now reason before ruling**, and the rationale is attached to the score as
  the evaluator's comment, so a failing case explains itself in the trace. A real failure
  now reads like: *"the answer states β-alanine has 'very-low certainty' but passage [2]
  assigns it 'Moderate-certainty evidence' — a factual contradiction."*
- **An unparseable verdict scores `None`, never a guess.** A dropped case shows up as a
  smaller `n`, which is visible; a fabricated 0 is not. This is why every score above
  carries its `n`.
- **Groundedness was never the problem.** With the judge fixed, both generators score
  0.97–1.00 on the 4,200-paper corpus, against 0.92 for Opus on the original 13. The
  scale-up did not make answers less grounded.

**`recall@k` 0.75 is the one real signal.** It's identical across every configuration
above — which is expected, since retrieval runs before generation and can't be moved by
the generator. Retrieval over 4,200 papers is genuinely harder than over 13, where almost
anything retrieved was right. Part of the gap is still measurement: `recall@k` is the only
metric `expected_source` feeds, and those DOIs are pinned to the original 13 papers, so an
answer that finds a different-but-equally-valid paper scores 0.

**How noisy is a 31-case LLM-judged set?** Measured, not guessed: Opus `correctness` read
**0.87 → 0.81 → 0.77** across three runs over *identical cached answers*, with only the
judge re-sampled. Differences under ~10 points on sets this size are inside the noise.

### What these numbers do and don't prove

They're small, self-authored *development* sets — smoke tests, not generalization claims.
`faithfulness` / `correctness` are LLM-judged by a **different family** than the generator
(`claude-haiku-4-5` by default — independent of both the Sonnet RAG generator and the Opus
agent) to blunt self-preference bias. A high score on a set you wrote yourself mostly
measures that the system agrees with your own expectations — and the agent baseline shows
exactly how that bites: **3 of its 4 behavior failures are stale expectations, not agent
faults.** Cases written for the 13-paper corpus expect the agent to say beta-alanine, BCAAs
and marathon pacing aren't covered; the current corpus holds 54, 196 and 9 papers on them
respectively, so the agent answered correctly and was marked wrong for it.

Two next steps follow directly:

- **v3 recalibration of the golden sets** — widen `expected_source` to accept the several
  valid papers a topic now has, and retire the out-of-corpus cases the 4.2k corpus covers.
  Until that lands, the real size of the retrieval gap is unknown, so it gates the work
  below. The sets stand at 31 RAG / 19 agent cases, after a v2 pass added an **adversarial
  slice** (false-premise, near-miss / out-of-corpus, subtle red-flag, megadose-safety).
- **A retrieval-quality pass, measured** — with groundedness at 0.97+ the bottleneck is
  demonstrably retrieval, not generation. Cheapest check first (`RAG_HNSW_EF_SEARCH`, free
  via `--retrieval-only`), then a Voyage reranker (`app/rag.py` already exposes the
  `fetch_k` hook for fetch-wide-then-rerank) and hybrid search, re-measuring each.

### Running the harnesses cheaply

- **Free retrieval tier.** `scripts/eval.py --retrieval-only` scores `recall@k`
  over Voyage retrieval with **no Anthropic spend** (embeddings are a separate
  provider) — use it to iterate on retrieval/chunking for free.
- **Response cache.** Set `LLM_CACHE_DIR=.llm-cache` to memoize generations and
  judge verdicts by request hash, so re-runs while you tweak scoring cost nothing
  (and become deterministic). Leave unset in production.
- **Model config.** `RAG_GEN_MODEL`, `AGENT_MODEL`, `EVAL_GEN_MODEL`, and
  `EVAL_JUDGE_MODEL` (see `.env.example`) let you run cheap smoke passes on Haiku
  and reserve Opus for reported baselines.

## Tech stack

FastAPI · PostgreSQL + pgvector · Claude (Opus 4.8 agent, Sonnet 5 RAG answers +
bloodwork extraction, Haiku 4.5 eval judge) · Voyage `voyage-3.5` embeddings · LangGraph + LangChain + LangSmith ·
WHOOP OAuth2 · PyMuPDF · Docker Compose · a thin static chat UI (Next.js is a documented
future upgrade).

## Deployment

Running on **Azure Container Apps** (Poland Central) with managed PostgreSQL 16 + pgvector,
the image pulled from Azure Container Registry via managed identity, and secrets in the
Container Apps secret store. Full runbook — deploy loop, DB seeding, cost controls,
gotchas — in [DEPLOY.md](DEPLOY.md).

**Accounts.** Email + password (argon2id) or Google / Microsoft sign-in, with email
verification as a dismissible reminder (never a lockout) and a self-serve password reset. Sessions are
server-side and revocable. Anyone can also pick **"Try Health Strategist now"** for a guest
session that reads a sample dataset and saves nothing. Every write endpoint derives the user
from the session, and WHOOP is connected per-account.

**Cost guardrails.** The endpoints that spend on the app's own API keys are rate-limited
(`app/ratelimit.py`): the agent per signed-in account and, more tightly, per IP for guests;
the public `/ask` and `/search` per IP. A stranger who finds the URL can't turn it into
unbounded Anthropic spend, and the ceilings are tunable via `RATE_*` env vars.

**Observability.** Container logs stream to a Log Analytics workspace (queryable via KQL),
and a scheduled GitHub Action (`scripts/prod_smoke_eval.py`) runs a daily black-box
smoke-eval against the live app — liveness, retrieval recall, and (on demand) `/ask`
citation validity — so the "measured quality" story extends past the local golden sets to
*production*. Details in [DEPLOY.md](DEPLOY.md#observability-logs--a-production-canary).

## Quickstart

```bash
# 1. configure secrets
cp .env.example .env          # then fill in the API keys

# 2. start the stack (FastAPI + Postgres/pgvector)
docker compose up -d --build

# 3. build the corpus
#    a) apply DB migrations (idempotent, tracked in schema_migrations; safe on a fresh DB
#       too — it baselines the schema init_db.sql already created)
docker compose exec api python scripts/migrate.py
#    b) fetch open-access papers from Europe PMC (data/corpus_topics.yml is the recipe).
#       --slice N caps each pillar for a smoke test; drop it for the full ~4k run.
docker compose exec api python scripts/fetch_corpus.py --slice 15
#    c) extract → chunk/ingest → embed
docker compose exec api python scripts/extract_text.py
docker compose exec api python scripts/ingest.py
docker compose exec api python scripts/embed_chunks.py
#    d) build the ANN index once embeddings exist
cat scripts/index_corpus.sql | docker compose exec -T db psql postgresql://phs:phs@db:5432/phs

# 4. open the chat UI
open http://localhost:8000
```

Connect WHOOP: register `http://localhost:8000/whoop/callback` as a redirect URI in your
WHOOP app, then visit `http://localhost:8000/whoop/connect?user_id=1`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` · `/login` · `/onboarding` | Chat UI, sign-in page, first-run setup |
| POST | `/chat` | Talk to the agent (`{message, thread_id?}`) |
| POST | `/ask` | Fixed RAG pipeline: retrieve → cited answer |
| POST | `/search` | Retrieval only (no LLM) |
| POST | `/auth/register` · `/auth/login` · `/auth/logout` | Accounts |
| GET | `/auth/oauth/{google\|microsoft}/start` · `/callback` | External sign-in |
| GET | `/auth/verify` · POST `/auth/forgot-password` · `/auth/reset-password` | Email flows |
| POST | `/auth/guest` | Start a read-only guest session |
| POST | `/metrics` · GET `/metrics/{id}` | Ingest / query normalized metrics |
| POST | `/upload/bloodwork` | Upload a lab PDF → extracted metrics |
| GET | `/whoop/connect` · `/whoop/callback` | WHOOP OAuth2 (per account) |

## Evals

```bash
docker compose exec api python scripts/eval.py         # RAG
docker compose exec api python scripts/eval_agent.py   # agent
```

## Project layout

```
app/        FastAPI app, agent, tools, ingestion, WHOOP, bloodwork, static UI
scripts/    init/migrate SQL, extract/ingest/embed, eval harnesses
data/       sources.md (corpus manifest) + eval golden sets
```

## Notes

Corpus source files, the derived plain text, and `.env` are gitignored — the repo tracks
code, the source manifest, and the eval sets. Open-access (CC BY) research only.
