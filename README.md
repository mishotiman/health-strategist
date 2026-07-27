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
(275k passages)**, so the RAG harness was re-run against it (`scripts/eval.py`, Sonnet 5
generator, Haiku 4.5 judge). Watching the numbers move across that scale-up is the point:

| RAG metric (`scripts/eval.py`) | v1 · 13-paper corpus, Opus gen | v2 · 4.2k corpus, Sonnet gen |
|---|---|---|
| recall@k | 1.00 | 0.75 |
| citation_validity | 1.00 | 1.00 |
| faithfulness | 0.92 | 0.52 |
| correctness | 0.88 | 0.81 |

Agent behavior (`scripts/eval_agent.py`, last run): tool routing **0.95** · behavior
correctness **0.93** · guardrail cases **100% pass**. This predates the corpus scale-up and
is due a re-run against the expanded set.

**Why the drop is expected — and useful.** Retrieval over 4,200 papers is genuinely harder
than over 13, so `recall@k` falling from a near-trivial 1.00 to 0.75 is a real test rather
than a regression. The `faithfulness` drop to 0.52 is largely a **measurement artifact**:
the golden set's `expected_source` DOIs were pinned to the original 13 papers, so the judge
now penalizes answers that cite different-but-equally-valid papers the larger corpus
surfaces. Two concrete next steps fall straight out of this:

- **v3 recalibration of the golden set** — widen `expected_source` to accept the several
  valid papers a topic now has, and add questions covering ground only the 4.2k corpus
  reaches. The sets currently stand at 31 RAG / 19 agent cases, after a v2 pass added an
  **adversarial slice** (false-premise, near-miss / out-of-corpus, subtle red-flag, and
  megadose-safety cases).
- **A retrieval-quality pass is now a *measured* need, not a hunch** — low groundedness is
  the signal to add a Voyage reranker (`app/rag.py` already exposes the `fetch_k` hook for
  fetch-wide-then-rerank) and hybrid search, then re-measure.

**What these numbers do and don't prove.** They're small, self-authored *development* sets —
smoke tests, not generalization claims. `faithfulness` / `correctness` are LLM-judged by a
**different family** than the generator (`claude-haiku-4-5` by default — independent of both
the Sonnet RAG generator and the Opus agent) to blunt self-preference bias; judges are still
noisy, so spot-check the LangSmith traces and use `EVAL_JUDGE_MODEL=claude-opus-4-8` for a
headline number. A high score on a set you wrote yourself mostly measures that the system
agrees with your own expectations.

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
