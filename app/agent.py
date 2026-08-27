"""The LangGraph agent — orchestrates three tools to produce grounded,
personalized health strategies.

Tools:
  knowledge_search — RAG over the sports-science corpus (evidence + citations)
  health_data      — the user's normalized WHOOP + bloodwork metrics
  workouts         — the user's logged WHOOP training sessions
  memory           — the user's profile (goals, physical data, injuries)

The guardrail is an always-on policy in the system prompt (never diagnose,
defer red flags), not a skippable tool. Conversation state is kept per thread
so the agent remembers earlier turns. Runs auto-trace to LangSmith.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading

# Trace agent trajectories to LangSmith (EU workspace).
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.citations import format_chunks_with_citations
from app.config import settings
from app.db import get_connection
from app.ingestion import query_metrics, tracked_metric_types
from app.prompts import GUARDRAILS
from app.rag import retrieve
from app.workouts import query_workouts

# Opus is the flagship agent. Override with AGENT_MODEL (e.g. claude-haiku-4-5)
# for cheap smoke runs of the agent eval.
MODEL = settings.agent_model

SYSTEM_PROMPT = f"""You are the user's AI Personal Health Strategist. You turn their \
own body data plus peer-reviewed sports-science research into practical, \
personalized guidance to achieve their goals, or discover practical tips \
to improve their health.

How to work:
- Call tools silently. Do NOT narrate or announce tool use ("Let me check…", \
"I'll look up…"). Produce prose only in your final answer — the response is \
streamed to the user as you write it, so any pre-tool narration would show up \
as noise before the real answer.
- Use `knowledge_search` for any claim about training, nutrition, supplements, \
sleep, recovery, or stress. Cite the sources it returns, e.g. [1], [2].
- Use `health_data` to personalize with the user's WHOOP metrics (recovery, HRV, \
resting HR, sleep) and bloodwork (vitamin D, ferritin, etc.).
- Use `workouts` for the user's logged training sessions (WHOOP workouts): sport, \
duration, strain, heart rate, calories, distance. Use it whenever the question is \
about what or how they've trained; `health_data` is daily recovery/sleep, not \
individual sessions.
- Use `memory` to read the user's goals, physical data, and injuries.
- Ground every recommendation in retrieved evidence or the user's own data.
- `knowledge_search` ALWAYS returns passages, so their presence is NOT evidence \
the corpus covers the topic — similarity is relative, and an unrelated question \
still returns its nearest neighbours. Read what came back and check it addresses \
the actual question. If it only covers adjacent subjects (altitude training when \
asked about training masks; adult lifting when asked about a 12-year-old), say \
the corpus doesn't cover it and stop — do not assemble an answer out of \
near-misses.
- Never report, estimate, or infer a metric the user's data does not contain. If \
a metric isn't tracked, say so plainly and name what is.
- When the user asks about their OWN data (a metric, a trend, their profile), \
lead with the direct answer first — the actual numbers or trend — then add any \
research or context after.
- Report durations (sleep, naps) in hours and minutes, e.g. "6 h 44 min", not \
decimal hours. Use each row's `value_display` field when present.
- Maintain a light, encouraging, and occasionally humorous tone.
- Supplement your answers (wherever possible, but not with every answer) with \
simple relevant science-based practical health-related tips, grounded \
in the research corpus, for the user to try right away.

{GUARDRAILS}
Be concise, practical, and honest about uncertainty."""


@dynamic_prompt
def _prompt_with_today(request) -> str:
    """The system prompt with today's date stamped on, recomputed every model
    call so the agent never has to guess 'today' / 'yesterday' / 'this week'.
    Uses the server's local date; swap in the user's timezone if near-midnight
    precision ever matters. (A create_agent middleware — the LangGraph v1
    replacement for the old `prompt` callable.)"""
    today = dt.date.today()
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Today's date is {today.isoformat()} ({today.strftime('%A')}). Use it "
        "for any date reasoning; never guess the date. The user's most recent "
        "WHOOP data can lag today by a day or two, so do not assume the latest "
        "logged entry is from today."
    )


# --------------------------------------------------------------------------- #
# Prompt caching
# --------------------------------------------------------------------------- #
# Each chat turn makes several LLM calls (decide-a-tool → read result → …
# → write the answer), and each re-sends the whole prefix: tool schemas, the
# system prompt, and every prior message. Caching lets calls after the first
# re-read that prefix at ~10% of the input price instead of paying full price
# each time — the single biggest lever on /chat cost.
#
# Mechanism: a `cache_control` breakpoint on a content block caches everything
# before it (render order is tools → system → messages). We put the breakpoint
# on the LAST message, so the cached unit is the entire prefix. Two facts drive
# that placement:
#   * Anthropic only caches a prefix of >= 1024 tokens (4096 on Opus 4.8), and
#     tools + system here is only ~1.7k — under Opus's floor, so a system-only
#     breakpoint would silently never cache. The win has to come from prefixes
#     large enough to matter (retrieved passages, metrics, longer threads) —
#     exactly the expensive turns; small turns just skip caching at no cost.
#   * A *top-level* `cache_control` request param (Anthropic's "automatic"
#     caching) was measured to no-op through this LangChain→SDK stack, so we
#     place the block-level breakpoint ourselves.
# The breakpoint is added only to the copy sent to the model; the checkpointer
# still stores clean messages (no cache_control persisted into thread state).
_EPHEMERAL = {"cache_control": {"type": "ephemeral"}}


def _with_cache_breakpoint(message):
    """A copy of `message` with a cache breakpoint on its last content block.
    Leaves the original (checkpointed) message untouched."""
    content = message.content
    if isinstance(content, str):
        new_content = [{"type": "text", "text": content, **_EPHEMERAL}]
    elif isinstance(content, list) and content:
        new_content = list(content)
        last = new_content[-1]
        new_content[-1] = ({**last, **_EPHEMERAL} if isinstance(last, dict)
                           else {"type": "text", "text": str(last), **_EPHEMERAL})
    else:
        return message  # nothing to attach a breakpoint to
    return message.model_copy(update={"content": new_content})


@wrap_model_call
def _cache_prefix(request, handler):
    """Cache the request prefix (tools + system + prior turns) by putting the
    breakpoint on the last message before each model call."""
    messages = request.messages
    if messages:
        request = request.override(
            messages=[*messages[:-1], _with_cache_breakpoint(messages[-1])])
    return handler(request)


def _fmt_hours(value: float) -> str:
    """Decimal hours -> 'Xh Ym' (e.g. 6.73 -> '6h 44m')."""
    total_min = round(float(value) * 60)
    h, m = divmod(total_min, 60)
    return f"{h}h {m}m" if h and m else (f"{h}h" if h else f"{m}m")


def _configurable(config: RunnableConfig | None) -> dict:
    """The `configurable` dict of a tool's config — the one access pattern every
    tool below uses. run()/stream_run() always populate it; the fallback covers
    a tool invoked bare (tests, ad-hoc eval probes)."""
    return (config or {}).get("configurable") or {}


# Vector search over 275k passages never comes back empty: it returns the nearest
# neighbours, and cosine similarity is a relative ranking signal, not a calibrated
# relevance score. Measured on this corpus, "nootropics" (0 papers) scores 0.638
# at top-1 while "sleep extension" (well covered) scores 0.627 — the distributions
# overlap, so NO similarity threshold separates covered from uncovered topics.
# The signal has to come from reading the passages, so the tool result says so
# outright rather than letting six confident-looking passages imply a coverage
# that isn't there. This is what the agent eval caught: asked about training
# masks (0 papers), it answered from adjacent altitude-training passages.
_COVERAGE_CAVEAT = (
    "\n\n---\n"
    "COVERAGE NOTE: this search always returns the corpus's nearest passages, even "
    "when the corpus holds nothing on the topic asked about. Their presence is not "
    "evidence of coverage. Check that the passages above address THIS specific "
    "question; if they only cover adjacent subjects, state plainly that the corpus "
    "does not cover it instead of answering from the near-misses."
)


@tool # @tool decorator (part of LangChain), generates per-tool JSON schema (from docstring) + signature that Claude can read.
def knowledge_search(query: str, config: RunnableConfig = None) -> str:
    """Search the peer-reviewed sports-science research corpus. Use for any claim
    about training, nutrition, supplements, sleep, recovery, or stress. Returns
    passages, each prefixed with a stable [n] citation number — cite claims with
    that number. Passages are ALWAYS returned, including for topics the corpus
    does not cover, so verify they address the question before relying on them."""
    chunks = retrieve(query, k=6)
    if not chunks:
        return "No relevant passages found in the corpus."

    # A per-turn registry (shared via config) gives each unique source a stable
    # global citation number, so [n] means the same paper across the whole turn.
    reg = _configurable(config).get("citations")
    if reg is None:
        reg = []
    return format_chunks_with_citations(chunks, reg) + _COVERAGE_CAVEAT


@tool
def health_data(metric_type: str = "", config: RunnableConfig = None) -> str:
    """Look up the user's recent health metrics. WHOOP: recovery_score, hrv_rmssd,
    resting_hr, sleep_hours, nap_hours, sleep_efficiency, respiratory_rate, spo2,
    skin_temp. sleep_hours is the overnight sleep; nap_hours is daytime naps,
    tracked separately — do not treat a nap as the night's sleep. Bloodwork
    (numeric): full CBC + differential, hemoglobin, biochemistry (creatinine,
    alt, ast, ggt, electrolytes, iron), inflammation (crp, hs_crp, calprotectin),
    vitamins (vitamin_d, vitamin_b12, folate, …), minerals (magnesium, zinc, …).
    Some results are qualitative (microbiology like clostridium_difficile_*,
    quantiferon_tb) — those carry a text_value ("negative"/"positive") instead of
    a numeric value. Pass a metric_type to filter, or leave empty for all recent."""
    user_id = _configurable(config)["user_id"]
    rows = query_metrics(user_id, metric_type or None, limit=40)
    if not rows and metric_type:
        # Naming the tracked metrics is what prevents a fabricated answer. Told
        # only "no data", the agent was observed reporting a plausible number for
        # a metric the user never recorded (a VO2max); told which metrics exist,
        # it can give the honest answer instead.
        available = tracked_metric_types(user_id)
        if available:
            return (f"No '{metric_type}' data is recorded for this user. Metrics "
                    f"actually tracked: {', '.join(available)}. Tell the user that "
                    f"{metric_type} is not tracked — never estimate, infer or "
                    "substitute a value for it.")
    for r in rows:  # give the model a ready-made "6h 44m" for hour-based metrics
        if r.get("unit") == "h" and r.get("value") is not None:
            r["value_display"] = _fmt_hours(r["value"])
    return json.dumps(rows, default=str) if rows else "No health metrics on record."


@tool
def workouts(config: RunnableConfig = None) -> str:
    """Look up the user's recent logged WHOOP workouts (training sessions). Each
    row has: sport (e.g. running, weightlifting), workout_date, start/end time,
    duration_min, strain (WHOOP 0–21), avg_hr, max_hr, calories (kcal), and
    distance_m (cardio only). Use for any question about the user's actual
    training — what they did, how hard, how long, how far, and trends across
    sessions. This is the training log; health_data holds daily recovery/sleep."""
    user_id = _configurable(config)["user_id"]
    rows = query_workouts(user_id, limit=25)
    return json.dumps(rows, default=str) if rows else "No workouts on record."


@tool
def memory(config: RunnableConfig = None) -> str:
    """Read the user's profile: goals, sex, birth year, height, weight, and
    injuries. Use to tailor advice to who they are and what they want."""
    user_id = _configurable(config)["user_id"]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT goals, sex, birth_year, height_cm, weight_kg, injuries "
            "FROM profiles WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        return "No profile on record for this user."
    cols = ["goals", "sex", "birth_year", "height_cm", "weight_kg", "injuries"]
    return json.dumps(dict(zip(cols, row)), default=str)


_llm = ChatAnthropic(model=MODEL, max_tokens=2000)

# Built lazily on first use (not at import): eval scripts and offline tests
# import this module without a database, and a transiently unreachable DB at
# boot self-heals on the next request instead of crashing the process.
_agent = None
_agent_lock = threading.Lock()


def _build_agent():
    """Conversation state lives in Postgres (the same DB as everything else), so
    threads survive restarts and scale-to-zero, and any replica sees them.
    (MemorySaver, the previous checkpointer, kept them in process RAM: every
    deploy or scale-down wiped every user's conversation.)"""
    pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1, max_size=4,
        # PostgresSaver requires autocommit + dict rows on its connections.
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # creates its checkpoint tables; no-op when they exist
    return create_agent(
        _llm,
        tools=[knowledge_search, health_data, workouts, memory],
        middleware=[_prompt_with_today, _cache_prefix],
        checkpointer=checkpointer,
    )


def _get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = _build_agent()
    return _agent


def _text(content) -> str:
    if isinstance(content, list):  # Anthropic may return a list of blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def run(user_id: int, message: str, thread_id: str | None = None) -> dict:
    thread_id = thread_id or f"user-{user_id}"
    citations: list[dict] = []  # filled by knowledge_search during this turn
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id,
                               "citations": citations}}
    result = _get_agent().invoke({"messages": [HumanMessage(content=message)]}, config=config)

    messages = result["messages"]
    tools_used = [
        tc["name"]
        for m in messages
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    return {
        "answer": _text(messages[-1].content),
        "thread_id": thread_id,
        "tools_used": tools_used,
        "sources": citations,  # [{n, title, url}] for linking [n] citations
    }


def stream_run(user_id: int, message: str, thread_id: str | None = None):
    """Same turn as run(), but yields the answer incrementally so the UI can
    reveal it as it's written. Emits plain dicts:

        {"type": "token", "text": ...}   an increment of the answer
        {"type": "done",  ...}           final metadata (sources, tools_used)

    stream_mode=["updates", "messages"] gives both signals in one pass: the
    "messages" mode streams per-token AIMessageChunks (the text), while
    "updates" surfaces each node's output so we can harvest which tools ran.
    Citations are filled by knowledge_search during the tool phase, which
    precedes the final answer, so `citations` is complete by the "done" event.
    """
    thread_id = thread_id or f"user-{user_id}"
    citations: list[dict] = []
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id,
                               "citations": citations}}
    tools_seen: set[str] = set()
    final_answer = ""

    for mode, chunk in _get_agent().stream(
            {"messages": [HumanMessage(content=message)]},
            config=config, stream_mode=["updates", "messages"]):
        if mode == "messages":
            msg_chunk, _meta = chunk
            # Stream ONLY assistant text. "messages" mode also emits ToolMessage
            # chunks (the raw retrieved passages) — streaming those would dump
            # the research corpus into the bubble before the answer. The model
            # calls tools silently (system prompt), so tool-calling turns carry
            # no prose and only the final answer streams.
            if isinstance(msg_chunk, AIMessageChunk):
                text = _text(msg_chunk.content)
                if text:
                    yield {"type": "token", "text": text}
        elif mode == "updates":
            for node_out in chunk.values():
                turn_called_tool = False
                for m in (node_out or {}).get("messages", []):
                    tool_calls = getattr(m, "tool_calls", None) or []
                    for tc in tool_calls:
                        tools_seen.add(tc["name"])
                    if tool_calls:
                        turn_called_tool = True
                    # The final answer is the last AI message with no tool call —
                    # the same message run() returns as `answer`.
                    if getattr(m, "type", "") == "ai" and not tool_calls:
                        text = _text(getattr(m, "content", ""))
                        if text:
                            final_answer = text
                # A turn that ends in a tool call means anything streamed during
                # it was pre-tool narration ("Let me look that up…") — tell the
                # client to discard it so only the real answer remains.
                if turn_called_tool:
                    yield {"type": "reset"}

    yield {"type": "done", "thread_id": thread_id, "answer": final_answer,
           "tools_used": sorted(tools_seen), "sources": citations}
