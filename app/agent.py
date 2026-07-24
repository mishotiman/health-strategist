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

# Trace agent trajectories to LangSmith (EU workspace).
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.citations import format_chunks_with_citations
from app.db import get_connection
from app.ingestion import query_metrics
from app.prompts import GUARDRAILS
from app.rag import retrieve
from app.workouts import query_workouts

# Opus is the flagship agent. Override with AGENT_MODEL (e.g. claude-haiku-4-5)
# for cheap smoke runs of the agent eval.
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")

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
- Ground every recommendation in retrieved evidence or the user's own data. If \
the corpus doesn't cover something, say so instead of guessing.
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


def _prompt_with_today(state) -> list:
    """The system prompt with today's date stamped on, recomputed every call so
    the agent never has to guess 'today' / 'yesterday' / 'this week'. Returned
    fresh each turn (not persisted into thread state). Uses the server's local
    date; swap in the user's timezone if near-midnight precision ever matters."""
    today = dt.date.today()
    dated = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Today's date is {today.isoformat()} ({today.strftime('%A')}). Use it "
        "for any date reasoning; never guess the date. The user's most recent "
        "WHOOP data can lag today by a day or two, so do not assume the latest "
        "logged entry is from today."
    )
    return [SystemMessage(content=dated)] + state["messages"]


def _fmt_hours(value: float) -> str:
    """Decimal hours -> 'Xh Ym' (e.g. 6.73 -> '6h 44m')."""
    total_min = round(float(value) * 60)
    h, m = divmod(total_min, 60)
    return f"{h}h {m}m" if h and m else (f"{h}h" if h else f"{m}m")


@tool
def knowledge_search(query: str, config: RunnableConfig = None) -> str:
    """Search the peer-reviewed sports-science research corpus. Use for any claim
    about training, nutrition, supplements, sleep, recovery, or stress. Returns
    passages, each prefixed with a stable [n] citation number — cite claims with
    that number."""
    chunks = retrieve(query, k=6)
    if not chunks:
        return "No relevant passages found in the corpus."

    # A per-turn registry (shared via config) gives each unique source a stable
    # global citation number, so [n] means the same paper across the whole turn.
    reg = ((config or {}).get("configurable") or {}).get("citations")
    if reg is None:
        reg = []
    return format_chunks_with_citations(chunks, reg)


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
    user_id = config["configurable"]["user_id"]
    rows = query_metrics(user_id, metric_type or None, limit=40)
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
    user_id = config["configurable"]["user_id"]
    rows = query_workouts(user_id, limit=25)
    return json.dumps(rows, default=str) if rows else "No workouts on record."


@tool
def memory(config: RunnableConfig = None) -> str:
    """Read the user's profile: goals, sex, birth year, height, weight, and
    injuries. Use to tailor advice to who they are and what they want."""
    user_id = config["configurable"]["user_id"]
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
_checkpointer = MemorySaver() # persist conversation state per thread in RAM, so the agent remembers earlier turns
_agent = create_react_agent(
    _llm,
    tools=[knowledge_search, health_data, workouts, memory],
    prompt=_prompt_with_today,
    checkpointer=_checkpointer,
)


def _text(content) -> str:
    if isinstance(content, list):  # Anthropic may return a list of blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def run(user_id: int, message: str, thread_id: str | None = None) -> dict:
    thread_id = thread_id or f"user-{user_id}"
    citations: list[dict] = []  # filled by knowledge_search during this turn
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id,
                               "citations": citations}}
    result = _agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)

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

    for mode, chunk in _agent.stream(
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
