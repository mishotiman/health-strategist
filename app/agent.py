"""The LangGraph agent — orchestrates three tools to produce grounded,
personalized health strategies.

Tools:
  knowledge_search — RAG over the sports-science corpus (evidence + citations)
  health_data      — the user's normalized WHOOP + bloodwork metrics
  memory           — the user's profile (goals, physical data, injuries)

The guardrail is an always-on policy in the system prompt (never diagnose,
defer red flags), not a skippable tool. Conversation state is kept per thread
so the agent remembers earlier turns. Runs auto-trace to LangSmith.
"""

from __future__ import annotations

import json
import os

# Trace agent trajectories to LangSmith (EU workspace).
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.citations import format_chunks_with_citations
from app.db import get_connection
from app.ingestion import query_metrics
from app.prompts import GUARDRAILS
from app.rag import retrieve

# Opus is the flagship agent. Override with AGENT_MODEL (e.g. claude-haiku-4-5)
# for cheap smoke runs of the agent eval.
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")

SYSTEM_PROMPT = f"""You are the user's Personal Health Strategist. You turn their \
own body data plus peer-reviewed sports-science research into practical, \
personalized guidance.

How to work:
- Use `knowledge_search` for any claim about training, nutrition, supplements, \
sleep, recovery, or stress. Cite the sources it returns, e.g. [1], [2].
- Use `health_data` to personalize with the user's WHOOP metrics (recovery, HRV, \
resting HR, sleep) and bloodwork (vitamin D, ferritin, etc.).
- Use `memory` to read the user's goals, physical data, and injuries.
- Ground every recommendation in retrieved evidence or the user's own data. If \
the corpus doesn't cover something, say so instead of guessing.
- When the user asks about their OWN data (a metric, a trend, their profile), \
lead with the direct answer first — the actual numbers or trend — then add any \
research or context after.
- Report durations (sleep, naps) in hours and minutes, e.g. "6 h 44 min", not \
decimal hours. Use each row's `value_display` field when present.

{GUARDRAILS}
Be concise, practical, and honest about uncertainty."""


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
    tracked separately — do not treat a nap as the night's sleep. Bloodwork:
    vitamin_d, ferritin, testosterone, crp, etc. Pass a metric_type to filter to
    one, or leave empty for all recent metrics."""
    user_id = config["configurable"]["user_id"]
    rows = query_metrics(user_id, metric_type or None, limit=40)
    for r in rows:  # give the model a ready-made "6h 44m" for hour-based metrics
        if r.get("unit") == "h" and r.get("value") is not None:
            r["value_display"] = _fmt_hours(r["value"])
    return json.dumps(rows, default=str) if rows else "No health metrics on record."


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
    tools=[knowledge_search, health_data, memory],
    prompt=SYSTEM_PROMPT,
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
