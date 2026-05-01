"""Session-end summarizer.

One LLM call at the end of a conversation.  Extracts what the assistant
would want to remember next time — intents expressed, decisions made,
preferences revealed, tools used.  The summary is appended to the
`summaries` table via MemoryStore.

Uses thinking=False: summarization is high-recall-from-context, not
reasoning-heavy.  Keeping it fast matters because this runs at the exact
moment the user expects silence.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.config import settings
from agent.memory import get_store

log = structlog.get_logger("agent.summarizer")

_SUMMARY_SYSTEM = """You write compact memory notes for a home assistant.

Given a conversation transcript, produce 2-4 short sentences capturing
ONLY what would help the assistant in a future, unrelated conversation:
- User preferences, habits, or routines expressed.
- Decisions or plans that may come up again.
- Entities named (people, places, devices) worth remembering.
- Outcomes of actions taken.

STRICT RULES:
- Copy specific values (numbers, units, names, dates) VERBATIM from the
  transcript. Never paraphrase or convert quantities (e.g. "almost two
  meters" stays as "almost two meters", never "98 cm").
- Do not infer or assume attributes the user did not explicitly state.
- If unsure of a value, omit it rather than guess.

Do NOT include:
- Small talk, greetings, transient details.
- Tool call mechanics ("I used the weather tool").
- Anything already obvious from context.

If nothing is worth remembering, respond with exactly: SKIP"""

_FACTS_SYSTEM = """You extract durable facts from a conversation that the
user explicitly asked the assistant to remember, or that are clearly
stable attributes of the user (e.g. height, allergies, family members,
preferred temperatures).

Output format: one fact per line as `key: value`. Use snake_case keys.

Examples:
height: almost two meters
allergies: peanuts
preferred_thermostat: 70F
spouse_name: Maria

STRICT RULES:
- Copy specific values verbatim. Never convert units. Never invent values.
- Only include facts the user stated about themselves or their preferences.
- Skip transient state (today's weather, current location, what they just did).
- If nothing meets the bar, output exactly: NONE"""

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.model_name,
            base_url=settings.vllm_url,
            api_key="not-needed",
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    return _llm


def _transcript(messages: list) -> str:
    lines = []
    for m in messages:
        kind = getattr(m, "type", None)
        if kind == "human":
            lines.append(f"User: {m.content}")
        elif kind == "ai":
            text = (m.content or "").strip()
            if text:
                lines.append(f"Assistant: {text}")
    return "\n".join(lines)


def _count_user_turns(messages: list) -> int:
    return sum(1 for m in messages if getattr(m, "type", None) == "human")


def summarize_and_store(messages: list, session_id: str | None = None) -> str | None:
    """Summarize a session and persist via MemoryStore.  Returns the summary text, or None."""
    if not settings.memory_enabled:
        return None

    turns = _count_user_turns(messages)
    if turns < settings.memory_summary_min_turns:
        log.info("summary_skipped_short", turns=turns)
        return None

    transcript = _transcript(messages)
    if not transcript.strip():
        return None

    try:
        resp = _get_llm().invoke([
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=transcript),
        ])
    except Exception as e:
        log.error("summary_llm_failed", error=str(e))
        return None

    summary = (resp.content or "").strip()
    if not summary or summary.upper() == "SKIP":
        log.info("summary_skipped_empty")
        return None

    get_store().write_summary(summary, session_id=session_id)
    log.info("summary_stored", session_id=session_id, chars=len(summary))
    return summary


def _parse_facts(raw: str) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key and value:
            facts.append((key, value))
    return facts


def extract_and_store_facts(messages: list, session_id: str | None = None) -> list[tuple[str, str]]:
    """Extract durable user facts from the session and persist via set_fact."""
    if not settings.memory_enabled:
        return []

    transcript = _transcript(messages)
    if not transcript.strip():
        return []

    try:
        resp = _get_llm().invoke([
            SystemMessage(content=_FACTS_SYSTEM),
            HumanMessage(content=transcript),
        ])
    except Exception as e:
        log.error("facts_llm_failed", error=str(e))
        return []

    raw = (resp.content or "").strip()
    if not raw or raw.upper() == "NONE":
        log.info("facts_none", session_id=session_id)
        return []

    facts = _parse_facts(raw)
    if not facts:
        log.info("facts_unparseable", session_id=session_id, raw_chars=len(raw))
        return []

    store = get_store()
    for key, value in facts:
        store.set_fact(key, value, source="session_extraction")
    log.info("facts_stored", session_id=session_id, count=len(facts), keys=[k for k, _ in facts])
    return facts
