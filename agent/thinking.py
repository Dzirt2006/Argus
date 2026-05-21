"""Three-tier classifier for thinking-mode routing.

Decides per-request whether Qwen3.5's reasoning mode should be enabled.
Simple/direct commands stay fast (no thinking). Complex multi-step
requests get deeper reasoning at the cost of more tokens.

Three tiers (cheap → expensive):
  1. Hard match — regex/length heuristics fire immediately → True.
  2. Hard skip — very short messages → False without LLM.
  3. Ambiguous — one short call to the fast (non-thinking) LLM.

Tier 3 results are cached so identical phrasings don't reclassify.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any

import structlog

# Words/phrases that suggest multi-step reasoning would help.
_THINKING_PATTERNS = [
    r"\bplan\b",
    r"\bcompare\b",
    r"\banalyze\b",
    r"\banalysis\b",
    r"\bexplain\b",
    r"\bfigure out\b",
    r"\bstep by step\b",
    r"\bpros and cons\b",
    r"\btrade.?offs?\b",
    r"\bweigh\b",
    r"\bdebug\b",
    r"\btroubleshoot\b",
    r"\bdiagnose\b",
    r"\boptimize\b",
    r"\brefactor\b",
    r"\barchitect\b",
    r"\bdesign\b",
    r"\bstrateg\w+\b",
    r"\bbased on\b.*\band\b",  # "based on calendar and weather"
    r"\bsummarize\b.*\band\b",  # "summarize X and Y"
    r"\bif .+ then\b",
    r"\bshould i\b",
    r"\bwhat would happen\b",
    r"\bhow would\b",
    r"\bwhy (does|did|is|are|do)\b",
]

_THINKING_RE = re.compile("|".join(_THINKING_PATTERNS), re.IGNORECASE)

# Minimum message length (chars) that alone triggers thinking,
# regardless of keywords. Long messages usually mean complex intent.
_LENGTH_THRESHOLD = 200

# Messages shorter than this many whitespace-separated tokens are
# treated as obviously trivial — skip the LLM fallback entirely.
_HARD_SKIP_WORD_THRESHOLD = 4

_CLASSIFIER_SYSTEM_PROMPT = (
    "You decide whether a user message requires multi-step reasoning "
    "to answer well. Answer with a single character: Y or N."
)
_CLASSIFIER_HUMAN_TEMPLATE = (
    "Does answering this question require multi-step reasoning?\n"
    "Message: {msg}\n"
    "Answer only Y or N."
)

_CACHE_MAX = 256
_cache: "OrderedDict[str, bool]" = OrderedDict()

_log = structlog.get_logger("agent.thinking")


def _cache_key(text: str) -> str:
    # Normalize so trivially-different phrasings ("Plan dinner", "plan dinner ",
    # "plan  dinner") share a cache entry.
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> bool | None:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: str, value: bool) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def _clear_cache_for_tests() -> None:
    _cache.clear()


def _llm_classify(user_message: str, llm_fast: Any) -> bool:
    """Ask the fast LLM whether the message needs reasoning. Defaults to False on anything ambiguous."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=_CLASSIFIER_HUMAN_TEMPLATE.format(msg=user_message)),
    ]
    # temperature=0 + tiny max_tokens keeps the extra call deterministic and cheap.
    # tool_choice="none" suppresses tool calls so the bound MCP tool schemas
    # aren't exercised here (they're still sent as input — unavoidable without
    # rebinding — but at least the response is guaranteed plain text).
    classifier = llm_fast.bind(temperature=0, max_tokens=4, tool_choice="none")
    try:
        response = classifier.invoke(messages)
    except Exception as exc:
        _log.warning("thinking_classifier_failed", error=str(exc))
        return False

    raw = getattr(response, "content", "")
    if isinstance(raw, list):
        # Some chat models return list-of-parts; concatenate text chunks.
        raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw)
    answer = str(raw).strip().lower()
    if not answer:
        # Empty content (e.g. tool-call response with no text) — default safely.
        return False
    return answer.startswith("y")


def should_think(user_message: str, llm_fast: Any | None = None) -> bool:
    """Return True if the request likely benefits from reasoning mode.

    Args:
        user_message: the latest human turn.
        llm_fast: optional fast (non-thinking) LLM used for the tier-3
            ambiguous fallback. When None — or when
            ``settings.thinking_llm_fallback_enabled`` is False — only
            the regex/length heuristics run and ambiguous inputs return False.
    """
    # Tier 1: hard match.
    if len(user_message) >= _LENGTH_THRESHOLD:
        return True
    if _THINKING_RE.search(user_message):
        return True

    # Tier 2: hard skip — very short messages are trivially direct.
    if len(user_message.split()) < _HARD_SKIP_WORD_THRESHOLD:
        return False

    # Tier 3: ambiguous — fall back to the fast LLM if available.
    from agent.config import settings

    if llm_fast is None or not settings.thinking_llm_fallback_enabled:
        return False

    key = _cache_key(user_message)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    verdict = _llm_classify(user_message, llm_fast)
    _cache_put(key, verdict)
    return verdict
