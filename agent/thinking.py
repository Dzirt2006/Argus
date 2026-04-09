"""Heuristic classifier for thinking mode routing.

Decides per-request whether Qwen3.5's reasoning mode should be enabled.
Simple/direct commands stay fast (no thinking). Complex multi-step
requests get deeper reasoning at the cost of more tokens.
"""

import re

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


def should_think(user_message: str) -> bool:
    """Return True if the request likely benefits from reasoning mode."""
    if len(user_message) >= _LENGTH_THRESHOLD:
        return True
    if _THINKING_RE.search(user_message):
        return True
    return False
