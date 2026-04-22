"""Heuristic classifier for memory retrieval routing.

Same shape as agent/thinking.py: default off, flip on when the user message
mentions past context or personal state the assistant would have stored.

Over-retrieval is worse than under-retrieval here — irrelevant memories in
the prompt confuse small models, so this classifier biases toward 'skip'.
"""

import re

_RECALL_PATTERNS = [
    r"\bremember\b",
    r"\brecall\b",
    r"\bforget\b",
    r"\blast time\b",
    r"\blast (week|month|year|night)\b",
    r"\byesterday\b",
    r"\bthis morning\b",
    r"\bearlier\b",
    r"\bwe (talked|discussed|said|agreed|decided)\b",
    r"\byou (said|told|mentioned|suggested)\b",
    r"\bi (said|told|mentioned|asked|wanted|told you|prefer|usually|always|never)\b",
    r"\bmy (preference|usual|favorite|routine|schedule|default)\b",
    r"\bwhat (did|was|were) (i|we|you)\b",
    r"\bwho (is|are|was|were)\b.*\b(my|our)\b",
    r"\blike (i )?(always|usually)\b",
    r"\bas (always|usual|before)\b",
]

_RECALL_RE = re.compile("|".join(_RECALL_PATTERNS), re.IGNORECASE)


def should_retrieve(user_message: str) -> bool:
    """Return True if the turn likely needs long-term memory injected."""
    if not user_message:
        return False
    return bool(_RECALL_RE.search(user_message))
