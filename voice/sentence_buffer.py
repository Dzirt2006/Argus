"""Sentence splitter for streaming TTS.

Incrementally accumulates text chunks from a token stream and emits complete
sentences as they become available. Designed to be cheap and conservative:

- A sentence ends on ``.``, ``!``, or ``?``...
- ...unless the preceding token is a known abbreviation (``Dr.``, ``e.g.``, ...)
- ...unless the preceding "word" is a single letter (initials, ``a.m.``,
  ``U.S.``, ``e.g.``)
- ...unless the period is between two digits (decimal numbers like ``1.5``)
- An ellipsis (``...`` or unicode ``…``) is treated as a single terminator

When feed sees a single period at the very end of the current buffer, it holds
it back — the next chunk could be a digit (making this a decimal), another
letter (part of a multi-dot abbreviation), or end-of-stream. ``flush`` commits
whatever is left.

The buffer never speaks chain-of-thought. The caller must filter the stream
to the visible ``content`` channel before feeding it here.
"""

from __future__ import annotations

import re

_ABBREVIATIONS: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "vs", "etc", "e.g", "i.e", "cf", "fig",
    "inc", "ltd", "co", "corp",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "a.m", "p.m", "u.s", "u.k",
})

# Captures the word ending right before a period. Allows interior dots so
# "e.g" and "a.m" are matched as single tokens.
_TRAIL_WORD = re.compile(r"([A-Za-z](?:\.?[A-Za-z])*)\.$")

_HARD_TERMINATORS = frozenset({"!", "?"})


class SentenceBuffer:
    """Accumulate streamed text, emit complete sentences in order."""

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, text: str) -> list[str]:
        """Append ``text`` and return any newly-completed sentences."""
        if not text:
            return []

        self._buf += text
        sentences: list[str] = []

        i = 0
        sent_start = 0
        n = len(self._buf)

        while i < n:
            ch = self._buf[i]

            if ch in _HARD_TERMINATORS:
                end = i + 1
                while end < n and self._buf[end] in _HARD_TERMINATORS:
                    end += 1
                while end < n and self._buf[end].isspace():
                    end += 1
                sentences.append(self._buf[sent_start:end].strip())
                sent_start = end
                i = end
                continue

            if ch == "…":
                end = i + 1
                if end >= n:
                    # Hold: caller may want trailing space consumed before split.
                    break
                while end < n and self._buf[end].isspace():
                    end += 1
                sentences.append(self._buf[sent_start:end].strip())
                sent_start = end
                i = end
                continue

            if ch == ".":
                if self._is_decimal_period(i):
                    i += 1
                    continue

                # Ellipsis run: `..` or longer. Treat as one terminator.
                if i + 1 < n and self._buf[i + 1] == ".":
                    end = i + 1
                    while end < n and self._buf[end] == ".":
                        end += 1
                    if end >= n:
                        # Buffer ended mid-run; more dots may follow.
                        break
                    while end < n and self._buf[end].isspace():
                        end += 1
                    sentences.append(self._buf[sent_start:end].strip())
                    sent_start = end
                    i = end
                    continue

                # Single period. We need at least one char after to decide
                # decimal vs. abbreviation vs. sentence end.
                if i + 1 >= n:
                    break

                trailing = self._buf[sent_start:i + 1]
                m = _TRAIL_WORD.search(trailing)
                if m:
                    word = m.group(1)
                    # Single-letter words before a period are almost never end
                    # of sentence (e.g. "e.g.", "a.m.", "U.S.", initials).
                    if len(word) == 1 or word.lower() in _ABBREVIATIONS:
                        i += 1
                        continue

                end = i + 1
                while end < n and self._buf[end].isspace():
                    end += 1
                sentences.append(self._buf[sent_start:end].strip())
                sent_start = end
                i = end
                continue

            i += 1

        self._buf = self._buf[sent_start:]
        return [s for s in sentences if s]

    def flush(self) -> str | None:
        """Return the remaining buffer as a final sentence, or None if empty."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining or None

    def _is_decimal_period(self, i: int) -> bool:
        if i == 0 or i + 1 >= len(self._buf):
            return False
        return self._buf[i - 1].isdigit() and self._buf[i + 1].isdigit()
