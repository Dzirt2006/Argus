"""Unit tests for the streaming-TTS sentence splitter and chunk filtering."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk

from voice.pipeline import _chunk_text
from voice.sentence_buffer import SentenceBuffer


def _drain(buf: SentenceBuffer, chunks: list[str]) -> list[str]:
    """Feed every chunk in order then flush; return all emitted sentences."""
    out: list[str] = []
    for c in chunks:
        out.extend(buf.feed(c))
    tail = buf.flush()
    if tail:
        out.append(tail)
    return out


# -- terminator basics --------------------------------------------------------


def test_single_complete_sentence_lives_until_flush() -> None:
    # A trailing period with no follow-up char can't yet be classified as
    # sentence-end vs. abbreviation, so it stays buffered until flush.
    buf = SentenceBuffer()
    assert buf.feed("Hello there.") == []
    assert buf.flush() == "Hello there."


def test_two_sentences_one_chunk() -> None:
    buf = SentenceBuffer()
    out = buf.feed("Hello there. How are you?")
    assert out == ["Hello there.", "How are you?"]
    assert buf.flush() is None


def test_three_terminator_types() -> None:
    out = _drain(SentenceBuffer(), ["Wait. Really! Are you sure?"])
    assert out == ["Wait.", "Really!", "Are you sure?"]


def test_partial_sentence_held_in_buffer() -> None:
    buf = SentenceBuffer()
    assert buf.feed("Hello") == []
    # The space after the period is what lets feed commit the sentence.
    assert buf.feed(" there. ") == ["Hello there."]
    assert buf.flush() is None


def test_token_level_streaming() -> None:
    text = "The cat sat on the mat. Then it left! Was anyone watching? Maybe."
    buf = SentenceBuffer()
    out: list[str] = []
    for ch in text:
        out.extend(buf.feed(ch))
    tail = buf.flush()
    if tail:
        out.append(tail)
    assert out == [
        "The cat sat on the mat.",
        "Then it left!",
        "Was anyone watching?",
        "Maybe.",
    ]


# -- abbreviations ------------------------------------------------------------


def test_abbreviation_dr_does_not_split() -> None:
    out = _drain(SentenceBuffer(), ["Dr. Smith arrived. He waved."])
    assert out == ["Dr. Smith arrived.", "He waved."]


@pytest.mark.parametrize("abbr", ["Mr.", "Mrs.", "Ms."])
def test_abbreviation_titles_do_not_split(abbr: str) -> None:
    out = _drain(SentenceBuffer(), [f"{abbr} Doe is here. Welcome!"])
    assert out == [f"{abbr} Doe is here.", "Welcome!"]


def test_abbreviation_eg() -> None:
    out = _drain(SentenceBuffer(), ["Bring snacks, e.g. chips. Drinks too."])
    assert out == ["Bring snacks, e.g. chips.", "Drinks too."]


def test_abbreviation_ie() -> None:
    out = _drain(SentenceBuffer(), ["Anything cold, i.e. ice cream. Got it?"])
    assert out == ["Anything cold, i.e. ice cream.", "Got it?"]


def test_partial_at_chunk_boundary_abbreviation() -> None:
    buf = SentenceBuffer()
    assert buf.feed("Visit Dr") == []
    assert buf.feed(".") == []
    # Period inside abbreviation must not split.
    assert buf.feed(" Jones today. ") == ["Visit Dr. Jones today."]
    assert buf.feed("Done.") == []
    assert buf.flush() == "Done."


# -- decimals -----------------------------------------------------------------


def test_decimal_number_does_not_split() -> None:
    out = _drain(SentenceBuffer(), ["Pi is about 3.14 today. Cool."])
    assert out == ["Pi is about 3.14 today.", "Cool."]


def test_decimal_one_point_five() -> None:
    out = _drain(SentenceBuffer(), ["Add 1.5 cups of flour. Mix well."])
    assert out == ["Add 1.5 cups of flour.", "Mix well."]


def test_partial_at_chunk_boundary_decimal() -> None:
    buf = SentenceBuffer()
    assert buf.feed("It costs 3") == []
    assert buf.feed(".") == []
    assert buf.feed("50 dollars. Cheap!") == ["It costs 3.50 dollars.", "Cheap!"]


# -- ellipsis -----------------------------------------------------------------


def test_ellipsis_treated_as_one_terminator() -> None:
    out = _drain(SentenceBuffer(), ["Wait... what? Really."])
    assert out == ["Wait...", "what?", "Really."]


def test_unicode_ellipsis() -> None:
    out = _drain(SentenceBuffer(), ["Hmm… odd. Yes."])
    assert out == ["Hmm…", "odd.", "Yes."]


def test_ellipsis_split_across_chunks() -> None:
    buf = SentenceBuffer()
    assert buf.feed("Wait.") == []
    assert buf.feed(".") == []
    # Trailing space is what finalizes "no." for emission.
    assert buf.feed(". no. ") == ["Wait...", "no."]


# -- buffering / partial behavior --------------------------------------------


def test_partial_at_chunk_boundary_period_alone() -> None:
    buf = SentenceBuffer()
    assert buf.feed("Hello") == []
    assert buf.feed(".") == []
    assert buf.feed(" World!") == ["Hello.", "World!"]


def test_flush_returns_trailing_partial() -> None:
    buf = SentenceBuffer()
    assert buf.feed("This has no terminator") == []
    assert buf.flush() == "This has no terminator"
    assert buf.flush() is None


def test_flush_is_idempotent_when_empty() -> None:
    buf = SentenceBuffer()
    buf.feed("Done. ")
    assert buf.flush() is None


def test_multiple_terminators_collapsed() -> None:
    out = _drain(SentenceBuffer(), ["Really?! No way. Yes."])
    assert out == ["Really?!", "No way.", "Yes."]


def test_empty_feed() -> None:
    buf = SentenceBuffer()
    assert buf.feed("") == []
    assert buf.flush() is None


def test_whitespace_only_does_not_emit() -> None:
    buf = SentenceBuffer()
    assert buf.feed("   ") == []
    assert buf.flush() is None


# -- _chunk_text: only the visible content channel is spoken ----------------


def test_chunk_text_string_content() -> None:
    chunk = AIMessageChunk(content="hello")
    assert _chunk_text(chunk) == "hello"


def test_chunk_text_ignores_reasoning_in_additional_kwargs() -> None:
    # Qwen3 reasoning-parser puts CoT in additional_kwargs.reasoning_content.
    # The chunk's .content should only ever be the visible reply; we must
    # never speak reasoning_content.
    chunk = AIMessageChunk(
        content="visible reply",
        additional_kwargs={"reasoning_content": "secret CoT, do not speak"},
    )
    out = _chunk_text(chunk)
    assert out == "visible reply"
    assert "secret CoT" not in out


def test_chunk_text_list_content_text_blocks() -> None:
    chunk = AIMessageChunk(
        content=[
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world."},
        ]
    )
    assert _chunk_text(chunk) == "Hello world."


def test_chunk_text_empty() -> None:
    chunk = AIMessageChunk(content="")
    assert _chunk_text(chunk) == ""


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 13])
def test_arbitrary_chunk_sizes_yield_same_sentences(chunk_size: int) -> None:
    text = (
        "Hi Dr. Smith. Pi is 3.14, e.g. roughly. "
        "Wait... really? Yes! Done."
    )
    expected = [
        "Hi Dr. Smith.",
        "Pi is 3.14, e.g. roughly.",
        "Wait...",
        "really?",
        "Yes!",
        "Done.",
    ]
    buf = SentenceBuffer()
    out: list[str] = []
    for i in range(0, len(text), chunk_size):
        out.extend(buf.feed(text[i:i + chunk_size]))
    tail = buf.flush()
    if tail:
        out.append(tail)
    assert out == expected
