from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent import thinking as thinking_mod
from agent.thinking import _LENGTH_THRESHOLD, should_think


@pytest.fixture(autouse=True)
def _clear_thinking_cache():
    thinking_mod._clear_cache_for_tests()
    yield
    thinking_mod._clear_cache_for_tests()


@pytest.mark.parametrize("text", [
    "plan a trip to Paris",
    "compare these two options",
    "analyze the failure",
    "give me the analysis",
    "explain how this works",
    "figure out why it broke",
    "walk me through it step by step",
    "what are the pros and cons",
    "the tradeoffs are unclear",
    "weigh the options for me",
    "debug this for me",
    "troubleshoot the error",
    "diagnose the slowdown",
    "optimize this query",
    "refactor this module",
    "architect a new system",
    "design a schema",
    "what's the strategy here",
    "based on calendar and weather, what should I wear",
    "summarize today and tomorrow",
    "if it rains then cancel",
    "should I take the train",
    "what would happen if I removed this",
    "how would you fix this",
    "why does this fail",
])
def test_keyword_triggers_thinking(text: str) -> None:
    assert should_think(text) is True


@pytest.mark.parametrize("text", [
    "turn on the kitchen light",
    "hello",
    "what time is it",
    "set the thermostat to 70",
    "play some music",
])
def test_short_unrelated_does_not_trigger(text: str) -> None:
    assert should_think(text) is False


def test_long_message_triggers_regardless_of_keywords() -> None:
    msg = "blah " * (_LENGTH_THRESHOLD // 4)
    assert len(msg) >= _LENGTH_THRESHOLD
    assert should_think(msg) is True


def test_case_insensitive() -> None:
    assert should_think("PLAN a trip") is True
    assert should_think("Compare these") is True
    assert should_think("REFACTOR this") is True


# ---------------------------------------------------------------------------
# Three-tier classifier tests
# ---------------------------------------------------------------------------


def _make_mock_llm(answer: str) -> MagicMock:
    """Build a mock that mimics ChatOpenAI.bind(...).invoke(...) -> AIMessage."""
    llm = MagicMock()
    response = MagicMock()
    response.content = answer
    bound = MagicMock()
    bound.invoke.return_value = response
    llm.bind.return_value = bound
    return llm


def test_regex_hard_match_skips_llm() -> None:
    llm = _make_mock_llm("N")
    assert should_think("plan a trip to Paris", llm_fast=llm) is True
    llm.bind.assert_not_called()


def test_hard_skip_short_message_skips_llm() -> None:
    llm = _make_mock_llm("Y")
    # 3 words — below the hard-skip threshold.
    assert should_think("play some music", llm_fast=llm) is False
    llm.bind.assert_not_called()


def test_llm_fallback_returns_true_on_yes() -> None:
    llm = _make_mock_llm("Y")
    # Ambiguous: no keyword, no length trigger, more than 3 words.
    assert should_think("can you put together a list of options for dinner", llm_fast=llm) is True
    llm.bind.assert_called_once()


def test_llm_fallback_returns_false_on_no() -> None:
    llm = _make_mock_llm("N")
    assert should_think("can you put together a list of options for dinner", llm_fast=llm) is False
    llm.bind.assert_called_once()


def test_llm_fallback_ambiguous_defaults_to_false() -> None:
    llm = _make_mock_llm("maybe? hard to say")
    assert should_think("can you put together a list of options for dinner", llm_fast=llm) is False


def test_llm_fallback_cached_for_repeat_calls() -> None:
    llm = _make_mock_llm("Y")
    text = "can you put together a list of options for dinner"
    assert should_think(text, llm_fast=llm) is True
    assert should_think(text, llm_fast=llm) is True
    # Bound classifier invoked exactly once across both calls.
    bound = llm.bind.return_value
    assert bound.invoke.call_count == 1


def test_flag_disabled_skips_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import settings

    monkeypatch.setattr(settings, "thinking_llm_fallback_enabled", False)
    llm = _make_mock_llm("Y")
    assert should_think("can you put together a list of options for dinner", llm_fast=llm) is False
    llm.bind.assert_not_called()


def test_no_llm_passed_falls_back_to_regex_only() -> None:
    # No llm_fast — function must not raise and must default to False
    # for ambiguous messages (preserves old behavior).
    assert should_think("can you put together a list of options for dinner") is False


def test_llm_failure_defaults_to_false() -> None:
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.side_effect = RuntimeError("vllm exploded")
    llm.bind.return_value = bound
    assert should_think("can you put together a list of options for dinner", llm_fast=llm) is False


def test_cache_key_normalizes_case_and_whitespace() -> None:
    llm = _make_mock_llm("Y")
    # Ambiguous message (no keyword, long enough to pass hard-skip). Surface
    # variants — case, trailing space, doubled internal whitespace — must
    # share a cache entry so the LLM is consulted exactly once.
    base = "can you put together a list of options for dinner"
    assert should_think(base.upper(), llm_fast=llm) is True
    assert should_think(base + "  ", llm_fast=llm) is True
    assert should_think(base.replace(" ", "  "), llm_fast=llm) is True
    bound = llm.bind.return_value
    assert bound.invoke.call_count == 1
