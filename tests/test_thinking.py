from __future__ import annotations

import pytest

from agent.thinking import _LENGTH_THRESHOLD, should_think


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
