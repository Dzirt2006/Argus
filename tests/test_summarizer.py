from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.summarizer import _count_user_turns, _parse_facts, _transcript


# ---- _parse_facts ---------------------------------------------------------


def test_parse_facts_basic() -> None:
    raw = "height: almost two meters\nallergies: peanuts"
    assert _parse_facts(raw) == [
        ("height", "almost two meters"),
        ("allergies", "peanuts"),
    ]


def test_parse_facts_strips_bullet_prefixes() -> None:
    raw = "- height: two meters\n* allergies: nuts\n• color: red"
    assert _parse_facts(raw) == [
        ("height", "two meters"),
        ("allergies", "nuts"),
        ("color", "red"),
    ]


def test_parse_facts_snake_cases_and_lowercases_keys() -> None:
    raw = "Spouse Name: Maria\nPreferred Thermostat: 70F"
    assert _parse_facts(raw) == [
        ("spouse_name", "Maria"),
        ("preferred_thermostat", "70F"),
    ]


def test_parse_facts_skips_lines_without_colon() -> None:
    raw = "key: value\njust some text\nanother: ok"
    assert _parse_facts(raw) == [("key", "value"), ("another", "ok")]


def test_parse_facts_skips_empty_value_or_key() -> None:
    raw = "key:\n: value\ngood: yes"
    assert _parse_facts(raw) == [("good", "yes")]


def test_parse_facts_empty_string() -> None:
    assert _parse_facts("") == []


def test_parse_facts_preserves_value_verbatim() -> None:
    raw = "height: almost two meters"
    assert _parse_facts(raw) == [("height", "almost two meters")]


# ---- _transcript ----------------------------------------------------------


def test_transcript_includes_human_and_ai() -> None:
    msgs = [
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
        HumanMessage(content="what's up"),
        AIMessage(content="not much"),
    ]
    assert _transcript(msgs) == (
        "User: hello\nAssistant: hi there\nUser: what's up\nAssistant: not much"
    )


def test_transcript_skips_empty_or_whitespace_ai_content() -> None:
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content=""),
        AIMessage(content="   "),
    ]
    assert _transcript(msgs) == "User: hi"


def test_transcript_ignores_other_message_types() -> None:
    msgs = [
        SystemMessage(content="you are an assistant"),
        ToolMessage(content="tool result", tool_call_id="t1"),
        HumanMessage(content="real user msg"),
    ]
    assert _transcript(msgs) == "User: real user msg"


# ---- _count_user_turns ----------------------------------------------------


def test_count_user_turns_counts_only_human() -> None:
    msgs = [
        HumanMessage(content="a"),
        AIMessage(content="b"),
        HumanMessage(content="c"),
        ToolMessage(content="t", tool_call_id="t1"),
        SystemMessage(content="s"),
        HumanMessage(content="d"),
    ]
    assert _count_user_turns(msgs) == 3


def test_count_user_turns_empty() -> None:
    assert _count_user_turns([]) == 0
