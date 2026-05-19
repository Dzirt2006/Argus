from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.guardrails import (
    MAX_TOOL_CALLS,
    _path_is_safe,
    check_guardrails,
)


def _ai_with_calls(calls: list[dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=calls)


def _call(name: str, args: dict | None = None, id_: str = "call_1") -> dict:
    return {"name": name, "args": args or {}, "id": id_, "type": "tool_call"}


# ---- _path_is_safe --------------------------------------------------------


def test_path_safe_relative_inside_base() -> None:
    assert _path_is_safe("notes.txt") is True
    assert _path_is_safe("sub/dir/file.md") is True


def test_path_safe_empty_resolves_to_base() -> None:
    assert _path_is_safe("") is True


def test_path_safe_blocks_parent_traversal() -> None:
    assert _path_is_safe("../etc/passwd") is False
    assert _path_is_safe("sub/../../etc/passwd") is False


def test_path_safe_blocks_absolute_outside_base() -> None:
    assert _path_is_safe("/etc/passwd") is False


def test_path_safe_allows_normalising_traversal_that_stays_inside() -> None:
    assert _path_is_safe("sub/dir/../file.md") is True


def test_path_safe_rejects_lookalike_prefix() -> None:
    # "/data2" must not be considered "under /data".
    assert _path_is_safe("/data2/file") is False


# ---- check_guardrails: no tool calls --------------------------------------


def test_no_tool_calls_attribute_returns_passthrough() -> None:
    state = {"messages": [HumanMessage(content="hi")]}
    assert check_guardrails(state) == {"messages": []}


def test_empty_tool_calls_returns_passthrough() -> None:
    state = {"messages": [_ai_with_calls([])]}
    assert check_guardrails(state) == {"messages": []}


# ---- check_guardrails: allowlist ------------------------------------------


def test_disallowed_tool_is_blocked() -> None:
    state = {"messages": [_ai_with_calls([_call("rm_rf")])]}
    out = check_guardrails(state)
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert "not allowed" in msg.content


def test_allowed_tool_passes_through() -> None:
    state = {"messages": [_ai_with_calls([_call("get_system_uptime")])]}
    assert check_guardrails(state) == {"messages": []}


# ---- check_guardrails: path validation ------------------------------------


def test_path_outside_data_is_blocked() -> None:
    state = {"messages": [_ai_with_calls([_call("read_file", {"path": "../etc/passwd"})])]}
    out = check_guardrails(state)
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert "outside the allowed" in msg.content


def test_path_inside_data_passes_through() -> None:
    state = {"messages": [_ai_with_calls([_call("read_file", {"path": "notes.txt"})])]}
    assert check_guardrails(state) == {"messages": []}


# ---- check_guardrails: max steps ------------------------------------------


def test_max_steps_blocks_all_calls() -> None:
    prior = [ToolMessage(content="ok", tool_call_id=f"t{i}") for i in range(MAX_TOOL_CALLS)]
    ai = _ai_with_calls([
        _call("list_directory", {"path": "a"}, id_="c1"),
        _call("get_system_uptime", id_="c2"),
    ])
    state = {"messages": prior + [ai]}
    out = check_guardrails(state)
    assert len(out["messages"]) == 2
    assert all(isinstance(m, ToolMessage) for m in out["messages"])
    assert all("Step limit" in m.content for m in out["messages"])


# ---- check_guardrails: mixed approve/block --------------------------------


def test_mix_of_allowed_and_blocked_trims_ai_message() -> None:
    state = {"messages": [_ai_with_calls([
        _call("list_directory", {"path": "stuff"}, id_="a"),
        _call("rm_rf", id_="b"),
    ])]}
    out = check_guardrails(state)
    msgs = out["messages"]
    assert len(msgs) == 2
    rewritten = msgs[0]
    assert isinstance(rewritten, AIMessage)
    assert len(rewritten.tool_calls) == 1
    assert rewritten.tool_calls[0]["name"] == "list_directory"
    assert isinstance(msgs[1], ToolMessage)
    assert "not allowed" in msgs[1].content
