"""Guardrails node for the LangGraph agent.

Sits between the 'agent' and 'tools' nodes. Inspects every tool call
the LLM wants to make and enforces four policies:

1. Tool allowlist   — only pre-approved tools can be called.
2. Path validation  — filesystem tools must target /data/.
3. Confirmation     — destructive tools pause for user approval via interrupt().
4. Max steps        — caps total tool invocations per conversation turn.
"""

from __future__ import annotations

import json
import posixpath

import structlog
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

log = structlog.get_logger("agent.guardrails")

# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

# Tools the agent is allowed to call.  Anything not listed here is blocked.
ALLOWED_TOOLS: set[str] = {
    # filesystem
    "read_file",
    "write_file",
    "list_directory",
    # system
    "get_system_uptime",
    "get_disk_usage",
    "get_memory_usage",
    "get_gpu_status",
    # search
    "web_search",
    "web_search_news",
    # weather
    "get_current_weather",
    "get_forecast",
    # calendar
    "get_upcoming_events",
    "search_events",
    "create_event",
    # media
    "play_music",
    "pause_music",
    "skip_track",
    "stop_music",
    "search_music",
    "get_now_playing",
    # switches (HA switch/light wrapper)
    "turn_on",
    "turn_off",
    "toggle",
    "get_state",
    "list_switches",
}

# Subset of ALLOWED_TOOLS that require explicit user confirmation before
# execution (e.g. anything that mutates state).
DESTRUCTIVE_TOOLS: set[str] = {
    "write_file",
    "create_event",
}

# Filesystem tools whose first positional arg is a path — validated against
# the allowed base directory.
PATH_TOOLS: dict[str, str] = {
    "read_file": "path",
    "write_file": "path",
    "list_directory": "path",
}

ALLOWED_BASE = "/data"

# Maximum tool calls the agent may make in a single invoke cycle.
MAX_TOOL_CALLS = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_tool_calls(messages: list) -> int:
    """Count ToolMessages already in the conversation (current turn)."""
    return sum(1 for m in messages if isinstance(m, ToolMessage))


def _block_tool_call(tool_call: dict, reason: str) -> ToolMessage:
    """Create a ToolMessage that tells the LLM a tool call was denied."""
    return ToolMessage(
        content=f"BLOCKED: {reason}",
        tool_call_id=tool_call["id"],
    )


def _path_is_safe(path: str) -> bool:
    """True if `path` joined to ALLOWED_BASE stays under ALLOWED_BASE.

    `posixpath.join` makes an absolute `path` arg replace the base (so
    "/etc/passwd" is caught), and `posixpath.normpath` resolves "..".
    The filesystem MCP server does its own resolve-and-verify; this is
    a defence-in-depth check on LLM-supplied paths.
    """
    full = posixpath.join(ALLOWED_BASE, path)
    resolved = posixpath.normpath(full)
    return resolved == ALLOWED_BASE or resolved.startswith(ALLOWED_BASE + "/")


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------


def check_guardrails(state: dict) -> dict:
    """LangGraph node: inspect pending tool calls and enforce policies.

    Returns a dict with ``messages`` — either the original AI message
    (all calls approved) or a list of ToolMessages for blocked calls
    plus any interrupt-based confirmation flow.
    """
    messages = state["messages"]
    last: AIMessage = messages[-1]

    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return {"messages": []}

    # --- max steps -----------------------------------------------------------
    prior_calls = _count_tool_calls(messages)
    if prior_calls >= MAX_TOOL_CALLS:
        log.warning("max_steps_reached", prior_calls=prior_calls, limit=MAX_TOOL_CALLS)
        return {
            "messages": [
                _block_tool_call(
                    tc,
                    f"Step limit reached ({MAX_TOOL_CALLS}). "
                    "Please finish without further tool calls.",
                )
                for tc in last.tool_calls
            ]
        }

    blocked: list[ToolMessage] = []
    needs_confirmation: list[dict] = []
    approved: list[dict] = []

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc.get("args", {})

        # 1. allowlist
        if name not in ALLOWED_TOOLS:
            log.warning("tool_blocked_allowlist", tool=name)
            blocked.append(_block_tool_call(tc, f"Tool '{name}' is not allowed."))
            continue

        # 2. path validation
        param = PATH_TOOLS.get(name)
        if param and param in args:
            if not _path_is_safe(args[param]):
                log.warning("tool_blocked_path", tool=name, path=args[param])
                blocked.append(
                    _block_tool_call(
                        tc,
                        f"Path '{args[param]}' is outside the allowed "
                        f"base directory ({ALLOWED_BASE}).",
                    )
                )
                continue

        # 3. confirmation gate — collect, don't interrupt yet
        if name in DESTRUCTIVE_TOOLS:
            needs_confirmation.append(tc)
            continue

        log.debug("tool_approved", tool=name)
        approved.append(tc)

    # --- handle confirmations ------------------------------------------------
    for tc in needs_confirmation:
        summary = f"{tc['name']}({json.dumps(tc.get('args', {}), indent=2)})"
        answer = interrupt(
            {
                "action": "confirm_tool_call",
                "tool_call_id": tc["id"],
                "description": summary,
            }
        )
        if str(answer).lower() in ("y", "yes"):
            log.info("tool_confirmed", tool=tc["name"])
            approved.append(tc)
        else:
            log.info("tool_denied", tool=tc["name"])
            blocked.append(_block_tool_call(tc, "User denied this action."))

    # If every call was blocked, return the block messages so the LLM sees
    # the errors and can adjust.
    if not approved:
        return {"messages": blocked}

    # If some were blocked and some approved, we need to rewrite the AI
    # message to only contain the approved tool_calls, then append blocks.
    if blocked:
        trimmed_ai = AIMessage(
            content=last.content,
            tool_calls=approved,
        )
        return {"messages": [trimmed_ai] + blocked}

    # Everything approved — pass through unchanged.
    return {"messages": []}
