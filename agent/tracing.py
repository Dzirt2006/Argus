"""Structured tracing for the LangGraph agent.

Wraps LLM calls and tool execution with timing, token counts, and
decision logging.  Uses structlog for JSON output in Docker and
pretty console output locally.
"""

from __future__ import annotations

import os
import sys
import time
import uuid

import structlog


def setup_logging() -> None:
    """Configure structlog.  Call once at startup."""
    is_docker = os.path.exists("/.dockerenv")

    if is_docker:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def new_request_id() -> str:
    """Generate a short request ID for correlating log lines in one turn."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Traced call_model — replaces the inline lambda in agent.py
# ---------------------------------------------------------------------------

def make_call_model(llm):
    """Return a traced call_model function bound to the given LLM."""
    log = structlog.get_logger("agent.llm")

    def call_model(state: dict) -> dict:
        messages = state["messages"]
        request_id = structlog.contextvars.get_contextvars().get("request_id", "?")

        log.info("llm_call_start", request_id=request_id, input_messages=len(messages))
        t0 = time.monotonic()
        response = llm.invoke(messages)
        duration = time.monotonic() - t0

        # Token usage — ChatOpenAI populates usage_metadata when available.
        usage = getattr(response, "usage_metadata", None) or {}
        tool_calls = [tc["name"] for tc in getattr(response, "tool_calls", []) or []]

        log.info(
            "llm_call_done",
            request_id=request_id,
            duration=round(duration, 3),
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
            tool_calls=tool_calls or None,
        )

        return {"messages": [response]}

    return call_model


# ---------------------------------------------------------------------------
# Traced ToolNode wrapper
# ---------------------------------------------------------------------------

class TracedToolNode:
    """Wraps LangGraph's ToolNode with per-tool timing and result logging."""

    def __init__(self, tool_node):
        self._inner = tool_node
        self._log = structlog.get_logger("agent.tools")

    def __call__(self, state: dict, config=None) -> dict:
        from langchain_core.messages import AIMessage

        last = state["messages"][-1]
        tool_names = [
            tc["name"]
            for tc in (getattr(last, "tool_calls", []) or [])
        ]

        request_id = structlog.contextvars.get_contextvars().get("request_id", "?")
        self._log.info("tool_exec_start", request_id=request_id, tools=tool_names)
        t0 = time.monotonic()

        if config is not None:
            result = self._inner(state, config)
        else:
            result = self._inner(state)

        duration = time.monotonic() - t0

        # Log each tool result (truncated).
        for msg in result.get("messages", []):
            content = str(msg.content)
            self._log.info(
                "tool_exec_done",
                request_id=request_id,
                tool=getattr(msg, "name", None),
                duration=round(duration, 3),
                result_preview=content[:200] + ("..." if len(content) > 200 else ""),
            )

        return result
