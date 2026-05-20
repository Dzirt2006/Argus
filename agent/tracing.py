"""Structured tracing for the LangGraph agent.

Wraps LLM calls and tool execution with timing, token counts, and
decision logging.  Uses structlog for JSON output in Docker and
pretty console output locally.

A side-channel processor also mirrors every event as JSONL to
``/data/logs/agent.jsonl`` so the webui can tail it.  This is purely
additive: stdout formatting is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from typing import Any

import structlog


AGENT_JSONL_PATH = os.environ.get("AGENT_LOG_FILE", "/data/logs/agent.jsonl")

_file_lock = threading.Lock()
_jsonl_fh: Any = None


def _open_jsonl_sink(path: str) -> Any:
    """Open the JSONL sink, creating parent dirs. Returns None on failure."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, "a", buffering=1, encoding="utf-8")
    except OSError:
        # Read-only FS or perms — keep stdout logging working regardless.
        return None


def _jsonl_sink_processor(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Structlog processor that mirrors the event to a JSONL file.

    Runs before the final renderer so we render our own JSON copy and
    leave the event_dict untouched for the configured stdout renderer.
    """
    if _jsonl_fh is None:
        return event_dict
    try:
        line = json.dumps(event_dict, default=str, ensure_ascii=False)
        with _file_lock:
            _jsonl_fh.write(line + "\n")
    except Exception:
        # Never let logging crash the agent.
        pass
    return event_dict


def setup_logging() -> None:
    """Configure structlog.  Call once at startup."""
    global _jsonl_fh
    is_docker = os.path.exists("/.dockerenv")

    if is_docker:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    if _jsonl_fh is None:
        _jsonl_fh = _open_jsonl_sink(AGENT_JSONL_PATH)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            _jsonl_sink_processor,
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


def _with_memory_block(messages: list, memory_block: str) -> list:
    """Return a new message list with memory appended to the first SystemMessage.

    Does not mutate `state['messages']` — LangGraph's checkpointer relies on
    message identity for interrupt/resume.
    """
    from langchain_core.messages import SystemMessage

    out = list(messages)
    for i, m in enumerate(out):
        if isinstance(m, SystemMessage):
            out[i] = SystemMessage(content=m.content + memory_block)
            return out
    out.insert(0, SystemMessage(content=memory_block.lstrip()))
    return out



def make_call_model(llm_fast, llm_think=None):
    """Return a traced call_model function that routes between fast/thinking LLMs.

    If llm_think is None, all requests use llm_fast (thinking disabled).
    """
    from agent.config import settings
    from agent.memory import get_store
    from agent.switches_cache import get_switches_block
    from agent.thinking import should_think

    log = structlog.get_logger("agent.llm")

    def call_model(state: dict) -> dict:
        messages = state["messages"]
        request_id = structlog.contextvars.get_contextvars().get("request_id", "?")

        # Find the last human message to classify.
        user_text = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                user_text = msg.content
                break

        thinking = llm_think is not None and should_think(user_text)
        llm = llm_think if thinking else llm_fast

        # Context injection: memory (facts + recent summaries) + switches list.
        context_block = ""
        summaries: list = []
        if settings.memory_enabled:
            store = get_store()
            facts = store.list_facts()
            summaries = store.list_recent_summaries()
            if facts or summaries:
                parts = []
                if facts:
                    parts.append("## Known facts")
                    parts.extend(f"- {k}: {v}" for k, v, _src in facts)
                if summaries:
                    parts.append("## Recent context")
                    parts.extend(f"- {s.text}" for s in summaries)
                context_block = "\n\n" + "\n".join(parts)
        context_block += get_switches_block()
        if context_block:
            messages = _with_memory_block(messages, context_block)

        log.info("llm_call_start", request_id=request_id,
                 input_messages=len(messages), thinking=thinking,
                 memory_summaries=len(summaries),
                 context_block_chars=len(context_block) or None)
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

    async def __call__(self, state: dict, config=None) -> dict:
        last = state["messages"][-1]
        tool_names = [
            tc["name"]
            for tc in (getattr(last, "tool_calls", []) or [])
        ]

        request_id = structlog.contextvars.get_contextvars().get("request_id", "?")
        self._log.info("tool_exec_start", request_id=request_id, tools=tool_names)
        t0 = time.monotonic()

        if config is not None:
            result = await self._inner.ainvoke(state, config)
        else:
            result = await self._inner.ainvoke(state)

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
