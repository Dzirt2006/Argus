"""Verify the additive JSONL file handler in agent/tracing.py."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import structlog


def test_jsonl_sink_writes_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "logs" / "agent.jsonl"
    monkeypatch.setenv("AGENT_LOG_FILE", str(log_path))

    import agent.tracing as tracing

    importlib.reload(tracing)
    # Reset module state so reload-on-setup picks up the patched env.
    tracing._jsonl_fh = None
    tracing.setup_logging()

    log = structlog.get_logger("test.jsonl")
    log.info("hello_world", a=1, b="two")
    log.info("another_event", x=True)

    # Force flush — the file handle is line-buffered but flush to be safe.
    if tracing._jsonl_fh is not None:
        tracing._jsonl_fh.flush()

    assert log_path.exists(), "JSONL sink did not create the file"
    lines = log_path.read_text().splitlines()
    assert len(lines) >= 2
    parsed = [json.loads(line) for line in lines]
    events = [p.get("event") for p in parsed]
    assert "hello_world" in events
    assert "another_event" in events
    hw = next(p for p in parsed if p.get("event") == "hello_world")
    assert hw.get("a") == 1
    assert hw.get("b") == "two"


def test_jsonl_sink_failure_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If we can't open the file, stdout logging must still work."""
    # Point at a path whose parent we can't create (file-in-place of dir).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    bad_path = blocker / "agent.jsonl"
    monkeypatch.setenv("AGENT_LOG_FILE", str(bad_path))

    import agent.tracing as tracing

    importlib.reload(tracing)
    tracing._jsonl_fh = None
    tracing.setup_logging()

    # If this raises, the test fails.
    log = structlog.get_logger("test.jsonl.fail")
    log.info("should_not_crash", k="v")

    assert tracing._jsonl_fh is None
