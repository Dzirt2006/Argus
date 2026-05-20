"""Smoke tests for the read-only observability webui.

Builds a fake /data tree (memory.db + agent.jsonl) and asserts each page
renders 200 with key strings present.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_fake_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE facts (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                source     TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE summaries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                text       TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        now = time.time()
        conn.execute(
            "INSERT INTO facts(key, value, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("height", "almost two meters", "user", now - 10, now - 10),
        )
        conn.execute(
            "INSERT INTO facts(key, value, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("color", "blue", "user", now - 5, now - 5),
        )
        conn.execute(
            "INSERT INTO summaries(session_id, text, created_at) VALUES (?, ?, ?)",
            ("s-1", "First test session summary.", now - 100),
        )
        conn.execute(
            "INSERT INTO summaries(session_id, text, created_at) VALUES (?, ?, ?)",
            ("s-2", "Second test session summary with longer text " * 5, now - 50),
        )
        conn.commit()
    finally:
        conn.close()


def _make_fake_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "timestamp": "2026-05-20T10:00:00",
            "request_id": "abc12345",
            "event": "tool_exec_start",
            "tools": ["search"],
        },
        {
            "timestamp": "2026-05-20T10:00:01",
            "request_id": "abc12345",
            "event": "tool_exec_done",
            "tool": "search",
            "duration": 0.123,
            "result_preview": "found 3 results for 'argus'",
        },
        {
            "timestamp": "2026-05-20T10:00:02",
            "request_id": "abc12345",
            "event": "tool_blocked_path",
            "tool": "filesystem_read",
            "args": {"path": "/etc/passwd"},
        },
        # non-tool event should be filtered out
        {
            "timestamp": "2026-05-20T10:00:03",
            "request_id": "abc12345",
            "event": "llm_call_done",
        },
    ]
    with path.open("w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


@pytest.fixture
def webui_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_path = tmp_path / "memory.db"
    log_path = tmp_path / "logs" / "agent.jsonl"
    _make_fake_db(db_path)
    _make_fake_log(log_path)

    # Point the app at our temp tree and use an empty MCP server map so
    # /health doesn't try to hit real network endpoints during tests.
    monkeypatch.setenv("WEBUI_MEMORY_DB", str(db_path))
    monkeypatch.setenv("WEBUI_AGENT_LOG", str(log_path))

    # Reimport with patched env so module-level constants pick it up.
    import importlib

    import webui.app as app_module

    importlib.reload(app_module)
    app_module.MCP_SERVERS = {}
    return TestClient(app_module.app)


def test_index_renders(webui_client: TestClient) -> None:
    r = webui_client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Argus" in body
    assert "Facts stored" in body
    assert "2" in body  # two facts inserted


def test_facts_page(webui_client: TestClient) -> None:
    r = webui_client.get("/memory/facts")
    assert r.status_code == 200
    body = r.text
    assert "height" in body
    assert "almost two meters" in body
    assert "color" in body
    assert "blue" in body


def test_summaries_page(webui_client: TestClient) -> None:
    r = webui_client.get("/memory/summaries")
    assert r.status_code == 200
    body = r.text
    assert "s-1" in body
    assert "s-2" in body
    assert "First test session summary" in body


def test_tools_page(webui_client: TestClient) -> None:
    r = webui_client.get("/tools/recent")
    assert r.status_code == 200
    body = r.text
    assert "tool_exec_start" in body
    assert "tool_exec_done" in body
    assert "tool_blocked_path" in body
    # llm_call_done is not a tool event — must be filtered out.
    assert "llm_call_done" not in body


def test_health_page_empty_mcp(webui_client: TestClient) -> None:
    r = webui_client.get("/health")
    assert r.status_code == 200
    body = r.text
    assert "MCP health" in body


def test_health_page_with_unreachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm an unreachable server renders the 'unreachable' badge."""
    db_path = tmp_path / "memory.db"
    log_path = tmp_path / "logs" / "agent.jsonl"
    _make_fake_db(db_path)
    _make_fake_log(log_path)
    monkeypatch.setenv("WEBUI_MEMORY_DB", str(db_path))
    monkeypatch.setenv("WEBUI_AGENT_LOG", str(log_path))

    import importlib

    import webui.app as app_module

    importlib.reload(app_module)
    # Point at a port nobody is listening on.
    app_module.MCP_SERVERS = {"fake": "http://127.0.0.1:1"}
    client = TestClient(app_module.app)

    r = client.get("/health")
    assert r.status_code == 200
    assert "fake" in r.text
    assert "unreachable" in r.text


def test_no_write_endpoints(webui_client: TestClient) -> None:
    """v1 invariant: no POST/PUT/DELETE/PATCH endpoints exist."""
    for method, path in [
        ("post", "/memory/facts"),
        ("delete", "/memory/facts"),
        ("post", "/tools/recent"),
        ("put", "/health"),
    ]:
        r = getattr(webui_client, method)(path)
        # 404 or 405 are both fine — neither indicates a write endpoint.
        assert r.status_code in (404, 405)


def test_missing_memory_db_is_graceful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard must render even with no memory.db / log file."""
    monkeypatch.setenv("WEBUI_MEMORY_DB", str(tmp_path / "missing.db"))
    monkeypatch.setenv("WEBUI_AGENT_LOG", str(tmp_path / "missing.jsonl"))

    import importlib

    import webui.app as app_module

    importlib.reload(app_module)
    app_module.MCP_SERVERS = {}
    client = TestClient(app_module.app)

    for path in ("/", "/memory/facts", "/memory/summaries", "/tools/recent", "/health"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"
