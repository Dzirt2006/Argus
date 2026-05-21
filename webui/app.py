"""Read-only observability dashboard for Argus.

Server-rendered HTML with jinja2.  No write endpoints — every route is GET
and the SQLite/log files are mounted read-only into the container.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ---------------------------------------------------------------------------
# Config (env-driven; no dependency on agent.config so the container stays
# self-contained per the v1 scope)
# ---------------------------------------------------------------------------

MEMORY_DB_PATH = os.environ.get("WEBUI_MEMORY_DB", "/data/memory.db")
AGENT_LOG_PATH = os.environ.get("WEBUI_AGENT_LOG", "/data/logs/agent.jsonl")

# Default mirrors agent/config.py:Settings.mcp_servers.  Overridable via
# WEBUI_MCP_SERVERS=name1=url1,name2=url2 for ad-hoc runs.
_DEFAULT_MCP_SERVERS: dict[str, str] = {
    "memory": "http://memory:8007",
    "switches": "http://switches:8008",
    "filesystem": "http://filesystem:8001",
    "system": "http://system:8002",
    "search": "http://search:8003",
    "weather": "http://weather:8004",
    "calendar": "http://calendar:8005",
    "media": "http://media:8006",
}


def _load_mcp_servers() -> dict[str, str]:
    raw = os.environ.get("WEBUI_MCP_SERVERS", "").strip()
    if not raw:
        return dict(_DEFAULT_MCP_SERVERS)
    out: dict[str, str] = {}
    for part in raw.split(","):
        if "=" in part:
            name, url = part.split("=", 1)
            out[name.strip()] = url.strip()
    return out or dict(_DEFAULT_MCP_SERVERS)


MCP_SERVERS = _load_mcp_servers()

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _truncate(text: str, n: int = 120) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= n else s[: n - 1] + "…"


templates.env.filters["truncate_chars"] = _truncate


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    key: str
    value: str
    source: str
    updated_at: float


@dataclass
class SummaryRow:
    text: str
    session_id: str
    created_at: float


@dataclass
class ToolEvent:
    timestamp: str
    request_id: str
    event: str
    tool: str
    args: str
    result: str


_TOOL_EVENT_NAMES = {
    "tool_exec_start",
    "tool_exec_done",
    "tool_approved",
    "tool_confirmed",
    "tool_denied",
}


def _is_tool_event(name: str) -> bool:
    return name in _TOOL_EVENT_NAMES or name.startswith("tool_blocked_")


def _open_db(path: str) -> sqlite3.Connection | None:
    p = Path(path)
    if not p.exists():
        return None
    # uri=true + mode=ro keeps us honest about read-only even if the bind
    # mount weren't already ro.
    try:
        return sqlite3.connect(
            f"file:{p}?mode=ro", uri=True, check_same_thread=False
        )
    except sqlite3.OperationalError:
        return None


def read_facts(db_path: str) -> list[Fact]:
    conn = _open_db(db_path)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT key, value, source, updated_at FROM facts "
            "ORDER BY updated_at DESC"
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()
    return [Fact(k, v, s, float(u)) for k, v, s, u in rows]


def read_summaries(db_path: str, limit: int = 50) -> list[SummaryRow]:
    conn = _open_db(db_path)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT text, session_id, created_at FROM summaries "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()
    return [SummaryRow(t, s or "", float(c)) for t, s, c in rows]


def latest_summary_ts(db_path: str) -> float | None:
    conn = _open_db(db_path)
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT MAX(created_at) FROM summaries"
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
    return float(row[0]) if row and row[0] is not None else None


def count_facts(db_path: str) -> int:
    conn = _open_db(db_path)
    if conn is None:
        return 0
    try:
        row = conn.execute("SELECT COUNT(*) FROM facts").fetchone()
    except sqlite3.DatabaseError:
        return 0
    finally:
        conn.close()
    return int(row[0]) if row else 0


def tail_tool_events(log_path: str, limit: int = 100) -> list[ToolEvent]:
    """Return the last ``limit`` tool-related events from agent.jsonl.

    Tails the file in one pass.  For the volumes we care about (one file
    per session, rotated by an external process) full-file read is fine.
    """
    p = Path(log_path)
    if not p.exists():
        return []
    out: list[ToolEvent] = []
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = str(obj.get("event", ""))
                if not _is_tool_event(event):
                    continue
                tool = obj.get("tool")
                if tool is None and isinstance(obj.get("tools"), list):
                    tool = ",".join(str(t) for t in obj["tools"])
                args = obj.get("args") or obj.get("tool_args") or ""
                result = obj.get("result_preview") or obj.get("result") or ""
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, default=str)
                if isinstance(result, (dict, list)):
                    result = json.dumps(result, default=str)
                out.append(
                    ToolEvent(
                        timestamp=str(obj.get("timestamp", "")),
                        request_id=str(obj.get("request_id", "")),
                        event=event,
                        tool=str(tool or ""),
                        args=str(args),
                        result=str(result),
                    )
                )
    except OSError:
        return []
    # Keep insertion order (file is chronological); take last N then reverse
    # so newest is on top.
    tail = out[-limit:]
    tail.reverse()
    return tail


def probe_mcp_health(servers: dict[str, str], timeout: float = 2.0) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, url in servers.items():
        t0 = time.monotonic()
        reachable = False
        status: int | None = None
        error: str | None = None
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True)
            status = resp.status_code
            # Any HTTP response (even 404) means the process is up.
            reachable = True
        except httpx.HTTPError as e:
            error = type(e).__name__
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        results.append({
            "name": name,
            "url": url,
            "reachable": reachable,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "error": error,
        })
    return results


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="Argus Observability", docs_url=None, redoc_url=None)

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        facts_n = count_facts(MEMORY_DB_PATH)
        latest_ts = latest_summary_ts(MEMORY_DB_PATH)
        health = probe_mcp_health(MCP_SERVERS)
        reachable = sum(1 for h in health if h["reachable"])
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "facts_count": facts_n,
                "latest_summary_ts": latest_ts,
                "mcp_reachable": reachable,
                "mcp_total": len(health),
            },
        )

    @app.get("/memory/facts", response_class=HTMLResponse)
    def memory_facts(request: Request) -> HTMLResponse:
        rows = read_facts(MEMORY_DB_PATH)
        return templates.TemplateResponse(
            request, "facts.html", {"rows": rows},
        )

    @app.get("/memory/summaries", response_class=HTMLResponse)
    def memory_summaries(request: Request) -> HTMLResponse:
        rows = read_summaries(MEMORY_DB_PATH, limit=50)
        return templates.TemplateResponse(
            request, "summaries.html", {"rows": rows},
        )

    @app.get("/tools/recent", response_class=HTMLResponse)
    def tools_recent(request: Request) -> HTMLResponse:
        rows = tail_tool_events(AGENT_LOG_PATH, limit=100)
        return templates.TemplateResponse(
            request, "tools.html", {"rows": rows, "log_path": AGENT_LOG_PATH},
        )

    @app.get("/health", response_class=HTMLResponse)
    def health(request: Request) -> HTMLResponse:
        results = probe_mcp_health(MCP_SERVERS)
        return templates.TemplateResponse(
            request, "health.html", {"rows": results},
        )

    return app


app = create_app()
