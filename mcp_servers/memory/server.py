"""Memory MCP server — explicit fact/preference writes.

Thin wrapper over the same SQLite file the agent reads from in-process.
Exposes three tools: remember_this, list_facts, forget.

Kept deliberately small: vector memory (session summaries) is handled by
the agent directly to avoid a round-trip on every turn.  This server is
only for the exact-key structured store.
"""

import os
import sqlite3
import threading
import time
from pathlib import Path

from fastmcp import FastMCP

DB_PATH = Path(os.environ.get("MEMORY_SQLITE_PATH", "/data/memory.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

_conn_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(
                str(DB_PATH), check_same_thread=False, isolation_level=None
            )
            c.execute("PRAGMA journal_mode=WAL;")
            c.executescript(_SCHEMA)
            _conn = c
        return _conn


mcp = FastMCP("memory")


@mcp.tool()
async def remember_this(key: str, value: str) -> str:
    """Store a fact or preference for future conversations.

    Use this when the user tells you something durable about themselves,
    their home, their preferences, or recurring routines — something that
    should still be true next week.  Do NOT use it for transient context
    inside the current turn.

    Args:
        key: Short snake_case identifier, e.g. 'preferred_bedroom_temp'
             or 'partner_name'.  Overwrites any existing value at this key.
        value: The fact itself, as a short natural-language phrase,
               e.g. '68F at night, 72F during the day'.
    """
    try:
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return "Error: both key and value are required."
        now = time.time()
        _db().execute(
            """
            INSERT INTO facts(key, value, source, created_at, updated_at)
            VALUES (?, ?, 'agent', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (key, value, now, now),
        )
        return f"Remembered: {key} = {value}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def list_facts() -> str:
    """List everything currently remembered about the user. Returns key: value pairs."""
    try:
        rows = _db().execute(
            "SELECT key, value FROM facts ORDER BY updated_at DESC"
        ).fetchall()
        if not rows:
            return "No facts stored yet."
        return "\n".join(f"{k}: {v}" for k, v in rows)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def forget(key: str) -> str:
    """Delete a stored fact by key.

    Args:
        key: The snake_case identifier used when the fact was stored.
             Use list_facts first if unsure of the exact key.
    """
    try:
        cur = _db().execute("DELETE FROM facts WHERE key = ?", (key.strip(),))
        if cur.rowcount == 0:
            return f"No fact found with key '{key}'."
        return f"Forgot: {key}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8007)
