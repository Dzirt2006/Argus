"""Long-term memory for the agent.

Single SQLite store with two tables:
- `facts`: structured key/value preferences with exact-key lookup.
- `summaries`: end-of-session natural-language notes, append-only.

No vector retrieval. The corpus is small enough (one note per session)
to inject the most recent N entries into the system prompt directly,
which gives full recall without an embedder, reranker, or vector DB.

Everything is a no-op when settings.memory_enabled is false, so the
rest of the codebase can call into MemoryStore unconditionally.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from agent.config import settings

log = structlog.get_logger("agent.memory")

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    text       TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summaries_created ON summaries(created_at DESC);
"""


@dataclass
class Summary:
    text: str
    session_id: str
    created_at: float


class MemoryStore:
    """Singleton wrapper around the SQLite memory file.

    Thread-safe for the expected access pattern (one writer, a few readers).
    SQLite WAL handles concurrent reads from the agent and the memory MCP.
    """

    _instance: MemoryStore | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._sqlite: sqlite3.Connection | None = None
        self._ready = False

    @classmethod
    def instance(cls) -> MemoryStore:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        if not settings.memory_enabled:
            return False
        try:
            path = Path(settings.memory_sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(_SQLITE_SCHEMA)
            self._sqlite = conn
            self._ready = True
            log.info("memory_ready", path=str(path))
        except Exception as e:
            log.error("memory_init_failed", error=str(e))
            self._ready = False
        return self._ready

    # -- summaries ---------------------------------------------------------
    def write_summary(self, text: str, session_id: str | None = None) -> int | None:
        """Persist a session summary. Returns the row id."""
        if not self._ensure_ready() or not text.strip():
            return None
        cur = self._sqlite.execute(
            "INSERT INTO summaries(session_id, text, created_at) VALUES (?, ?, ?)",
            (session_id, text, time.time()),
        )
        log.info("memory_summary_stored", session_id=session_id, chars=len(text))
        return cur.lastrowid

    def list_recent_summaries(self, limit: int | None = None) -> list[Summary]:
        """Return the most recent summaries, newest-first."""
        if not self._ensure_ready():
            return []
        n = limit if limit is not None else settings.memory_summaries_inject_n
        rows = self._sqlite.execute(
            "SELECT text, session_id, created_at FROM summaries "
            "ORDER BY created_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [Summary(text=t, session_id=s or "", created_at=float(c)) for t, s, c in rows]

    # -- structured facts --------------------------------------------------
    def set_fact(self, key: str, value: str, source: str = "user") -> None:
        if not self._ensure_ready():
            return
        now = time.time()
        self._sqlite.execute(
            """
            INSERT INTO facts(key, value, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (key, value, source, now, now),
        )

    def get_fact(self, key: str) -> str | None:
        if not self._ensure_ready():
            return None
        row = self._sqlite.execute(
            "SELECT value FROM facts WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def list_facts(self) -> list[tuple[str, str, str]]:
        """Return [(key, value, source), ...]."""
        if not self._ensure_ready():
            return []
        return self._sqlite.execute(
            "SELECT key, value, source FROM facts ORDER BY updated_at DESC"
        ).fetchall()

    def forget_fact(self, key: str) -> bool:
        if not self._ensure_ready():
            return False
        cur = self._sqlite.execute("DELETE FROM facts WHERE key = ?", (key,))
        return cur.rowcount > 0


def get_store() -> MemoryStore:
    return MemoryStore.instance()
