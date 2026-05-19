"""Shared test fixtures.

All tests are fully isolated:
- `tmp_path` (pytest builtin) gives each test its own scratch directory.
- `monkeypatch` (pytest builtin) reverts attribute patches after each test.
- `_reset_memory_singleton` (autouse below) clears MemoryStore's class-level
  cache and closes any open SQLite handle around every test.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_memory_singleton():
    from agent.memory import MemoryStore

    MemoryStore._instance = None
    yield
    inst = MemoryStore._instance
    if inst is not None:
        conn = getattr(inst, "_sqlite", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    MemoryStore._instance = None


@pytest.fixture
def memory_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Enable memory with an isolated SQLite file under tmp_path."""
    from agent.config import settings

    db_path = tmp_path / "memory.db"
    monkeypatch.setattr(settings, "memory_enabled", True)
    monkeypatch.setattr(settings, "memory_sqlite_path", str(db_path))
    return db_path
