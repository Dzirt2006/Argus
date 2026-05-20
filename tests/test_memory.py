from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent.memory import get_store


def test_disabled_mode_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import settings

    monkeypatch.setattr(settings, "memory_enabled", False)

    store = get_store()
    assert store.set_fact("k", "v") is None
    assert store.get_fact("k") is None
    assert store.list_facts() == []
    assert store.forget_fact("k") is False
    assert store.write_summary("text", session_id="s") is None
    assert store.list_recent_summaries() == []


def test_set_and_get_fact(memory_enabled: Path) -> None:
    store = get_store()
    store.set_fact("height", "almost two meters")
    assert store.get_fact("height") == "almost two meters"


def test_set_fact_upsert_overwrites(memory_enabled: Path) -> None:
    store = get_store()
    store.set_fact("color", "red")
    store.set_fact("color", "blue")
    assert store.get_fact("color") == "blue"


def test_list_facts_orders_by_updated_at_desc(memory_enabled: Path) -> None:
    store = get_store()
    store.set_fact("a", "1")
    time.sleep(0.01)
    store.set_fact("b", "2")
    time.sleep(0.01)
    store.set_fact("a", "1-updated")
    keys = [k for k, _, _ in store.list_facts()]
    assert keys == ["a", "b"]


def test_forget_fact_returns_bool(memory_enabled: Path) -> None:
    store = get_store()
    store.set_fact("k", "v")
    assert store.forget_fact("k") is True
    assert store.forget_fact("k") is False
    assert store.get_fact("k") is None


def test_set_fact_source_recorded(memory_enabled: Path) -> None:
    store = get_store()
    store.set_fact("k", "v", source="session_extraction")
    facts = store.list_facts()
    assert facts == [("k", "v", "session_extraction")]


def test_write_summary_and_list_order(memory_enabled: Path) -> None:
    store = get_store()
    store.write_summary("first", session_id="s1")
    time.sleep(0.01)
    store.write_summary("second", session_id="s2")
    rows = store.list_recent_summaries()
    assert [r.text for r in rows] == ["second", "first"]
    assert rows[0].session_id == "s2"
    assert rows[0].created_at > rows[1].created_at


def test_list_recent_summaries_respects_limit(memory_enabled: Path) -> None:
    store = get_store()
    for i in range(5):
        store.write_summary(f"note {i}", session_id="s")
    assert len(store.list_recent_summaries(limit=3)) == 3


def test_write_summary_skips_empty_text(memory_enabled: Path) -> None:
    store = get_store()
    assert store.write_summary("", session_id="s") is None
    assert store.write_summary("   ", session_id="s") is None
    assert store.list_recent_summaries() == []
