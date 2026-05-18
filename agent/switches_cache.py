"""HA switches list cache for system-prompt injection.

Pulls friendly names of switch/light entities from HA every TTL seconds and
formats them as a Markdown block to append to the conversation's system
message. State is NOT included — only stable names — so vLLM's prefix cache
stays warm across turns even when devices toggle between turns.
"""

from __future__ import annotations

import threading
import time

import httpx
import structlog

from agent.config import settings

log = structlog.get_logger("agent.switches_cache")

_DOMAINS = ("light", "switch")
_TTL_S = 30.0

_lock = threading.Lock()
_names: list[str] = []
_at: float = 0.0


def _fetch_names() -> list[str]:
    if not settings.ha_url or not settings.ha_token:
        return []
    url = settings.ha_url.rstrip("/") + "/api/states"
    headers = {"Authorization": f"Bearer {settings.ha_token}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=3.0)
        resp.raise_for_status()
        names: set[str] = set()
        for s in resp.json():
            eid = s.get("entity_id", "")
            domain = eid.split(".", 1)[0] if "." in eid else ""
            if domain not in _DOMAINS:
                continue
            attrs = s.get("attributes") or {}
            name = (attrs.get("friendly_name") or eid.split(".", 1)[1] or eid).strip()
            if name:
                names.add(name)
        return sorted(names, key=str.casefold)
    except Exception as e:
        log.warning("switches_fetch_failed", error=str(e))
        return []


def get_switches_block() -> str:
    """Markdown block with available switch/light friendly names. Empty if HA unconfigured or unreachable."""
    global _names, _at
    now = time.time()
    if now - _at <= _TTL_S and _names:
        block_names = _names
    else:
        with _lock:
            if now - _at > _TTL_S or not _names:
                _names = _fetch_names()
                _at = now
            block_names = _names
    if not block_names:
        return ""
    return "\n\n## Available switches and lights\n" + "\n".join(f"- {n}" for n in block_names)
