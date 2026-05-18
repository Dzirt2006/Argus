"""Switches MCP — narrow Home Assistant wrapper for switch/light entities only.

Talks directly to HA REST API. Caches the friendly_name -> (entity_id, domain)
registry for ~30s so most tool calls cost a single REST round-trip. States are
never cached — they change frequently and stale answers would be worse than
slow ones.

Why custom instead of HA's MCP: HA exposes every domain (sensors, scenes,
automations, climate, scripts...) which blows the prompt budget and isn't what
we want. Argus uses HA only for switch/light manipulation.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
from fastmcp import FastMCP

HA_URL = os.environ.get("HA_URL", "").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

DOMAINS = ("light", "switch")
REGISTRY_TTL_S = 30.0

mcp = FastMCP("switches")

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()

_registry: dict[str, tuple[str, str]] = {}  # friendly_name_lower -> (entity_id, domain)
_registry_at: float = 0.0
_registry_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(
                    base_url=HA_URL,
                    headers={
                        "Authorization": f"Bearer {HA_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    timeout=5.0,
                    limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60),
                )
    return _client


async def _refresh_registry() -> None:
    global _registry, _registry_at
    client = await _get_client()
    resp = await client.get("/api/states")
    resp.raise_for_status()
    new: dict[str, tuple[str, str]] = {}
    for s in resp.json():
        eid = s.get("entity_id", "")
        domain = eid.split(".", 1)[0] if "." in eid else ""
        if domain not in DOMAINS:
            continue
        attrs = s.get("attributes") or {}
        name = (attrs.get("friendly_name") or eid.split(".", 1)[1] or eid).strip()
        new[name.lower()] = (eid, domain)
    _registry = new
    _registry_at = time.time()


async def _ensure_registry() -> dict[str, tuple[str, str]]:
    if _registry and (time.time() - _registry_at) <= REGISTRY_TTL_S:
        return _registry
    async with _registry_lock:
        if not _registry or (time.time() - _registry_at) > REGISTRY_TTL_S:
            await _refresh_registry()
    return _registry


def _resolve(name: str, registry: dict[str, tuple[str, str]]) -> tuple[tuple[str, str] | None, list[str]]:
    """Return ((entity_id, domain), []) on hit, or (None, candidates) on miss/ambiguous."""
    key = name.strip().lower()
    if not key:
        return None, []
    if key in registry:
        return registry[key], []
    matches = [(fname, ed) for fname, ed in registry.items() if key in fname]
    if len(matches) == 1:
        return matches[0][1], []
    if len(matches) > 1:
        return None, sorted(m[0] for m in matches)
    return None, []


async def _call_service(domain: str, service: str, entity_id: str) -> None:
    client = await _get_client()
    resp = await client.post(
        f"/api/services/{domain}/{service}",
        json={"entity_id": entity_id},
    )
    resp.raise_for_status()


def _format_miss(name: str, candidates: list[str]) -> str:
    if candidates:
        return f"Ambiguous '{name}'. Candidates: {', '.join(candidates)}."
    return f"No switch or light named '{name}'. Use list_switches to see available names."


@mcp.tool()
async def turn_on(name: str) -> str:
    """Turn on a switch or light by its friendly name (e.g. 'fan', 'kitchen light')."""
    try:
        registry = await _ensure_registry()
        resolved, candidates = _resolve(name, registry)
        if resolved is None:
            return _format_miss(name, candidates)
        entity_id, domain = resolved
        await _call_service(domain, "turn_on", entity_id)
        return f"Turned on {name}."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def turn_off(name: str) -> str:
    """Turn off a switch or light by its friendly name."""
    try:
        registry = await _ensure_registry()
        resolved, candidates = _resolve(name, registry)
        if resolved is None:
            return _format_miss(name, candidates)
        entity_id, domain = resolved
        await _call_service(domain, "turn_off", entity_id)
        return f"Turned off {name}."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def toggle(name: str) -> str:
    """Toggle a switch or light by its friendly name."""
    try:
        registry = await _ensure_registry()
        resolved, candidates = _resolve(name, registry)
        if resolved is None:
            return _format_miss(name, candidates)
        entity_id, domain = resolved
        await _call_service(domain, "toggle", entity_id)
        return f"Toggled {name}."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_state(name: str) -> str:
    """Return the current state of a switch or light by its friendly name."""
    try:
        registry = await _ensure_registry()
        resolved, candidates = _resolve(name, registry)
        if resolved is None:
            return _format_miss(name, candidates)
        entity_id, _domain = resolved
        client = await _get_client()
        resp = await client.get(f"/api/states/{entity_id}")
        resp.raise_for_status()
        state = resp.json().get("state", "unknown")
        return f"{name} is {state}."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def list_switches() -> str:
    """List all available switch and light friendly names. Use when uncertain about a name."""
    try:
        registry = await _ensure_registry()
        if not registry:
            return "No switches or lights found."
        names = sorted({fname for fname, _ in registry.items()}, key=str.casefold)
        return "\n".join(f"- {n}" for n in names)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


if __name__ == "__main__":
    if not HA_URL or not HA_TOKEN:
        raise SystemExit("HA_URL and HA_TOKEN must be set in the environment.")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8008)
