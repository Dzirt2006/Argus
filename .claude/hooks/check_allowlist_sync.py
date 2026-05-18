#!/usr/bin/env python3
"""PostToolUse hook: warn if a new @mcp.tool is missing from ALLOWED_TOOLS.

Reads the Claude Code hook payload from stdin. If the edited file is an MCP
server (mcp_servers/<name>/server.py), parses out every @mcp.tool()-decorated
function name and compares against the ALLOWED_TOOLS set in
agent/guardrails.py. Any missing names are reported to stderr with exit 2 so
the warning is fed back to the model.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUARDRAILS = PROJECT_ROOT / "agent" / "guardrails.py"

TOOL_DECL_RE = re.compile(
    r"@mcp\.tool\([^)]*\)\s*\n\s*(?:async\s+)?def\s+(\w+)",
    re.MULTILINE,
)
ALLOWED_SET_RE = re.compile(
    r"ALLOWED_TOOLS\s*:\s*set\[str\]\s*=\s*\{(.*?)\}",
    re.DOTALL,
)
STRING_LITERAL_RE = re.compile(r"[\"']([^\"']+)[\"']")


def extract_tool_names(source: str) -> list[str]:
    return TOOL_DECL_RE.findall(source)


def extract_allowed_tools(source: str) -> set[str] | None:
    m = ALLOWED_SET_RE.search(source)
    if not m:
        return None
    return set(STRING_LITERAL_RE.findall(m.group(1)))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return 0

    rel = Path(file_path)
    try:
        rel = rel.relative_to(PROJECT_ROOT)
    except ValueError:
        pass

    parts = rel.parts
    if len(parts) < 3 or parts[0] != "mcp_servers" or parts[-1] != "server.py":
        return 0

    try:
        server_source = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return 0

    tools = extract_tool_names(server_source)
    if not tools:
        return 0

    try:
        guardrails_source = GUARDRAILS.read_text(encoding="utf-8")
    except OSError:
        print(f"check_allowlist_sync: cannot read {GUARDRAILS}", file=sys.stderr)
        return 0

    allowed = extract_allowed_tools(guardrails_source)
    if allowed is None:
        print(
            "check_allowlist_sync: could not parse ALLOWED_TOOLS in "
            f"{GUARDRAILS.relative_to(PROJECT_ROOT)}",
            file=sys.stderr,
        )
        return 0

    missing = [t for t in tools if t not in allowed]
    if not missing:
        return 0

    server_rel = "/".join(parts)
    for name in missing:
        print(
            f"WARN: Tool '{name}' is defined in {server_rel} but missing from "
            "agent/guardrails.py:ALLOWED_TOOLS - add it or the LLM call will be "
            "blocked at runtime.",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
