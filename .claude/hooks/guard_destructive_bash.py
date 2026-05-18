#!/usr/bin/env python3
"""PreToolUse hook: block destructive Bash ops on /data and the memory DB.

Reads the Claude Code hook payload from stdin. If the bash command matches a
known-destructive pattern (volume wipe, rm -rf data/, sqlite DROP/DELETE,
truncating redirect onto memory.db, etc.), exits 2 with an explanation. Set
CLAUDE_ALLOW_DESTRUCTIVE=1 in the environment to bypass the guard for a
session.
"""

from __future__ import annotations

import json
import os
import re
import sys

# Match a command at a "command boundary" — start of string, or after a shell
# separator like ;, &&, ||, |, or a subshell paren. This keeps the regexes from
# false-positiving on the same words quoted inside another command's arguments
# (e.g. `git commit -m "...rm -rf data/..."`).
CMD_START = r"(?:^|[;&|()`]|\$\()\s*"

PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(CMD_START + r"docker\s+compose\s+down\b[^;&|]*\s(?:-v\b|--volumes\b)"),
        "`docker compose down -v` wipes the data/ volume "
        "(/data/memory.db lives there). Drop -v or set CLAUDE_ALLOW_DESTRUCTIVE=1.",
    ),
    (
        re.compile(CMD_START + r"docker\s+volume\s+rm\b[^;&|]*argus", re.IGNORECASE),
        "`docker volume rm` on an argus volume wipes persistent state. "
        "Set CLAUDE_ALLOW_DESTRUCTIVE=1 if intentional.",
    ),
    (
        re.compile(
            CMD_START + r"rm\s+-[a-zA-Z]*r[a-zA-Z]*\b[^;&|]*?\s(?:/data\b|data/|\./data\b)"
        ),
        "`rm -r` targeting data/ wipes /data/memory.db. "
        "Set CLAUDE_ALLOW_DESTRUCTIVE=1 if intentional.",
    ),
    (
        re.compile(
            CMD_START + r"sqlite3\b[^;&|]*memory\.db[^;&|]*(?:DROP\s+TABLE|DELETE\s+FROM|TRUNCATE)",
            re.IGNORECASE,
        ),
        "sqlite3 DROP/DELETE/TRUNCATE on memory.db destroys facts + summaries. "
        "Set CLAUDE_ALLOW_DESTRUCTIVE=1 if intentional.",
    ),
    (
        re.compile(r"(?:^|[\s;&|])>\s*\S*memory\.db\b"),
        "Truncating redirect onto memory.db overwrites the file. "
        "Use sqlite3 commands instead, or set CLAUDE_ALLOW_DESTRUCTIVE=1.",
    ),
    (
        re.compile(
            CMD_START + r"git\s+clean\b[^;&|]*-[a-zA-Z]*[fx][a-zA-Z]*\b[^;&|]*(?:data/|\./data)"
        ),
        "`git clean -fx` against data/ removes untracked DB files. "
        "Set CLAUDE_ALLOW_DESTRUCTIVE=1 if intentional.",
    ),
]


def main() -> int:
    if os.environ.get("CLAUDE_ALLOW_DESTRUCTIVE") == "1":
        return 0

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    if payload.get("tool_name") != "Bash":
        return 0

    command = (payload.get("tool_input") or {}).get("command", "")
    if not command:
        return 0

    for regex, message in PATTERNS:
        if regex.search(command):
            print(f"BLOCKED: {message}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
