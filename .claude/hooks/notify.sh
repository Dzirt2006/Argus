#!/usr/bin/env bash
# Notification hook: desktop notify-send when Claude Code needs attention.
# Reads the hook payload from stdin and pulls the .message field.

set -u

if ! command -v notify-send >/dev/null 2>&1; then
    exit 0
fi

payload=$(cat)

if command -v jq >/dev/null 2>&1; then
    message=$(printf '%s' "${payload}" | jq -r '.message // empty')
else
    message=$(printf '%s' "${payload}" | sed -n 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
fi

if [ -z "${message}" ]; then
    message="Claude Code needs your input"
fi

notify-send -u normal -i dialog-question "Claude Code (Argus)" "${message}" >/dev/null 2>&1 || true
exit 0
