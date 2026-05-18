#!/usr/bin/env bash
# SessionStart hook: print a short Argus stack status block.
# Stdout becomes additional system context for the session.

set -u

cd "$(dirname "$0")/../.." || exit 0

echo "=== Argus stack status ==="

if command -v docker >/dev/null 2>&1; then
    vllm_running=$(timeout 3 docker compose ps --services --filter status=running 2>/dev/null | grep -E '^vllm' | paste -sd',' -)
    if [ -n "${vllm_running}" ]; then
        echo "vLLM service running: ${vllm_running}"
    else
        echo "vLLM service running: (none)"
    fi

    mcp_count=$(timeout 3 docker compose ps --services --filter status=running 2>/dev/null \
        | grep -E '^(filesystem|system|search|weather|calendar|media|memory|switches)$' \
        | wc -l | tr -d ' ')
    echo "MCP containers running: ${mcp_count}"
else
    echo "docker: not on PATH"
fi

host_mcp=$(pgrep -af 'mcp_servers/.*/server\.py' 2>/dev/null | wc -l | tr -d ' ')
echo "Host-mode MCP server processes: ${host_mcp}"

db="data/memory.db"
if [ -f "${db}" ]; then
    size=$(stat -c '%s' "${db}" 2>/dev/null || echo '?')
    mtime=$(stat -c '%y' "${db}" 2>/dev/null | cut -d'.' -f1 || echo '?')
    echo "memory.db: ${size} bytes, modified ${mtime}"
else
    echo "memory.db: not present at ${db}"
fi

echo "=========================="
