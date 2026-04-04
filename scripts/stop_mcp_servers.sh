#!/usr/bin/env bash
# Stop all MCP servers started by start_mcp_servers.sh

if [ -f /tmp/argus-mcp-pids ]; then
    while read -r pid; do
        kill "$pid" 2>/dev/null && echo "Stopped pid $pid"
    done < /tmp/argus-mcp-pids
    rm /tmp/argus-mcp-pids
    echo "All MCP servers stopped."
else
    echo "No PID file found. Killing by port..."
    for port in 8001 8002 8003 8004 8005 8006; do
        fuser -k "$port/tcp" 2>/dev/null && echo "  Killed process on :$port"
    done
fi
