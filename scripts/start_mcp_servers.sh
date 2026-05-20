#!/usr/bin/env bash
# Start all MCP servers in the background. Run from project root.
# Usage: bash scripts/start_mcp_servers.sh
# Stop:  bash scripts/stop_mcp_servers.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p data

echo "Starting MCP servers..."
python -m mcp_servers.filesystem.server &
echo "  filesystem :8001  (pid $!)"

python -m mcp_servers.system.server &
echo "  system     :8002  (pid $!)"

python -m mcp_servers.search.server &
echo "  search     :8003  (pid $!)"

python -m mcp_servers.weather.server &
echo "  weather    :8004  (pid $!)"

python -m mcp_servers.memory.server &
echo "  memory     :8007  (pid $!)"

# switches needs HA_URL and HA_TOKEN exported in the shell; inherited from env.
python -m mcp_servers.switches.server &
echo "  switches   :8008  (pid $!)"

echo ""
echo "All servers started (filesystem:8001, system:8002, search:8003, weather:8004, memory:8007, switches:8008)."
echo "PIDs saved to /tmp/argus-mcp-pids"
jobs -p > /tmp/argus-mcp-pids
echo "Run 'bash scripts/stop_mcp_servers.sh' to stop them."
echo ""
echo "Next steps:"
echo "  1. Start vLLM:  docker compose up vllm"
echo "  2. Run CLI:     python -m agent.cli"

wait
