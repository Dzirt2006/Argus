# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run commands

Full stack (vLLM + agent + MCP servers + SearXNG + HA):
```bash
docker compose up                                # default vllm service (4B)
docker compose up vllm-awq agent filesystem ...  # 9B-AWQ variant — current deployment target
```

Interactive use after the stack is up:
```bash
docker compose exec agent python -m agent    # text REPL
docker compose exec agent python -m voice    # wake-word + STT + TTS loop
```

Local dev (run MCP servers on the host, vLLM in Docker):
```bash
bash scripts/start_mcp_servers.sh             # spawns filesystem/system/search/weather
docker compose up vllm                        # or vllm-awq
python -m agent                               # uses settings.mcp_servers (localhost ports)
bash scripts/stop_mcp_servers.sh
```

Tests: `pytest` (the `tests/` package is currently empty).

Wake-word models must be downloaded once:
```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
```

## Architecture

### Agent graph (`agent/agent.py`)

```
START → agent → tools_condition → guardrails → tools → agent → ... → END
```

`tools_condition` is remapped so a tool-call decision routes through `guardrails` first. The graph compiles with `MemorySaver` — required so `interrupt()` (the confirmation gate) can suspend and resume mid-turn. Each CLI turn uses a fresh `thread_id` for the checkpointer.

### Dual LLM bindings

`create_agent` binds two `ChatOpenAI` clients to the same vLLM endpoint: one with `enable_thinking=False` (fast path) and one with `enable_thinking=True`. `agent/tracing.py:make_call_model` selects per-request via `agent.thinking.should_think(user_text)` — a regex/length heuristic over the most recent human message. If you add new "wants reasoning" intents, extend `_THINKING_PATTERNS` there.

### Context injection (read this before changing message handling)

`make_call_model` builds a context block from memory facts + recent summaries + the HA switches list and **appends it to the first SystemMessage in a copy of the message list**. It never mutates `state["messages"]` — the checkpointer relies on message identity for interrupt/resume to work. Preserve that invariant.

The switches block is **names only, not states** — switch state is fetched at tool-call time. This keeps vLLM's prefix cache warm across turns.

### Guardrails (`agent/guardrails.py`)

A graph node, not a decorator or tool wrapper. The node-based design was chosen specifically so destructive tools can call `interrupt()` and yield control back to the CLI/voice loop for user confirmation (see `docs/concepts.md` for the design rationale).

Four checks run in order for every tool call: max-steps cap → `ALLOWED_TOOLS` allowlist → path validation (filesystem tools must stay under `/data`) → confirmation gate for `DESTRUCTIVE_TOOLS`. `ALLOWED_TOOLS` is hardcoded — **adding a new MCP tool requires adding its name here too**, otherwise the LLM's call gets blocked even after the tool loads.

### MCP servers (`mcp_servers/<name>/`)

Each server is an independent FastMCP 2.x process speaking `streamable-http`. The agent discovers tools via `MultiServerMCPClient` (see `agent/mcp.py`). The active server set is in `agent/config.py:Settings.mcp_servers` — weather/calendar/media are currently commented out.

Two FastMCP 2.x gotchas: `FastMCP()` no longer takes `port`; pass it to `mcp.run()`. `MultiServerMCPClient` cannot be used as `async with`; instantiate directly and `await client.get_tools()`.

### Memory (`agent/memory.py`, `agent/summarizer.py`, `mcp_servers/memory/`)

SQLite-only at `/data/memory.db` (WAL mode, shared between agent and memory MCP). Two tables:

- `facts` — exact-key `key/value` store; the most recent N are injected into the prompt every turn under "## Known facts".
- `summaries` — append-only end-of-session notes; the most recent N (default 20) are injected under "## Recent context".

There is **no vector retrieval, embedder, or reranker** — that path was dropped 2026-05-01 in favor of always-inject of recent entries. Don't reintroduce Qdrant/embeddings without checking with the user.

Memory is disabled by default (`MEMORY_ENABLED=false`). `MemoryStore` is a no-op when disabled, so callers don't need to guard.

End-of-session summarization runs in `agent/cli.py` and `voice/pipeline.py` after the conversation ends. It uses a temperature-0, non-thinking LLM call and is rate-limited by `memory_summary_min_turns`. Both `summarize_and_store` and `extract_and_store_facts` are called sequentially.

### Home Assistant integration

Two separate paths, intentionally:

1. **`mcp_servers/switches/`** — narrow HA REST wrapper, **switch + light domains only**. Resolves by friendly name with a 30s registry cache. Custom (not HA's own MCP server) because HA exposes every domain, which blows the prompt budget.
2. **`agent/switches_cache.py`** — separate in-process cache (also 30s TTL) that injects the **names** of switches/lights into the system prompt so the LLM picks the right one without a tool call.

Both require `HA_URL` and `HA_TOKEN` env vars. The system MCP server (`mcp_servers/system/`) is GPU-aware and uses the NVIDIA container runtime.

### Voice pipeline (`voice/`)

`wake-word → record-until-silence → STT → agent → TTS → playback`, with a follow-up mode that records again immediately after the agent responds with a question (`_is_question` heuristic on trailing `?`).

- Wake word: `openwakeword` with **`inference_framework="onnx"`** — `tflite-runtime` has no Python 3.12+ wheels. Install with `pip install openwakeword --no-deps`.
- STT: `faster-whisper`. Default `cpu`/`int8`; switch to `cuda`/`float16` via `VOICE_WHISPER_DEVICE=cuda` (and lower vLLM's `--gpu-memory-utilization` if sharing the GPU).
- TTS: selected by `VOICE_TTS_ENGINE` (`kokoro` or `piper`).
- VAD: silero, 0.5 speech-probability cutoff.
- Confirmation gates are handled in voice too: TTS speaks the question, STT transcribes the answer, agent resumes with `Command(resume="y"|"n")`.

### Tracing

`agent/tracing.py` configures structlog: JSON renderer inside Docker (auto-detected via `/.dockerenv`), pretty console output locally. Every turn binds a `request_id` (8-char UUID) via contextvars so a single turn's log lines can be filtered together. Key events: `llm_call_start/done`, `tool_exec_start/done`, `tool_approved/blocked_*`, `tool_confirmed/denied`, `turn_done`.

## Deployment notes

- The deployment target is **`vllm-awq` (cyankiwi/Qwen3.5-9B-AWQ-4bit)** on a 5070 Ti, not the 4B config in README and docs.
- For the 4B service, `--max-num-seqs 4` is required (default 256 fails CUDA graph capture on hybrid Qwen3.5 with limited VRAM). `--enforce-eager` is the fallback.
- `MODEL_NAME` is auto-discovered from `/v1/models` — `wait_for_vllm()` polls on startup and returns the first listed model ID. Don't hardcode it unless you have a reason to.

## Config

All settings come from `.env` via pydantic-settings. Agent settings are in `agent/config.py:Settings` (no env prefix). Voice settings are in `voice/config.py:VoiceSettings` (env prefix `VOICE_`). Both set `extra: "ignore"` so they coexist in the same `.env` file.

# Use comments sparingly. Only comment complex code.    