# Argus

A fully local, privacy-first home assistant that runs entirely on your own hardware. No cloud. No API keys for core inference. Every request stays on your network.

Argus combines a local LLM (Qwen 3.5 via vLLM) with a voice pipeline and modular tool servers to create a home assistant you actually own. The current deployment target is the AWQ-quantized 9B variant (`cyankiwi/Qwen3.5-9B-AWQ-4bit`, served by the `vllm-awq` compose service) on a 5070 Ti; the 4B service (`vllm`) remains as the lower-VRAM default.

## Why

Commercial assistants send everything to the cloud. Argus doesn't. All inference, speech recognition, and speech synthesis run on consumer GPU hardware (starting from RTX 3070). The only outbound calls are the ones you explicitly configure (weather lookups, calendar sync, web search).

## Architecture

```
Microphone
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Voice Pipeline                                          │
│  Wake Word (OpenWakeWord) → STT (faster-whisper)         │
│       → LangGraph Agent → TTS (Kokoro / Piper) → Speaker │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  LangGraph Agent                                         │
│  [agent] → [guardrails] → [tools] → [agent] → response  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  MCP Tool Servers (FastMCP 2.x, streamable-http)         │
│                                                          │
│  Wired into compose by default:                          │
│  ┌────────────┐ ┌────────┐ ┌────────┐ ┌──────────┐      │
│  │ Filesystem │ │ System │ │ Search │ │ Switches │      │
│  └────────────┘ └────────┘ └────────┘ └──────────┘      │
│  ┌────────┐                                              │
│  │ Memory │                                              │
│  └────────┘                                              │
│                                                          │
│  Coded but not wired into compose:                       │
│  ┌─────────┐ ┌──────────┐ ┌────────┐                    │
│  │ Weather │ │ Calendar │ │ Media  │                    │
│  └─────────┘ └──────────┘ └────────┘                    │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  vLLM  (Qwen3.5-9B-AWQ-4bit by default, or 4B; OpenAI    │
│         API-compatible)                                  │
└──────────────────────────────────────────────────────────┘
```

## Features

- **Fully local LLM inference** — vLLM serving Qwen 3.5 (9B-AWQ on the reference target, 4B as the lower-VRAM default) with tool-calling support, prefix caching, and structured reasoning
- **Voice interaction** — wake word detection (OpenWakeWord), speech-to-text (faster-whisper), text-to-speech (Kokoro or Piper, selected via `VOICE_TTS_ENGINE`), silence detection, all running locally
- **Modular tool system** — MCP (Model Context Protocol) servers over streamable-http, each in its own container
- **Guardrails** — tool allowlisting, path sandboxing, confirmation gates for destructive actions, per-turn step limits
- **Structured tracing** — per-request ID correlation, token counts, execution timing via structlog
- **CLI and voice interfaces** — text REPL for development, voice pipeline for hands-free use
- **Docker-first deployment** — full stack orchestrated via docker-compose

## MCP Tool Servers

| Server | Port | Tools |
|--------|------|-------|
| **Filesystem** | 8001 | `read_file`, `write_file`, `list_directory` (sandboxed to `/data/`) |
| **System** | 8002 | `get_system_uptime`, `get_disk_usage`, `get_memory_usage`, `get_gpu_status` |
| **Search** | 8003 | `web_search`, `web_search_news` (via self-hosted SearXNG) |
| **Weather** | 8004 | `get_current_weather`, `get_forecast` (Open-Meteo, no API key) |
| **Calendar** | 8005 | `get_upcoming_events`, `search_events`, `create_event` (Google Calendar OAuth) |
| **Media** | 8006 | `play_music`, `pause_music`, `skip_track`, `stop_music`, `search_music`, `get_now_playing` (YouTube Music + mpv) |
| **Memory** | 8007 | SQLite-backed facts and session summaries (shared with the agent via `/data/memory.db`) |
| **Switches** | 8008 | Home Assistant switch + light wrapper (resolves by friendly name; needs `HA_URL` and `HA_TOKEN`) |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Inference | vLLM + Qwen 3.5 (9B-AWQ default target, 4B as lower-VRAM option) |
| Agent Framework | LangChain + LangGraph |
| Tool Protocol | FastMCP 2.x (streamable-http) |
| Speech-to-Text | faster-whisper |
| Text-to-Speech | Kokoro (default) or Piper (`VOICE_TTS_ENGINE`) |
| Wake Word | OpenWakeWord (ONNX) |
| Search Engine | SearXNG (self-hosted) |
| Config | Pydantic v2 + pydantic-settings |
| Logging | structlog |
| Runtime | Python 3.12+, asyncio |
| Packaging | uv, Docker |

## Quick Start

### Prerequisites

- NVIDIA GPU with 8 GB+ VRAM (RTX 3070 floor for the 4B service; the AWQ 9B variant is sized for a beefier card — 5070 Ti is the reference target)
- Docker and Docker Compose
- NVIDIA Container Toolkit

### 1. Clone and configure

```bash
git clone https://github.com/your-username/Argus.git
cd Argus
cp .env.example .env
# Edit .env with your settings
```

### 2. Start the full stack

```bash
docker compose up
```

This starts vLLM, the agent, all MCP servers, and SearXNG.

### 3. Use the CLI

```bash
docker compose exec agent python -m agent
```

### 4. Use voice mode

```bash
docker compose exec agent python -m voice
```

Say the wake word, speak your request, and Argus responds through the speaker.

## Configuration

All configuration is via environment variables (`.env` file). See `.env.example` for the full list.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://vllm:8000/v1` | vLLM inference endpoint |
| `MODEL_NAME` | Auto-detected | Model name for the LLM |
| `MAX_STEPS` | `10` | Max tool calls per turn |
| `VOICE_WHISPER_MODEL` | `small` | Whisper model size (`tiny`/`small`/`medium`/`large-v3`) |
| `VOICE_WHISPER_DEVICE` | `cpu` | Whisper device (`cpu`/`cuda`) |
| `VOICE_PIPER_MODEL` | `en_US-lessac-medium` | Piper voice model |
| `VOICE_WAKEWORD_MODEL` | `hey_jarvis` | Wake word model |
| `VOICE_SILENCE_DURATION` | `1.2` | Seconds of silence before stopping recording |

## Project Structure

```
agent/              Core LangGraph agent, CLI, config, guardrails, tracing
mcp_servers/        Six modular MCP tool servers
  filesystem/       File operations (sandboxed)
  system/           System metrics (uptime, disk, RAM, GPU)
  search/           Web search (SearXNG)
  weather/          Weather and forecasts (Open-Meteo)
  calendar/         Google Calendar integration
  media/            Music playback (YouTube Music + mpv)
voice/              Voice pipeline (wake word, STT, TTS, audio I/O)
docs/               Architecture docs, system prompt, development plan
scripts/            Helper scripts for starting/stopping MCP servers
docker-compose.yml  Full stack orchestration
```

## Roadmap

- [x] Phase 1 — Agent core (LangGraph + vLLM + MCP tools)
- [x] Phase 2 — Guardrails (allowlist, path validation, confirmation gates, tracing)
- [x] Phase 3 — Voice pipeline (wake word, STT, TTS, end-to-end)
- [x] Phase 4 — Home Assistant integration + expanded tool suite (switches MCP + HA REST wrapper)
- [x] Phase 5 — Memory (SQLite facts + recent-summary injection) — SQLite-only, the Qdrant/embedding path was dropped
- [ ] Phase 6 — Web dashboard (FastAPI + HTMX) + monitoring + hardening

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.