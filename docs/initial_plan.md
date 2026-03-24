# Project Clawed — Final Build Plan

## Legend

- 🤖 **CODE** — Claude Code writes this
- 🧠 **YOU** — you write this yourself
- 🛑 **EXPLAIN** — Claude Code stops coding, explains what just happened and why
- ✅ **TEST** — manual testing checkpoint before moving on

---

## Phase 1 — Agent core (1 weekend)

### 1.1 Scaffold + vLLM

🤖 **CODE:** Project structure, pyproject.toml, Dockerfiles, docker-compose.yml
with vLLM service, .env.example, agent/config.py

🛑 **EXPLAIN:** Walk through docker-compose.yml — what each vLLM flag does,
why `ipc: host`, what `--tool-call-parser qwen3_coder` means, how
`--language-model-only` saves VRAM by skipping the vision encoder.

✅ **TEST:** `docker compose up vllm`, then:
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"hi"}]}'
```

### 1.2 LangGraph agent

🧠 **YOU:** Write `agent/agent.py` — the ~15-line LangGraph state machine
with call_model node, ToolNode, and tools_condition routing.

🛑 **EXPLAIN:** Claude Code explains the graph execution model:
- What `MessagesState` holds and how it accumulates
- What `tools_condition` checks (are there tool_calls in the last message?)
- Why the "tools" → "agent" edge creates the loop
- When the loop terminates (no tool_calls → END)
- What `ToolNode` does internally (executes tools, appends ToolMessages)

### 1.3 First MCP server

🧠 **YOU:** Write `mcp_servers/filesystem/server.py` with FastMCP —
3 tools: read_file, write_file, list_directory. Pay attention to
docstrings — the LLM only sees these when choosing tools.

🛑 **EXPLAIN:** Claude Code explains MCP protocol flow:
- How the agent discovers tools (list_tools RPC call at startup)
- How tool schemas get converted to OpenAI function format
- How a tool call travels: LLM → agent → MCP client → HTTP → MCP server → response back
- Difference between stdio and streamable-http transport

🤖 **CODE:** `mcp_servers/system/server.py`, MCP client wiring via
MultiServerMCPClient, CLI REPL, tests, remaining Dockerfiles,
full docker-compose.yml

📖 Read the system MCP server — are the tool descriptions good enough?

✅ **TEST:** `docker compose up`, then in CLI:
- "What files are in /data/notes?" → filesystem tool
- "Check GPU status and write a summary to /data/notes/gpu.md" → multi-step
- "What's 2+2?" → no tool call, direct answer
- Try something ambiguous — see what happens

---

## Phase 2 — Guardrails + traces (1-2 evenings)

### 2.1 Guardrails

🧠 **YOU:** Write `agent/guardrails.py`:
- Tool allowlist — which tools can be called per context
- Path validation — filesystem ops restricted to /data/
- Confirmation gate — destructive tools (write, delete) need user OK
- Max steps — cap tool call loops (backup to LangGraph's recursion_limit)

🛑 **EXPLAIN:** Claude Code explains how to hook guardrails into LangGraph:
- Option A: custom node between "agent" and "tools" in the graph
- Option B: wrapper around ToolNode that intercepts before execution
- Option C: tool-level decorators
- Tradeoffs of each approach, which fits best for voice (where
  confirmation means asking the user to speak "yes")

### 2.2 Traces

🤖 **CODE:** Structured logging with structlog — every LLM call,
tool call, timing, token usage. JSON format. Pretty terminal output.

🛑 **EXPLAIN:** Walk through a sample trace of a multi-step request.
Point out: which step was slowest, how many tokens each LLM call used,
where you'd look if the agent picked the wrong tool.

✅ **TEST:**
- Agent tries to read /etc/passwd → guardrail blocks
- Agent tries to delete a file → confirmation gate fires
- Run a multi-step request → read the trace, understand every step

---

## Phase 3 — Voice pipeline (3-4 evenings)

### 3.1 Voice services

🤖 **CODE:** Docker services for Whisper, Piper, OpenWakeWord.
Voice pipeline orchestrator: wake → STT → agent.chat() → TTS → speaker.

🛑 **EXPLAIN:** The full audio flow, step by step:
- Mic → OpenWakeWord (always listening, tiny CPU footprint)
- Wake word detected → audio stream redirected to Whisper
- Whisper transcribes → text to your LangGraph agent
- Agent responds → text to Piper
- Piper synthesizes → audio to speaker
- Wyoming protocol: what events flow between services, TCP connections
- Where latency hides: Whisper model load, LLM first-token, Piper synthesis
- Why streaming TTS matters (Piper can start speaking before full response)

### 3.2 Tuning

🧠 **YOU DO:**
- Test Whisper models: tiny (~50ms) vs small (~150ms) vs medium (~400ms)
- Measure end-to-end latency at each stage
- Pick Piper voice, test speed settings
- Adjust `--gpu-memory-utilization` if Whisper runs on same GPU

### 3.3 Wake word

🧠 **YOU DO:** Train "Hey Clawed" via openWakeWord pipeline.

🛑 **EXPLAIN:** How openWakeWord training works — Piper generates
thousands of synthetic clips, augmented with room acoustics and noise,
fed into a small classifier. Why 3-4 syllables work better than 2.

✅ **TEST:** "Hey Clawed, what time is it?" → spoken response < 3 seconds.
If not, Claude Code helps identify the bottleneck from trace timestamps.

---

## Phase 4 — Home + tools (1-1.5 weeks)

### 4.1 Home Assistant

🧠 **YOU DO:** Install HA in Docker, configure devices, enable MCP server.
This is YAML config and HA UI — can't be meaningfully coded by an agent.

🛑 **EXPLAIN:** How HA's built-in MCP server works:
- Entity exposure — which devices the agent can see/control
- Long-lived access tokens for auth
- What tools HA exposes (call_service, get_states, etc.)
- How entity IDs map to physical devices

### 4.2 MCP servers

🤖 **CODE:** 4-5 MCP servers (search, calendar, weather, notes, media).
All follow same fastmcp pattern.

📖 Read each server. Focus on tool descriptions.

### 4.3 Tool description engineering

🧠 **YOU DO:** The most important manual work in the project.

Test every tool with natural language. The LLM picks tools based entirely
on name + description + parameter descriptions. Iterate until reliable.

🛑 **EXPLAIN:** Claude Code reviews your descriptions and suggests improvements.
Explains common failure patterns:
- Overlapping descriptions (LLM can't decide between two tools)
- Too vague ("do stuff with files" — when does LLM use this vs notes tool?)
- Missing context ("Search the web" — LLM doesn't know this can do weather too)
- Parameter descriptions matter too ("city: str" vs "city: str — full city name, e.g. 'Madison, WI'")

### 4.4 Thinking mode

🧠 **YOU:** Write routing logic — when to enable Qwen3.5's thinking per-request.

🛑 **EXPLAIN:** How thinking mode works under the hood:
- `<think>` tags in model output, reasoning-parser strips them
- Trade-off: better multi-step reasoning vs 2-5x more tokens (slower)
- When it helps (planning, multi-tool chains) vs hurts (simple commands)
- How to pass `enable_thinking` per-request via vLLM's extra_body

✅ **TEST:**
- "Turn off the living room lights" → HA tool, correct entity
- "What's the weather tomorrow?" → weather tool
- "I'm heading to bed" → multi-tool chain (lights, locks, thermostat)
- "Plan my week based on calendar and weather" → thinking mode activates

---

## Phase 5 — Memory + multi-room (3-4 evenings)

### 5.1 Memory

🧠 **YOU:** Write the retrieval logic — summarize past conversations,
embed, store in Qdrant, retrieve similar ones, inject into system prompt.

🤖 **CODE:** Qdrant Docker service, client boilerplate, SQLite structured
store, system prompt template with memory injection slot.

🛑 **EXPLAIN:** How RAG-style memory works for agents:
- Embedding models compress text to vectors (nomic-embed-text)
- Qdrant finds similar vectors via approximate nearest neighbor search
- Retrieved context gets prepended to the system prompt
- Why top-3 results usually beats top-10 (too much context confuses small models)
- Structured memory vs vector memory — when to use each
  (user preferences → SQLite, conversation history → Qdrant)

### 5.2 Multi-room

🧠 **YOU DO:** Flash HA Voice PE satellites, configure WiFi + Wyoming

🤖 **CODE:** Room-aware context injection, multi-room announcements

🛑 **EXPLAIN:** How room context flows:
- Satellite detects wake word → sends audio with satellite_id
- Pipeline resolves satellite_id → room name
- Room name injected into system prompt as {satellite_room}
- Agent uses room context for ambiguous commands ("the lights" = this room)

✅ **TEST:**
- "Hey Clawed, remind me about my dentist appointment" (from past memory)
- Voice works from 3+ rooms
- "Turn off the lights" in bedroom → only bedroom lights

---

## Phase 6 — Dashboard + hardening (ongoing)

🤖 **CODE:** FastAPI + HTMX dashboard (conversations, traces, system status),
Ansible playbook, Prometheus monitoring, backup scripts

🧠 **YOU DESIGN:** Failure policies, retention rules, guest mode, scheduled routines

---

## Timeline

| Phase | Time | What eats the hours |
|-------|------|---------------------|
| **1. Agent core** | 1 weekend | Qwen3.5 tool-call debugging |
| **2. Guardrails** | 1-2 evenings | Straightforward Python |
| **3. Voice** | 3-4 evenings | Latency tuning, wake word training |
| **4. Home + tools** | 1-1.5 weeks | HA config rabbit hole, description iteration |
| **5. Memory + rooms** | 3-4 evenings | Retrieval tuning, satellite hardware |
| **Total** | **~4-5 weeks** | |

CLI agent: end of weekend 1. Voice: end of week 2.

---

## Stack

- **Inference:** vLLM Docker → `http://vllm:8000/v1`
- **Model:** Qwen3.5-4B + `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
- **Agent:** LangChain + LangGraph, `ChatOpenAI`
- **MCP:** FastMCP servers + `langchain-mcp-adapters`
- **Voice:** OpenWakeWord → Whisper → Agent → Piper (Wyoming)
- **Memory:** Qdrant + SQLite
- **Home:** Home Assistant w/ built-in MCP server
- **Guardrails:** Custom (tool allowlist, path validation, confirmation gate)
- **Start on RTX 3070.** Migrate to basement server later.