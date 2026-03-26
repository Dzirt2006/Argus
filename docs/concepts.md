# Concepts & Explanations

Notes from building Argus — answers to questions that came up during development.

---

## LangGraph: Edges vs Conditional Edges

### Edges (fixed wires)

```python
graph.add_edge("guardrails", "tools")   # guardrails ALWAYS goes to tools
graph.add_edge("tools", "agent")        # tools ALWAYS goes back to agent
```

No decision — when A finishes, B runs.

### Conditional edges (router)

```python
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "guardrails", END: END},
)
```

`tools_condition` is a built-in function that inspects the last AI message:
- Returns `"tools"` if the message contains `tool_calls`
- Returns `END` if it's just a text reply (no tools needed)

The third argument **remaps** the return values. `"tools"` gets redirected to
`"guardrails"` instead of going straight to `ToolNode`. This is how guardrails
are inserted without rewriting the condition function.

---

## Hooking Guardrails into LangGraph

Three approaches were considered:

### Option A: Custom graph node (chosen)

Insert a `guardrails` node between agent and tools:

```
START → agent → tools_condition → guardrails → tools → agent → ... → END
```

**Pros:**
- Most explicit — visible as its own step in traces
- Full access to `MessagesState` (can count steps, inspect history)
- Supports `interrupt()` for confirmation gates (critical for voice)

**Cons:**
- Slightly more graph wiring (need to remap `tools_condition` output)

### Option B: Wrapper around ToolNode

Keep the graph topology the same, wrap `ToolNode` with a class that intercepts:

```python
class GuardedToolNode:
    def __init__(self, tools, allowed_tools):
        self._inner = ToolNode(tools)
    def __call__(self, state):
        # check allowlist, then call self._inner(state)
```

**Pros:** Drop-in replacement, no graph changes.
**Cons:** Confirmation gate is awkward (no interrupt support). Harder to trace
separately — looks like one "tools" step in logs.

### Option C: Tool-level decorators

Wrap each tool function with validation logic.

**Pros:** Good for tool-specific checks (like `_safe_path` in filesystem server).
**Cons:** Cannot do confirmation gates (tools run deep in the stack, can't pause).
Cannot enforce cross-cutting rules like max steps or allowlists (no state access).

### Why Option A wins

1. Confirmation gate needs graph-level control. For voice, "confirm" means the
   agent speaks a question and waits for the user to respond — that's a graph
   interrupt.
2. Traceability. A dedicated node shows up in structured logs as its own step.

---

## LangGraph Interrupts (Confirmation Gate)

### The problem

When the LLM says "call `write_file`", the graph needs to **pause**, ask the
user "OK to write?", and only continue if they approve. The graph execution must
literally stop and yield control back to the CLI/voice loop.

### `interrupt()` — the mechanism

```python
from langgraph.types import interrupt

def check_guardrails(state):
    for tc in last.tool_calls:
        if tc["name"] in DESTRUCTIVE_TOOLS:
            answer = interrupt({"action": "confirm", "description": "..."})
            # execution STOPS here, resumes when caller provides a value
            if answer == "y":
                # approved
```

Step by step:
1. Graph runs: `agent → guardrails → ...`
2. `guardrails` calls `interrupt({...})`
3. Graph execution **stops**. Returns state + interrupt payload to caller.
4. Caller (CLI) prints "Allow write_file? [y/n]"
5. User types "y"
6. Caller calls `graph.invoke(Command(resume="y"), config)`
7. Graph **resumes** inside guardrails — `interrupt()` returns `"y"`
8. Guardrails approves → `tools` node executes the tool

Think of `interrupt()` like `input()` but for graphs — it suspends the
coroutine, hands control to the outer loop, and resumes when a value is provided.

### Checkpointer requirement

For the graph to resume where it left off, it must save state. LangGraph uses a
"checkpointer" for this:

```python
from langgraph.checkpoint.memory import MemorySaver
graph.compile(checkpointer=MemorySaver())
```

Without a checkpointer, `interrupt()` raises an error. `MemorySaver` stores
state in memory. Later (Phase 5) this can be swapped for `SqliteSaver` so
conversations survive restarts.

### CLI side

```python
result = await agent.ainvoke({"messages": messages}, config)

while result.get("__interrupt__"):
    # ask user for each pending confirmation
    for irq in result["__interrupt__"]:
        answer = input(f"Allow {irq.value}? [y/n] ")
        result = await agent.ainvoke(Command(resume=answer), config)
```

The `while` loop handles cases where the LLM called multiple destructive tools
in one turn — each triggers a separate interrupt.

For voice (Phase 3), this same pattern becomes: speak the question via TTS, wait
for STT, resume with the transcribed answer.

---

## Guardrail Policies

The four checks run in order for each tool call:

1. **Max steps** — counts `ToolMessage`s in conversation. If >= limit, blocks
   all further tool calls with an explanation so the LLM wraps up gracefully.
   Backup to LangGraph's `recursion_limit` (which kills the graph hard).

2. **Allowlist** — tool name must be in `ALLOWED_TOOLS`. Matters when more MCP
   servers are added (Phase 4) — can scope per context (e.g. "guest mode" =
   read only).

3. **Path validation** — filesystem tools must target `/data/`. Defence-in-depth:
   the MCP server already does `_safe_path()`, but this catches it at the agent
   level for any future MCP server with file access.

4. **Confirmation gate** — destructive tools (e.g. `write_file`) trigger
   `interrupt()`. Graph pauses, CLI/voice asks user, resumes on approval.

---

## Structured Tracing (Phase 2.2)

### What gets logged

- **LLM calls**: `llm_call_start` / `llm_call_done` with duration, input/output
  token counts, list of tool calls requested
- **Guardrail decisions**: `tool_approved`, `tool_blocked_allowlist`,
  `tool_blocked_path`, `tool_confirmed`, `tool_denied`, `max_steps_reached`
- **Tool execution**: `tool_exec_start` / `tool_exec_done` with tool name,
  duration, and truncated result preview
- **Turn summary**: `turn_done` with total duration

### Request ID correlation

Each user turn gets a short UUID (`request_id`). All log lines within that turn
share the same ID via structlog's contextvars, so you can filter logs for a
single request.

### Output format

- **In Docker**: JSON (machine-readable, ready for Prometheus/dashboard in Phase 6)
- **Locally**: Pretty colored console output (human-readable during development)

Detection is automatic via `os.path.exists("/.dockerenv")`.

### Sample trace

```
[a3f1c9e2] llm_call_start  input_messages=2
[a3f1c9e2] llm_call_done   duration=1.2s tokens_in=245 tokens_out=38 tool_calls=["write_file"]
[a3f1c9e2] tool_confirmed  tool=write_file
[a3f1c9e2] tool_exec_start tools=["write_file"]
[a3f1c9e2] tool_exec_done  tool=write_file duration=0.05s result_preview="Success: wrote 5 bytes"
[a3f1c9e2] llm_call_done   duration=0.8s tokens_in=310 tokens_out=18 tool_calls=null
[a3f1c9e2] turn_done       duration=2.1s
```

Reading this trace: the slowest step is the first LLM call (1.2s). If the agent
picked the wrong tool, you'd see it in `tool_calls=["wrong_tool"]` on the
`llm_call_done` line. Token counts tell you how much context is accumulating.

---

## Voice Pipeline (Phase 3)

### Stack

| Component | Library | Role | Runs on |
|---|---|---|---|
| **STT** | `faster-whisper` | Speech → text | CPU (default) or GPU |
| **TTS** | `piper-tts` | Text → speech | CPU |
| **Wake word** | `openwakeword` | Always-on keyword detection | CPU |
| **Audio I/O** | `sounddevice` | Mic capture & speaker playback | Host |

All libraries are MIT or Apache 2.0 licensed.

### Audio flow

```
Mic (always on)
  → OpenWakeWord (listens for "hey jarvis" in ~80ms chunks)
  → Wake detected → record_until_silence()
  → faster-whisper transcribes audio → text
  → text sent to LangGraph agent (same as CLI)
  → agent response → Piper synthesizes speech
  → sounddevice plays audio through speaker
  → back to listening for wake word
```

### Confirmation via voice

When the agent hits a guardrail confirmation gate (e.g. `write_file`):
1. Pipeline speaks the question via Piper: "Should I proceed with write_file(...)?"
2. Records the user's answer via mic
3. Transcribes with Whisper
4. Matches against approval words ("yes", "yeah", "sure", "go ahead", etc.)
5. Resumes the graph with `Command(resume="y")` or `"n"`

### CPU → GPU switch

One env var change — no code changes needed:

```bash
VOICE_WHISPER_DEVICE=cuda
VOICE_WHISPER_COMPUTE_TYPE=float16
```

If running on the same GPU as vLLM, lower vLLM's memory reservation:
`--gpu-memory-utilization 0.55` (from 0.70) to leave room for Whisper.

### Config overview (`voice/config.py`)

All settings are env vars with `VOICE_` prefix:

| Var | Default | Notes |
|---|---|---|
| `VOICE_WHISPER_MODEL` | `small` | tiny (~50ms), small (~150ms), medium (~400ms) |
| `VOICE_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `VOICE_WHISPER_COMPUTE_TYPE` | `int8` | `int8` for CPU, `float16` for GPU |
| `VOICE_WHISPER_BEAM_SIZE` | `1` | 1 = greedy (fastest), 5 = beam search (better accuracy) |
| `VOICE_PIPER_MODEL` | `en_US-lessac-medium` | Piper voice model name |
| `VOICE_PIPER_LENGTH_SCALE` | `1.0` | < 1.0 = faster speech |
| `VOICE_WAKEWORD_MODEL` | `hey_jarvis` | Swap to `hey_clawed` after training |
| `VOICE_WAKEWORD_THRESHOLD` | `0.5` | Higher = fewer false positives |
| `VOICE_SILENCE_THRESHOLD` | `0.02` | RMS level below which = silence |
| `VOICE_SILENCE_DURATION` | `1.2` | Seconds of silence before stopping recording |

### How to test

```bash
# Install voice dependencies
pip install -r requirements-voice.txt

# Run the voice pipeline
python -m voice
```

Expected behavior:
1. Prints "Listening for wake word..."
2. Say "Hey Jarvis" → detects wake word
3. Prints "Listening..." → records until you stop speaking
4. Transcribes and shows: `You: <your text>`
5. Agent processes, shows: `Argus: <response>`
6. Speaks the response through speakers
7. Back to "Listening for wake word..."

### Testing individual components

```bash
# Test STT only
python -c "
from voice.audio import record_until_silence
from voice.stt import transcribe
audio = record_until_silence()
print(transcribe(audio))
"

# Test TTS only
python -c "
from voice.tts import synthesize
from voice.audio import play_audio
audio, sr = synthesize('Hello, I am Argus.')
play_audio(audio, sr)
"
```

### Where latency hides

1. **Whisper model load** — first call only (few seconds). Lazy-loaded, cached after.
2. **STT transcription** — depends on model size and audio length. `tiny` ~50ms, `small` ~150ms.
3. **LLM first token** — vLLM inference, typically 0.5-2s on 3070.
4. **Piper synthesis** — usually fast (~100-300ms), depends on response length.
5. **Silence detection** — 1.2s of silence before recording stops (configurable).

Total target: wake-to-speech < 3 seconds for simple commands.
