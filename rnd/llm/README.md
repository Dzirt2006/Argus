# LLM R&D

LLM inference experiments and model evaluation for Argus.

## Goal

Evaluate local LLM models for the Argus agent — tool calling accuracy,
response quality, latency, and VRAM usage on RTX 3070 (8 GB).

## Models Under Evaluation

| Model              | Params | VRAM (fp16) | Tool Calling | Status   |
|--------------------|--------|-------------|--------------|----------|
| Qwen3.5-4B        | 4B     | ~6 GB       | Yes          | Current  |

## Usage

```bash
# benchmark model on tool-calling scenarios
python benchmark.py --model Qwen/Qwen3.5-4B --scenarios scenarios/

# test tool calling accuracy
python tool_call_test.py --model Qwen/Qwen3.5-4B --tools ../../mcp_servers/

# compare models
python compare.py --models Qwen/Qwen3.5-4B,other-model --scenarios scenarios/
```

## Files

- `benchmark.py` — latency, throughput, VRAM usage benchmarks
- `tool_call_test.py` — test tool selection and argument accuracy
- `compare.py` — side-by-side model comparison
- `scenarios/` — test scenarios (multi-turn, tool chains, edge cases)
