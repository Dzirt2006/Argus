# Deployment Notes

## vLLM (Qwen3.5-9B-AWQ-4bit, current deployment target)

- Compose service: `vllm-awq`. Model: `cyankiwi/Qwen3.5-9B-AWQ-4bit`.
- Reference hardware: 5070 Ti. Runs with `--gpu-memory-utilization 0.90`,
  `--max-model-len 16384`, `--max-num-seqs 2`, `--enable-prefix-caching`,
  `--reasoning-parser qwen3`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`.
- The 4B service (`vllm`) is kept for lower-VRAM hosts — see notes below.

## vLLM (Qwen3.5-4B)

- **CUDA graph capture fails** on hybrid models (Qwen3.5) with limited VRAM.
  Fix: add `--max-num-seqs 4` (default is 256, far too high for ~3 GiB KV cache).
  Alternative: `--enforce-eager` disables CUDA graphs entirely.

## MCP Servers (FastMCP 2.x)

- `FastMCP()` no longer accepts `port` in the constructor.
  Pass `port` and `host` to `mcp.run()` instead.

## Agent (langchain-mcp-adapters 0.1.x)

- `MultiServerMCPClient` can no longer be used as `async with` context manager.
  Use direct instantiation + `await client.get_tools()`.

## Voice / Wake Word (openwakeword)

- `tflite-runtime` has no wheels for Python 3.12+. Google deprecated it as a standalone package.
- Use ONNX runtime instead: install `onnxruntime` and pass `inference_framework="onnx"` to `openwakeword.Model`.
- Built-in wake word models ship in both formats, so no conversion needed.
- Install voice deps: `pip install openwakeword --no-deps && pip install onnxruntime numpy scipy scikit-learn tqdm requests`
