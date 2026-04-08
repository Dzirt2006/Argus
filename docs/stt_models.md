# Speech-to-Text Model Comparison

Comparison of open-source STT models relevant to Argus, benchmarked for both
English and Russian accuracy, CPU/GPU performance, and resource usage.

## Model Overview

| Model                        | Params | Runtime                      | License    | Languages | CPU int8 support |
|------------------------------|--------|------------------------------|------------|-----------|------------------|
| Whisper tiny                 | 39M    | faster-whisper (CTranslate2) | MIT        | 99        | Yes              |
| Whisper small                | 244M   | faster-whisper (CTranslate2) | MIT        | 99        | Yes              |
| Whisper medium               | 769M   | faster-whisper (CTranslate2) | MIT        | 99        | Yes              |
| Whisper large-v3             | 1,550M | faster-whisper (CTranslate2) | MIT        | 99        | Yes (slow)       |
| Whisper large-v3-turbo       | 809M   | faster-whisper (CTranslate2) | MIT        | 99        | Yes              |
| NVIDIA Parakeet-TDT 0.6B v3 | 600M   | sherpa-onnx / NeMo           | CC-BY-4.0  | 25 EU     | Partial (ONNX)   |
| NVIDIA Canary-1B-v2          | 978M   | sherpa-onnx / NeMo           | CC-BY-4.0  | 25 EU     | No (GPU only)    |
| Qwen3-ASR-0.6B              | ~900M  | HF Transformers / vLLM       | Apache 2.0 | 52        | Impractical      |
| Qwen3-ASR-1.7B              | ~2B    | HF Transformers / vLLM       | Apache 2.0 | 52        | Impractical      |

## English Accuracy (WER% — lower is better)

Benchmarked on LibriSpeech test-clean / test-other.

| Model                        | test-clean | test-other | HF Open ASR avg |
|------------------------------|------------|------------|-----------------|
| Whisper tiny                 | 6.7        | 15.0       | —               |
| Whisper small                | 3.2        | 6.7        | —               |
| Whisper medium               | 2.7        | 5.6        | —               |
| Whisper large-v3             | 2.01       | 3.91       | 7.44            |
| Whisper large-v3-turbo       | 2.1        | 4.24       | ~7.7            |
| Parakeet-TDT 0.6B v3        | **1.93**   | **3.59**   | 6.34            |
| Canary-1B-v2                 | 2.18       | 3.56       | 7.15            |
| Qwen3-ASR-0.6B              | 2.11       | 4.55       | —               |
| Qwen3-ASR-1.7B              | **1.63**   | **3.38**   | —               |

## Russian Accuracy (WER% — lower is better)

| Model                        | Fleurs (ru)  | Common Voice (ru) | CoVoST2 (ru) |
|------------------------------|--------------|-------------------|---------------|
| Whisper tiny                 | ~31          | ~40.6             | —             |
| Whisper small                | ~14–15       | ~15.0             | —             |
| Whisper medium               | ~5–7         | ~9.3              | —             |
| Whisper large-v3             | ~5–6         | ~9.8              | —             |
| Whisper large-v3-turbo       | ~5–6 (est.)  | —                 | —             |
| Parakeet-TDT 0.6B v3        | **5.51**     | —                 | **3.00**      |
| Canary-1B-v2                 | 6.90         | —                 | 5.14          |
| Qwen3-ASR-0.6B              | 9.91         | 14.07             | —             |
| Qwen3-ASR-1.7B              | **5.99**     | **8.28**          | —             |

## Resource Usage

### RAM / VRAM

| Model                        | RAM (CPU, int8)    | VRAM (GPU, float16) |
|------------------------------|--------------------|---------------------|
| Whisper tiny                 | ~200–300 MB        | ~1 GB               |
| Whisper small                | ~500 MB – 1.5 GB  | ~2 GB               |
| Whisper medium               | ~1.0–2.1 GB       | ~5 GB               |
| Whisper large-v3             | ~2.3–3.9 GB       | ~10 GB              |
| Whisper large-v3-turbo       | ~1.5–2.3 GB       | ~2 GB               |
| Parakeet-TDT 0.6B v3        | ~1.5–2 GB (est.)  | ~1.5 GB             |
| Canary-1B-v2                 | N/A (GPU only)     | ~6 GB               |
| Qwen3-ASR-0.6B              | ~2–3 GB (est.)    | ~1.5 GB             |
| Qwen3-ASR-1.7B              | ~4–6 GB (est.)    | ~3.5 GB             |

### Latency (short utterance, ~5–10s audio)

| Model                        | CPU (i5-13600K, int8) | GPU (RTX 3070/1080, float16) |
|------------------------------|-----------------------|------------------------------|
| Whisper tiny                 | ~50 ms                | ~30 ms                       |
| Whisper small                | ~150 ms               | ~80 ms                       |
| Whisper medium               | ~400 ms               | ~150 ms                      |
| Whisper large-v3             | ~1–2 s                | ~300–400 ms                  |
| Whisper large-v3-turbo       | ~300–500 ms           | ~60–100 ms                   |
| Parakeet-TDT 0.6B v3        | ~100–200 ms (est.)    | ~30–50 ms                    |
| Canary-1B-v2                 | N/A                   | ~100–200 ms                  |
| Qwen3-ASR-0.6B              | ~45 s (impractical)   | ~200–400 ms                  |
| Qwen3-ASR-1.7B              | Impractical           | ~400–800 ms                  |

GPU RTFx (real-time factor, higher = faster):
- Parakeet-TDT 0.6B v3: **3,332x** realtime
- Canary-1B-v2: **749x** realtime
- Whisper large-v3: **145x** realtime
- Whisper large-v3-turbo: ~**815x** realtime (est.)

## Integration Complexity

| Model                        | Drop-in for Argus? | What's needed                                              |
|------------------------------|--------------------|------------------------------------------------------------|
| Whisper (any size)           | **Yes**            | Change `VOICE_WHISPER_MODEL` in `.env`                     |
| Whisper large-v3-turbo       | **Yes**            | Config change only                                         |
| Parakeet-TDT 0.6B v3        | No                 | Replace faster-whisper with sherpa-onnx in `voice/stt.py`  |
| Canary-1B-v2                 | No                 | sherpa-onnx or NeMo integration, GPU required              |
| Qwen3-ASR                    | No                 | vLLM or HF Transformers backend, GPU required              |

## Beam Search

| Model                        | Supports beam search   | Default      | Impact                              |
|------------------------------|------------------------|--------------|-------------------------------------|
| Whisper (all)                | Yes                    | beam_size=5  | ~0.5–2% WER improvement over greedy |
| Parakeet-TDT 0.6B v3        | Modified (sherpa-onnx) | Greedy       | Minimal improvement                 |
| Canary-1B-v2                 | Yes                    | —            | Standard seq2seq benefit            |
| Qwen3-ASR                    | **No**                 | Greedy only  | N/A (by design)                     |

Note: Argus defaults to `beam_size=1` (greedy). Increasing to 5 improves accuracy
at the cost of ~2x latency.

## Recommendations for Argus

### CPU-only (i5-13600K)

**Best option: Whisper large-v3-turbo** (int8, CTranslate2)
- Near-large-v3 accuracy for English and Russian
- ~300–500 ms latency, ~1.5–2.3 GB RAM
- Zero code changes — config only

### With RTX 1080 as dedicated STT GPU

**Best accuracy: Parakeet-TDT 0.6B v3**
- Best English WER (1.93% LS-clean), strong Russian (5.51% Fleurs)
- Extremely fast (~30–50 ms on GPU)
- Requires sherpa-onnx integration (code change in `voice/stt.py`)

**Easiest: Whisper large-v3-turbo on CUDA**
- ~60–100 ms latency, ~2 GB VRAM
- Same faster-whisper runtime, config change only

### Best Russian accuracy (GPU required)

1. Parakeet-TDT 0.6B v3 — 5.51% Fleurs, 3.00% CoVoST2
2. Qwen3-ASR-1.7B — 5.99% Fleurs, 8.28% Common Voice
3. Whisper large-v3 — ~5–6% Fleurs, ~9.8% Common Voice

## Sources

- [OpenAI Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
- [Whisper large-v3-turbo — HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
- [NVIDIA Parakeet-TDT 0.6B v3 — HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NVIDIA Canary-1B-v2 — HuggingFace](https://huggingface.co/nvidia/canary-1b-v2)
- [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)
- [faster-whisper benchmarks](https://github.com/SYSTRAN/faster-whisper)
- [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [Northflank STT benchmarks 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)