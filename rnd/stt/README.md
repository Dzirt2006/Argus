# Speech-to-Text R&D

Benchmarking and evaluating STT models for Argus.

## Goal

Find the best STT model for English and Russian on our hardware
(i5-13600K CPU, RTX 3070, RTX 1080).

## Models Under Evaluation

| Model                        | Runtime                      | Status      |
|------------------------------|------------------------------|-------------|
| Whisper small                | faster-whisper (CTranslate2) | Baseline    |
| Whisper medium               | faster-whisper (CTranslate2) | Testing     |
| Whisper large-v3-turbo       | faster-whisper (CTranslate2) | Next        |
| NVIDIA Parakeet-TDT 0.6B v3 | sherpa-onnx                  | Planned     |
| NVIDIA Canary-1B-v2          | sherpa-onnx                  | Planned     |
| Qwen3-ASR-0.6B              | HF Transformers / vLLM       | Planned     |

See `docs/stt_models.md` for full accuracy and performance comparison.

## Usage

```bash
# benchmark a model on test audio
python benchmark.py --model whisper-large-v3-turbo --device cpu --audio test_data/

# compare two models side by side
python compare.py --models whisper-medium,whisper-large-v3-turbo --audio test_data/

# test with live microphone
python live_test.py --model whisper-large-v3-turbo --device cpu
```

## Files

- `benchmark.py` — run WER/latency benchmarks on test audio files
- `compare.py` — side-by-side model comparison (accuracy, speed, memory)
- `live_test.py` — interactive live microphone transcription for quick testing
- `test_data/` — sample audio files (English and Russian)
