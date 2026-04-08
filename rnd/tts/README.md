# Text-to-Speech R&D

TTS voice experiments and evaluation for Argus.

## Goal

Evaluate TTS models for natural-sounding, low-latency speech synthesis
in English and Russian.

## Models Under Evaluation

| Model              | Runtime       | Languages        | Status   |
|--------------------|---------------|------------------|----------|
| Piper (lessac)     | piper-tts     | English          | Current  |
| Piper (other)      | piper-tts     | English, Russian | Testing  |
| Kokoro             | ONNX          | 8 languages      | Planned  |

## Usage

```bash
# compare voices on sample text
python compare_voices.py --text "Hello, how can I help you?" --models piper,kokoro

# benchmark latency
python benchmark.py --model piper --voice en_US-lessac-medium --iterations 50

# test Russian voices
python compare_voices.py --text "Привет, чем могу помочь?" --lang ru
```

## Files

- `compare_voices.py` — generate and play audio from multiple TTS models
- `benchmark.py` — measure synthesis latency and audio quality metrics
