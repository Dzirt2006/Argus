# R&D

Research and development playground for Argus AI models.

Training scripts, benchmarks, and evaluation tools for all model components.
These are standalone scripts — not part of the production Argus pipeline.

## Structure

| Directory    | Purpose                                          |
|--------------|--------------------------------------------------|
| `wakeword/`  | Custom wake word training (OpenWakeWord)          |
| `stt/`       | Speech-to-text model benchmarks and evaluation    |
| `tts/`       | Text-to-speech voice experiments                  |
| `llm/`       | LLM inference and tool-calling evaluation         |

## Hardware

| Component    | Spec                  | Role                          |
|--------------|-----------------------|-------------------------------|
| CPU          | Intel i5-13600K       | STT (int8), wake word, TTS    |
| GPU 1        | NVIDIA RTX 3070 (8GB) | vLLM inference (primary)      |
| GPU 2        | NVIDIA RTX 1080 (8GB) | STT/TTS offload (planned)     |

## Quick Start

```bash
cd rnd/
pip install -r requirements.txt  # shared R&D dependencies

# run any experiment
python stt/benchmark.py --model whisper-large-v3-turbo --device cpu
python wakeword/train.py --config wakeword/config.yaml
```
