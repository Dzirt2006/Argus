# Voice Pipeline Setup

## System Dependencies

```bash
sudo apt-get install -y libportaudio2
```

## Python Dependencies

```bash
source .venv/bin/activate
pip install -r requirements-voice.txt
pip install --no-deps "openwakeword>=0.6"
```

**Why `--no-deps` for openwakeword?** It pulls `tflite-runtime` which has no
Python 3.12+ wheels. We use `onnxruntime` instead (listed separately in
requirements-voice.txt).

## First-Run: Download Wake Word Models

```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
```

This downloads pre-trained models (hey_jarvis, alexa, hey_mycroft, etc.) in
both tflite and onnx formats. We use the onnx variants.

## Fixes Applied During Setup

### 1. VoiceSettings rejected non-VOICE_ env vars

`voice/config.py` reads from `.env` which also contains `VLLM_URL`, `MAX_STEPS`,
etc. Pydantic's default `extra = "forbid"` rejected those. Fixed by setting
`"extra": "ignore"` in `model_config`.

### 2. openwakeword 0.6 API change

`Model.get_prediction()` was removed in 0.6. `Model.predict(audio)` now returns
the scores dict directly. Updated `voice/wakeword.py` accordingly.

### 3. No tflite-runtime on Python 3.12+

The `Model()` constructor defaults to tflite inference. We pass
`inference_framework="onnx"` explicitly in `voice/wakeword.py`.

## Configuration (.env)

```bash
# Wake word
VOICE_WAKEWORD_THRESHOLD=0.3    # default 0.5 is too strict for natural speech
                                 # 0.2-0.3 works well, lower = more sensitive

# STT (faster-whisper)
VOICE_WHISPER_MODEL=small        # tiny|small|medium|large-v3
VOICE_WHISPER_DEVICE=cpu         # cpu|cuda
VOICE_WHISPER_COMPUTE_TYPE=int8  # int8 for cpu, float16 for cuda

# TTS (piper)
VOICE_PIPER_MODEL=en_US-lessac-medium
```

## Quick Test: Wake Word Only

```bash
python -c "from voice.wakeword import listen_for_wakeword; listen_for_wakeword(on_wake=lambda: print('WAKE DETECTED!'))"
```

Say "Hey Jarvis" into the mic. You should see `WAKE DETECTED!`.

## Full Pipeline Test

Requires vLLM running + MCP servers started:

```bash
python -m voice
```

## Audio Hardware

The pipeline uses the system default audio device via `sounddevice`.
To list devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`.
To override, set `SOUNDDEVICE_DEFAULT_DEVICE` env var or configure in PulseAudio/PipeWire.
