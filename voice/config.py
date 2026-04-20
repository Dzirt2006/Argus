"""Voice pipeline configuration.

Change ``whisper_device`` to ``cuda`` to move STT to GPU.
Adjust ``whisper_model`` for speed/accuracy tradeoff.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class VoiceSettings(BaseSettings):
    model_config = {"env_prefix": "VOICE_", "env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}

    # -- STT (faster-whisper) -------------------------------------------------
    whisper_model: str = "small"          # tiny | small | medium | large-v3
    whisper_device: str = "cpu"           # cpu | cuda
    whisper_compute_type: str = "int8"    # int8 (cpu) | float16 (cuda)
    whisper_language: str = "en"
    whisper_beam_size: int = 1            # 1 = greedy (fastest), 5 = beam search

    # -- TTS (piper) ----------------------------------------------------------
    piper_model: str = "en_US-lessac-medium"
    piper_speaker_id: int | None = None
    piper_length_scale: float = 1.0       # < 1.0 = faster speech, > 1.0 = slower
    piper_sentence_silence: float = 0.3   # pause between sentences (seconds)

    # -- Wake word (openwakeword) ---------------------------------------------
    wakeword_model: str = "hey_jarvis"    # swap to custom "hey_clawed" after 3.3
    wakeword_threshold: float = 0.5       # 0.0-1.0, higher = fewer false positives

    # -- Audio ----------------------------------------------------------------
    sample_rate: int = 16000              # 16kHz for Whisper, OpenWakeWord, Silero-VAD
    channels: int = 1
    audio_chunk_ms: int = 80              # chunk size for wake word processing
    vad_threshold: float = 0.5            # silero-vad speech probability cutoff
    silence_duration: float = 0.5         # seconds of non-speech before stopping recording
    max_record_seconds: float = 7.0       # hard cap on recording length
    followup_start_timeout: float = 5.0   # seconds to wait for user to start speaking in follow-up mode
    max_followup_turns: int = 5           # cap on consecutive follow-up turns before falling back to wake word


voice_settings = VoiceSettings()
