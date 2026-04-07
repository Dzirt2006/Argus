"""Text-to-speech via Piper."""

from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import numpy as np
import structlog
import piper
from piper.download_voices import download_voice

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.tts")

_voice: piper.PiperVoice | None = None
_PIPER_DIR = Path(__file__).resolve().parent.parent / "data" / "piper"


def _get_voice() -> piper.PiperVoice:
    """Lazy-load the Piper voice model, downloading if needed."""
    global _voice
    if _voice is None:
        model_path = _PIPER_DIR / f"{vs.piper_model}.onnx"
        if not model_path.exists():
            log.info("piper_downloading", model=vs.piper_model)
            _PIPER_DIR.mkdir(parents=True, exist_ok=True)
            download_voice(vs.piper_model, _PIPER_DIR)

        log.info("piper_loading", model=str(model_path))
        t0 = time.monotonic()
        _voice = piper.PiperVoice.load(str(model_path))
        log.info("piper_loaded", duration=round(time.monotonic() - t0, 2))
    return _voice


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Synthesize text to audio.

    Returns (audio_float32, sample_rate).
    """
    from piper.config import SynthesisConfig

    voice = _get_voice()

    t0 = time.monotonic()
    syn_config = SynthesisConfig(
        speaker_id=vs.piper_speaker_id,
        length_scale=vs.piper_length_scale,
    )

    chunks = []
    sample_rate = 22050
    for audio_chunk in voice.synthesize(text, syn_config):
        chunks.append(audio_chunk.audio_float_array)
        sample_rate = audio_chunk.sample_rate

    if not chunks:
        return np.array([], dtype=np.float32), sample_rate

    audio = np.concatenate(chunks)

    duration = time.monotonic() - t0
    log.info(
        "tts_done",
        duration=round(duration, 3),
        text_length=len(text),
        audio_seconds=round(len(audio) / sample_rate, 2),
    )
    return audio, sample_rate
