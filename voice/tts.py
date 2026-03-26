"""Text-to-speech via Piper."""

from __future__ import annotations

import io
import time
import wave

import numpy as np
import structlog
import piper

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.tts")

_voice: piper.PiperVoice | None = None


def _get_voice() -> piper.PiperVoice:
    """Lazy-load the Piper voice model."""
    global _voice
    if _voice is None:
        log.info("piper_loading", model=vs.piper_model)
        t0 = time.monotonic()
        _voice = piper.PiperVoice.load(vs.piper_model)
        log.info("piper_loaded", duration=round(time.monotonic() - t0, 2))
    return _voice


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Synthesize text to audio.

    Returns (audio_float32, sample_rate).
    """
    voice = _get_voice()

    t0 = time.monotonic()
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        voice.synthesize(
            text,
            wf,
            speaker_id=vs.piper_speaker_id,
            length_scale=vs.piper_length_scale,
            sentence_silence=vs.piper_sentence_silence,
        )

    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    duration = time.monotonic() - t0
    log.info(
        "tts_done",
        duration=round(duration, 3),
        text_length=len(text),
        audio_seconds=round(len(audio) / sample_rate, 2),
    )
    return audio, sample_rate
