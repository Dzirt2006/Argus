"""Speech-to-text via faster-whisper.

Flip ``VOICE_WHISPER_DEVICE=cuda`` in .env to move to GPU.
"""

from __future__ import annotations

import time

import numpy as np
import structlog
from faster_whisper import WhisperModel

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.stt")

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazy-load the Whisper model (first call takes a few seconds)."""
    global _model
    if _model is None:
        log.info(
            "whisper_loading",
            model=vs.whisper_model,
            device=vs.whisper_device,
            compute_type=vs.whisper_compute_type,
        )
        t0 = time.monotonic()
        _model = WhisperModel(
            vs.whisper_model,
            device=vs.whisper_device,
            compute_type=vs.whisper_compute_type,
        )
        log.info("whisper_loaded", duration=round(time.monotonic() - t0, 2))
    return _model


def warm_up() -> None:
    """Pre-load the Whisper model so the first turn isn't slow."""
    _get_model()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe a float32 numpy array (16kHz mono) to text.

    Returns the concatenated text of all segments, stripped.
    Returns empty string if nothing was detected.
    """
    model = _get_model()

    t0 = time.monotonic()
    segments, info = model.transcribe(
        audio,
        language=vs.whisper_language,
        beam_size=vs.whisper_beam_size,
        vad_filter=True,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    duration = time.monotonic() - t0

    log.info(
        "stt_done",
        duration=round(duration, 3),
        language=info.language,
        language_prob=round(info.language_probability, 2),
        text_length=len(text),
    )
    return text
