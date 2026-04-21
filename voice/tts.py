"""Text-to-speech — Kokoro (default) with Piper fallback.

Engine is selected via ``voice_settings.tts_engine`` ("kokoro" | "piper").
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import numpy as np
import structlog

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.tts")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PIPER_DIR = _DATA_DIR / "piper"
_KOKORO_DIR = _DATA_DIR / "kokoro"

_KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
_KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

_piper_voice = None  # type: ignore[var-annotated]
_kokoro = None  # type: ignore[var-annotated]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("kokoro_downloading", url=url, dest=str(dest))
    with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.rename(dest)
    log.info(
        "kokoro_downloaded",
        dest=str(dest),
        size_mb=round(dest.stat().st_size / 1024 / 1024, 1),
    )


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        model_path = _KOKORO_DIR / "kokoro-v1.0.onnx"
        voices_path = _KOKORO_DIR / "voices-v1.0.bin"

        if not model_path.exists():
            _download(_KOKORO_MODEL_URL, model_path)
        if not voices_path.exists():
            _download(_KOKORO_VOICES_URL, voices_path)

        log.info("kokoro_loading", model=str(model_path), voice=vs.kokoro_voice)
        t0 = time.monotonic()
        _kokoro = Kokoro(str(model_path), str(voices_path))
        log.info("kokoro_loaded", duration=round(time.monotonic() - t0, 2))
    return _kokoro


def _get_piper():
    global _piper_voice
    if _piper_voice is None:
        import piper
        from piper.download_voices import download_voice

        model_path = _PIPER_DIR / f"{vs.piper_model}.onnx"
        if not model_path.exists():
            log.info("piper_downloading", model=vs.piper_model)
            _PIPER_DIR.mkdir(parents=True, exist_ok=True)
            download_voice(vs.piper_model, _PIPER_DIR)

        log.info("piper_loading", model=str(model_path))
        t0 = time.monotonic()
        _piper_voice = piper.PiperVoice.load(str(model_path))
        log.info("piper_loaded", duration=round(time.monotonic() - t0, 2))
    return _piper_voice


def _synthesize_kokoro(text: str) -> tuple[np.ndarray, int]:
    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(
        text,
        voice=vs.kokoro_voice,
        speed=vs.kokoro_speed,
        lang=vs.kokoro_lang,
    )
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def _synthesize_piper(text: str) -> tuple[np.ndarray, int]:
    from piper.config import SynthesisConfig

    voice = _get_piper()
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

    return np.concatenate(chunks), sample_rate


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Synthesize text to audio. Returns (audio_float32, sample_rate)."""
    t0 = time.monotonic()

    if vs.tts_engine == "kokoro":
        audio, sample_rate = _synthesize_kokoro(text)
    elif vs.tts_engine == "piper":
        audio, sample_rate = _synthesize_piper(text)
    else:
        raise ValueError(
            f"Unknown tts_engine: {vs.tts_engine!r} (expected 'kokoro' or 'piper')"
        )

    duration = time.monotonic() - t0
    log.info(
        "tts_done",
        engine=vs.tts_engine,
        duration=round(duration, 3),
        text_length=len(text),
        audio_seconds=round(len(audio) / sample_rate, 2) if sample_rate else 0.0,
    )
    return audio, sample_rate
