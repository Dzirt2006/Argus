"""Mic capture, speaker playback, and speech endpointing via silero-vad."""

from __future__ import annotations

import numpy as np
import sounddevice as sd
import structlog
import torch

from silero_vad import load_silero_vad

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.audio")

# Silero-VAD requires exactly 512 samples per inference at 16kHz.
_VAD_FRAME_SAMPLES = 512

_vad_model = None


def _get_vad():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
        log.info("silero_vad_loaded")
    return _vad_model


def warm_up() -> None:
    """Pre-load the VAD model so the first recording isn't slow."""
    _get_vad()


def record_until_silence() -> np.ndarray:
    """Record from the default mic until the user stops speaking.

    Uses silero-vad to detect speech vs. non-speech per 32ms frame. Stops when
    non-speech exceeds ``silence_duration`` after speech has started, or when
    ``max_record_seconds`` is reached.
    """
    vad = _get_vad()
    sr = vs.sample_rate
    max_frames = int(vs.max_record_seconds * sr / _VAD_FRAME_SAMPLES)
    silence_frames_needed = int(vs.silence_duration * sr / _VAD_FRAME_SAMPLES)

    chunks: list[np.ndarray] = []
    silence_count = 0
    speech_started = False

    log.debug("recording_start")

    with sd.InputStream(
        samplerate=sr,
        channels=vs.channels,
        dtype="float32",
        blocksize=_VAD_FRAME_SAMPLES,
    ) as stream:
        for _ in range(max_frames):
            data, _ = stream.read(_VAD_FRAME_SAMPLES)
            audio = data[:, 0] if data.ndim > 1 else data.flatten()
            chunks.append(audio)

            prob = vad(torch.from_numpy(audio), sr).item()

            if prob >= vs.vad_threshold:
                speech_started = True
                silence_count = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_frames_needed:
                    break

    recording = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    log.debug("recording_done", duration=round(len(recording) / sr, 2))
    return recording


def play_audio(audio: np.ndarray, sample_rate: int | None = None) -> None:
    """Play a numpy audio array through the default speaker. Blocks until done."""
    sr = sample_rate or vs.sample_rate
    log.debug("playback_start", duration=round(len(audio) / sr, 2))
    sd.play(audio, samplerate=sr)
    sd.wait()
    log.debug("playback_done")
