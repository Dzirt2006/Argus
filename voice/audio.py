"""Mic capture, speaker playback, and silence detection."""

from __future__ import annotations

import numpy as np
import sounddevice as sd
import structlog

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.audio")


def record_until_silence() -> np.ndarray:
    """Record from the default mic until the user stops speaking.

    Returns a 1-D float32 numpy array at ``voice_settings.sample_rate``.
    Stops when silence exceeds ``silence_duration`` or ``max_record_seconds``
    is reached.
    """
    chunk_samples = int(vs.sample_rate * vs.audio_chunk_ms / 1000)
    max_chunks = int(vs.max_record_seconds * 1000 / vs.audio_chunk_ms)
    silence_chunks_needed = int(vs.silence_duration * 1000 / vs.audio_chunk_ms)

    chunks: list[np.ndarray] = []
    silence_count = 0
    speech_started = False

    log.debug("recording_start")

    with sd.InputStream(
        samplerate=vs.sample_rate,
        channels=vs.channels,
        dtype="float32",
        blocksize=chunk_samples,
    ) as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk_samples)
            audio = data[:, 0] if data.ndim > 1 else data.flatten()
            chunks.append(audio)

            rms = float(np.sqrt(np.mean(audio**2)))

            if rms >= vs.silence_threshold:
                speech_started = True
                silence_count = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_chunks_needed:
                    break

    recording = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    log.debug("recording_done", duration=round(len(recording) / vs.sample_rate, 2))
    return recording


def play_audio(audio: np.ndarray, sample_rate: int | None = None) -> None:
    """Play a numpy audio array through the default speaker. Blocks until done."""
    sr = sample_rate or vs.sample_rate
    log.debug("playback_start", duration=round(len(audio) / sr, 2))
    sd.play(audio, samplerate=sr)
    sd.wait()
    log.debug("playback_done")
