"""Wake word detection via openWakeWord."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sounddevice as sd
import structlog
from openwakeword.model import Model as OWWModel

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.wakeword")


def listen_for_wakeword(on_wake: Callable[[], None]) -> None:
    """Block forever, calling ``on_wake()`` each time the wake word is detected.

    Runs on the main thread. Capture audio in small chunks, feed to
    openWakeWord, and fire the callback when confidence exceeds the threshold.
    """
    chunk_samples = int(vs.sample_rate * vs.audio_chunk_ms / 1000)

    log.info("wakeword_loading", model=vs.wakeword_model)
    oww = OWWModel(wakeword_models=[vs.wakeword_model], inference_framework="onnx")
    log.info("wakeword_ready", threshold=vs.wakeword_threshold)

    with sd.InputStream(
        samplerate=vs.sample_rate,
        channels=vs.channels,
        dtype="int16",
        blocksize=chunk_samples,
    ) as stream:
        while True:
            data, _ = stream.read(chunk_samples)
            audio = data[:, 0] if data.ndim > 1 else data.flatten()

            oww.predict(audio)

            for name, score in oww.get_prediction().items():
                if score >= vs.wakeword_threshold:
                    log.info("wakeword_detected", model=name, score=round(score, 3))
                    oww.reset()
                    on_wake()
