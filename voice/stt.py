"""Speech-to-text via faster-whisper.

Flip ``VOICE_WHISPER_DEVICE=cuda`` in .env to move to GPU.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np
import structlog

from voice.config import voice_settings as vs

log = structlog.get_logger("voice.stt")


def _preload_cuda_libs() -> None:
    """Preload cuBLAS/cuDNN from nvidia-* pip wheels so ctranslate2 can dlopen them.

    The wheel-installed .so files live under site-packages/nvidia/*/lib and are
    not on LD_LIBRARY_PATH or in the ldconfig cache. ctranslate2 loads cuBLAS
    lazily on the first encode, which then fails with 'libcublas.so.12 not
    found'. Loading them here with RTLD_GLOBAL makes the symbols available to
    any subsequent dlopen in the process.
    """
    try:
        import nvidia.cublas  # type: ignore[import-not-found]
        import nvidia.cudnn  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning("cuda_wheels_missing", error=str(e))
        return

    cublas_dir = Path(next(iter(nvidia.cublas.__path__))) / "lib"
    cudnn_dir = Path(next(iter(nvidia.cudnn.__path__))) / "lib"

    # Order matters: cuDNN sub-libs before cuDNN, cublasLt before cublas.
    for path in (
        cudnn_dir / "libcudnn_ops.so.9",
        cudnn_dir / "libcudnn_cnn.so.9",
        cudnn_dir / "libcudnn_graph.so.9",
        cudnn_dir / "libcudnn.so.9",
        cublas_dir / "libcublasLt.so.12",
        cublas_dir / "libcublas.so.12",
    ):
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            log.warning("cuda_lib_preload_failed", lib=path.name, error=str(e))


if vs.whisper_device == "cuda":
    _preload_cuda_libs()

from faster_whisper import WhisperModel  # noqa: E402

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
