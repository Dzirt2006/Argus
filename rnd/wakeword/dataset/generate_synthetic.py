"""Generate synthetic wake word samples using Piper TTS.

Uses the libritts multi-speaker model (904 speakers) with speed variation
to create diverse synthetic training data.

Run from IDE or CLI:
    python rnd/wakeword/dataset/generate_synthetic.py
    python rnd/wakeword/dataset/generate_synthetic.py --count 10000 --phrase "hey argus"
"""

import argparse
import io
import json
import random
import struct
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None


def load_config(config_path: str = "rnd/wakeword/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_piper_model(model_name: str) -> tuple[Path | None, Path | None]:
    """Search for a Piper model in common locations."""
    search_dirs = [
        Path("data/piper"),
        Path("rnd/wakeword/models/piper"),
        Path.home() / ".local/share/piper/voices",
        Path("/usr/share/piper-voices"),
    ]

    for base in search_dirs:
        onnx = base / f"{model_name}.onnx"
        config = base / f"{model_name}.onnx.json"
        if onnx.exists() and config.exists():
            return onnx, config

    # Try glob for partial matches
    for base in search_dirs:
        if not base.exists():
            continue
        matches = list(base.rglob(f"*{model_name}*.onnx"))
        if matches:
            onnx = matches[0]
            config = onnx.parent / f"{onnx.name}.json"
            if config.exists():
                return onnx, config

    return None, None


def get_speaker_count(config_path: Path) -> int:
    """Get number of speakers from the Piper model config."""
    with open(config_path) as f:
        model_config = json.load(f)
    num_speakers = model_config.get("num_speakers", 1)
    return num_speakers


def synthesize_sample(
    voice: "PiperVoice",
    text: str,
    speaker_id: int | None,
    length_scale: float,
) -> np.ndarray:
    """Synthesize a single audio sample."""
    audio_bytes = io.BytesIO()

    with wave.open(audio_bytes, "wb") as wav:
        voice.synthesize(
            text,
            wav,
            speaker_id=speaker_id,
            length_scale=length_scale,
        )

    audio_bytes.seek(0)
    with wave.open(audio_bytes, "rb") as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)

    samples = struct.unpack(f"<{n_frames}h", raw)
    audio = np.array(samples, dtype=np.float32) / 32768.0
    return audio


def resample_if_needed(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Simple resample using linear interpolation."""
    if source_sr == target_sr:
        return audio
    ratio = target_sr / source_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wake word samples via Piper TTS")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml", help="Path to config file")
    parser.add_argument("--count", type=int, default=None, help="Number of samples to generate")
    parser.add_argument("--phrase", default=None, help="Override wake phrase")
    parser.add_argument("--model", default=None, help="Piper model name")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    if PiperVoice is None:
        print("Error: piper-tts not installed. Run: pip install piper-tts")
        return

    config = load_config(args.config)
    synth_cfg = config["synthetic"]
    audio_cfg = config["audio"]

    phrase = args.phrase or config["wake_phrase"]
    count = args.count or synth_cfg["num_samples"]
    model_name = args.model or synth_cfg["model"]
    output_dir = Path(args.output_dir or config["paths"]["synthetic_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find model
    onnx_path, config_path = find_piper_model(model_name)
    if onnx_path is None:
        print(f"Error: Piper model '{model_name}' not found.")
        print("Download it with:")
        print(f"  pip install piper-tts")
        print(f"  # Models are auto-downloaded, or manually place in data/piper/")
        print()
        print("For libritts multi-speaker model:")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json")
        return

    print(f"--- Synthetic Sample Generation ---")
    print(f"  Phrase:  \"{phrase}\"")
    print(f"  Model:   {onnx_path.name}")
    print(f"  Samples: {count}")
    print(f"  Output:  {output_dir}")

    # Load model
    print("  Loading Piper model...")
    voice = PiperVoice.load(str(onnx_path), str(config_path))

    num_speakers = get_speaker_count(config_path)
    print(f"  Speakers available: {num_speakers}")

    speed_min, speed_max = synth_cfg["speed_range"]

    # Determine speaker IDs to use
    if synth_cfg["speaker_ids"]:
        speaker_ids = synth_cfg["speaker_ids"]
    elif num_speakers > 1:
        speaker_ids = list(range(num_speakers))
    else:
        speaker_ids = [None]

    # Phrase variations (add period for natural prosody)
    phrase_variants = [
        phrase,
        f"{phrase}.",
        f"{phrase}!",
        f"{phrase}?",
    ]

    existing = list(output_dir.glob("synthetic_*.wav"))
    start_idx = len(existing) + 1
    generated = 0
    errors = 0

    print(f"  Generating...")
    for i in tqdm(range(count), desc="  Generating"):
        try:
            speaker_id = random.choice(speaker_ids)
            length_scale = random.uniform(1.0 / speed_max, 1.0 / speed_min)  # piper: lower = faster
            text = random.choice(phrase_variants)

            audio = synthesize_sample(voice, text, speaker_id, length_scale)

            # Resample to target sample rate if needed
            audio = resample_if_needed(audio, voice.config.sample_rate, audio_cfg["sample_rate"])

            duration = len(audio) / audio_cfg["sample_rate"]
            if duration < audio_cfg["min_duration_sec"] or duration > audio_cfg["max_duration_sec"]:
                continue

            filename = f"synthetic_{start_idx + generated:06d}.wav"
            sf.write(str(output_dir / filename), audio, audio_cfg["sample_rate"], subtype="PCM_16")
            generated += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n    Error on sample {i}: {e}")
            elif errors == 6:
                print(f"\n    (suppressing further errors...)")

    print(f"\n  Generated {generated} samples ({errors} errors)")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
