"""Augment wake word samples with noise, reverb, speed, and volume changes.

Creates multiple augmented versions of each positive sample to improve
model robustness across different acoustic conditions.

Run from IDE or CLI:
    python rnd/wakeword/dataset/augment.py
    python rnd/wakeword/dataset/augment.py --input-dir data/positive/ --multiplier 5
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

try:
    import audiomentations as am

    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False


def load_config(config_path: str = "rnd/wakeword/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# --- Manual augmentation fallbacks (when audiomentations is not installed) ---


def add_noise(audio: np.ndarray, noise_files: list[Path], snr_db: float, sample_rate: int) -> np.ndarray:
    """Add background noise at a given SNR."""
    if not noise_files:
        return audio

    noise_path = random.choice(noise_files)
    noise, _ = sf.read(str(noise_path), dtype="float32")
    if noise.ndim > 1:
        noise = noise[:, 0]

    # Loop or trim noise to match audio length
    while len(noise) < len(audio):
        noise = np.concatenate([noise, noise])
    start = random.randint(0, max(0, len(noise) - len(audio)))
    noise = noise[start : start + len(audio)]

    # Scale noise to target SNR
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    return audio + noise


def apply_rir(audio: np.ndarray, rir_files: list[Path]) -> np.ndarray:
    """Apply room impulse response convolution."""
    if not rir_files:
        return audio

    rir_path = random.choice(rir_files)
    rir, _ = sf.read(str(rir_path), dtype="float32")
    if rir.ndim > 1:
        rir = rir[:, 0]

    # Normalize RIR
    rir = rir / (np.max(np.abs(rir)) + 1e-10)

    # Convolve and trim to original length
    convolved = np.convolve(audio, rir)[:len(audio)]

    # Normalize to original peak
    peak = np.max(np.abs(audio))
    conv_peak = np.max(np.abs(convolved))
    if conv_peak > 1e-10:
        convolved = convolved * (peak / conv_peak)

    return convolved


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Change speed by resampling (no pitch correction)."""
    new_length = int(len(audio) / factor)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)


def change_volume(audio: np.ndarray, db: float) -> np.ndarray:
    """Change volume by a given dB amount."""
    return audio * (10 ** (db / 20))


def find_audio_files(directory: Path) -> list[Path]:
    """Find all WAV files in a directory recursively."""
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.wav"))


def create_audiomentations_pipeline(config: dict, noise_dir: Path, rir_dir: Path) -> "am.Compose":
    """Create an audiomentations augmentation pipeline."""
    aug_cfg = config["augmentation"]
    transforms = []

    noise_files = find_audio_files(noise_dir)
    if noise_files:
        transforms.append(
            am.AddBackgroundNoise(
                sounds_path=str(noise_dir),
                min_snr_in_db=aug_cfg["noise_snr_range"][0],
                max_snr_in_db=aug_cfg["noise_snr_range"][1],
                p=0.7,
            )
        )

    rir_files = find_audio_files(rir_dir)
    if rir_files:
        transforms.append(
            am.ApplyImpulseResponse(
                ir_path=str(rir_dir),
                p=aug_cfg["rir_probability"],
            )
        )

    transforms.extend([
        am.TimeStretch(
            min_rate=aug_cfg["speed_range"][0],
            max_rate=aug_cfg["speed_range"][1],
            p=0.5,
        ),
        am.Gain(
            min_gain_db=aug_cfg["volume_range"][0],
            max_gain_db=aug_cfg["volume_range"][1],
            p=0.5,
        ),
        am.PitchShift(
            min_semitones=aug_cfg["pitch_shift_range"][0],
            max_semitones=aug_cfg["pitch_shift_range"][1],
            p=0.3,
        ),
    ])

    return am.Compose(transforms)


def augment_manual(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
    noise_files: list[Path],
    rir_files: list[Path],
) -> np.ndarray:
    """Apply random augmentations without audiomentations library."""
    aug_cfg = config["augmentation"]
    result = audio.copy()

    # Random noise
    if noise_files and random.random() < 0.7:
        snr = random.uniform(*aug_cfg["noise_snr_range"])
        result = add_noise(result, noise_files, snr, sample_rate)

    # Random RIR
    if rir_files and random.random() < aug_cfg["rir_probability"]:
        result = apply_rir(result, rir_files)

    # Random speed
    if random.random() < 0.5:
        factor = random.uniform(*aug_cfg["speed_range"])
        result = change_speed(result, factor)

    # Random volume
    if random.random() < 0.5:
        db = random.uniform(*aug_cfg["volume_range"])
        result = change_volume(result, db)

    # Clip to prevent overflow
    result = np.clip(result, -1.0, 1.0)
    return result


def main():
    parser = argparse.ArgumentParser(description="Augment wake word samples")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml", help="Path to config file")
    parser.add_argument("--input-dir", default=None, help="Directory with source samples (default: all positive)")
    parser.add_argument("--multiplier", type=int, default=None, help="Augmented copies per sample")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    aug_cfg = config["augmentation"]
    audio_cfg = config["audio"]
    multiplier = args.multiplier or aug_cfg["num_augmented_per_sample"]

    output_dir = Path(args.output_dir or config["paths"]["augmented_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all positive samples
    source_dirs = []
    if args.input_dir:
        source_dirs.append(Path(args.input_dir))
    else:
        source_dirs.extend([
            Path(config["paths"]["positive_dir"]),
            Path(config["paths"]["synthetic_dir"]),
        ])

    source_files = []
    for d in source_dirs:
        source_files.extend(find_audio_files(d))

    if not source_files:
        print("Error: no source audio files found")
        print(f"  Searched: {[str(d) for d in source_dirs]}")
        return

    noise_dir = Path(config["paths"]["noise_dir"])
    rir_dir = Path(config["paths"]["rir_dir"])
    noise_files = find_audio_files(noise_dir)
    rir_files = find_audio_files(rir_dir)

    print(f"--- Audio Augmentation ---")
    print(f"  Source files:  {len(source_files)}")
    print(f"  Multiplier:    {multiplier}x")
    print(f"  Total output:  ~{len(source_files) * multiplier}")
    print(f"  Noise files:   {len(noise_files)}")
    print(f"  RIR files:     {len(rir_files)}")
    print(f"  Output:        {output_dir}")

    if not noise_files:
        print("  Warning: no noise files found — run download_negatives.py first for best results")
    if not rir_files:
        print("  Warning: no RIR files found — run download_negatives.py first for best results")

    # Setup pipeline
    pipeline = None
    if HAS_AUDIOMENTATIONS and (noise_files or rir_files):
        print("  Using audiomentations pipeline")
        try:
            pipeline = create_audiomentations_pipeline(config, noise_dir, rir_dir)
        except Exception as e:
            print(f"  audiomentations setup failed ({e}), using manual augmentation")
    else:
        if not HAS_AUDIOMENTATIONS:
            print("  audiomentations not installed — using manual augmentation")

    generated = 0
    for filepath in tqdm(source_files, desc="  Augmenting"):
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            continue

        stem = filepath.stem

        for j in range(multiplier):
            try:
                if pipeline is not None:
                    augmented = pipeline(samples=audio, sample_rate=audio_cfg["sample_rate"])
                else:
                    augmented = augment_manual(audio, audio_cfg["sample_rate"], config, noise_files, rir_files)

                augmented = np.clip(augmented, -1.0, 1.0)

                duration = len(augmented) / audio_cfg["sample_rate"]
                if duration < audio_cfg["min_duration_sec"] or duration > audio_cfg["max_duration_sec"] + 1.0:
                    continue

                out_path = output_dir / f"aug_{stem}_{j:02d}.wav"
                sf.write(str(out_path), augmented, audio_cfg["sample_rate"], subtype="PCM_16")
                generated += 1

            except Exception:
                continue

    print(f"\n  Generated {generated} augmented samples in {output_dir}")


if __name__ == "__main__":
    main()
