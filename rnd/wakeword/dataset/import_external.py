"""Import and normalize audio files recorded on external devices.

Handles resampling, mono conversion, silence trimming, volume normalization,
and format conversion from any common audio format to 16kHz mono WAV.

Run from IDE or CLI:
    python rnd/wakeword/dataset/import_external.py
    python rnd/wakeword/dataset/import_external.py --input-dir data/raw_phone/ --speaker alex
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

try:
    import librosa
except ImportError:
    librosa = None


def load_config(config_path: str = "rnd/wakeword/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aac", ".webm"}


def load_audio(filepath: Path, target_sr: int) -> np.ndarray | None:
    """Load audio file and resample to target sample rate."""
    if librosa is None:
        # Fallback to soundfile (WAV/FLAC only)
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            print(f"    Error reading {filepath.name}: {e}")
            return None
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            print(f"    Warning: {filepath.name} is {sr}Hz, install librosa for resampling")
            return None
        return audio

    try:
        audio, _ = librosa.load(str(filepath), sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"    Error reading {filepath.name}: {e}")
        return None


def trim_silence(audio: np.ndarray, threshold_db: float, sample_rate: int, pad_ms: int = 50) -> np.ndarray:
    """Trim leading and trailing silence based on dB threshold."""
    threshold_linear = 10 ** (threshold_db / 20)
    abs_audio = np.abs(audio)
    above = abs_audio > threshold_linear

    if not np.any(above):
        return audio

    first = np.argmax(above)
    last = len(above) - np.argmax(above[::-1])

    pad_samples = int(pad_ms / 1000 * sample_rate)
    start = max(0, first - pad_samples)
    end = min(len(audio), last + pad_samples)
    return audio[start:end]


def normalize_volume(audio: np.ndarray, target_peak_db: float) -> np.ndarray:
    """Normalize audio to target peak level in dB."""
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return audio
    target_linear = 10 ** (target_peak_db / 20)
    return audio * (target_linear / peak)


def process_file(
    filepath: Path,
    output_dir: Path,
    config: dict,
    index: int,
) -> bool:
    """Process a single audio file: load, normalize, save."""
    import_cfg = config["import"]
    audio_cfg = config["audio"]

    audio = load_audio(filepath, import_cfg["target_sample_rate"])
    if audio is None:
        return False

    if import_cfg["trim_silence"]:
        audio = trim_silence(audio, import_cfg["trim_threshold_db"], import_cfg["target_sample_rate"])

    if import_cfg["normalize_volume"]:
        audio = normalize_volume(audio, import_cfg["target_peak_db"])

    duration = len(audio) / import_cfg["target_sample_rate"]
    if duration < audio_cfg["min_duration_sec"]:
        print(f"    Skipping {filepath.name}: too short ({duration:.2f}s)")
        return False
    if duration > audio_cfg["max_duration_sec"]:
        max_samples = int(audio_cfg["max_duration_sec"] * import_cfg["target_sample_rate"])
        audio = audio[:max_samples]
        duration = audio_cfg["max_duration_sec"]

    output_path = output_dir / f"imported_{index:04d}.wav"
    sf.write(str(output_path), audio, import_cfg["target_sample_rate"], subtype="PCM_16")

    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    print(f"    {filepath.name} -> {output_path.name} ({duration:.2f}s, peak {peak_db:.1f} dB)")
    return True


def find_audio_files(input_dir: Path) -> list[Path]:
    """Recursively find all supported audio files."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Import and normalize external audio recordings")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml", help="Path to config file")
    parser.add_argument("--input-dir", default=None, help="Directory with raw recordings (default: data/external/)")
    parser.add_argument("--speaker", default=None, help="Speaker name (auto-detected from folder if not given)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--type", choices=["positive", "negative"], default="positive", help="Sample type")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = Path(args.input_dir or config["paths"]["external_dir"])

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        print(f"Create it and place your recordings there, organized by speaker:")
        print(f"  {input_dir}/alex/hey_argus_01.wav")
        print(f"  {input_dir}/maria/hey_argus_01.m4a")
        return

    # Find speaker subdirectories or process flat directory
    speaker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if speaker_dirs:
        print(f"Found {len(speaker_dirs)} speaker folders: {[d.name for d in speaker_dirs]}")
        for speaker_dir in speaker_dirs:
            speaker = speaker_dir.name
            files = find_audio_files(speaker_dir)
            if not files:
                print(f"\n  {speaker}: no audio files found")
                continue

            if args.type == "positive":
                output_dir = Path(config["paths"]["positive_dir"]) / speaker
            else:
                output_dir = Path(config["paths"]["negative_dir"])
            output_dir = Path(args.output_dir) if args.output_dir else output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            existing = list(output_dir.glob("imported_*.wav"))
            start_idx = len(existing) + 1

            print(f"\n  {speaker}: {len(files)} files -> {output_dir}")
            imported = 0
            for i, filepath in enumerate(files):
                if process_file(filepath, output_dir, config, start_idx + i):
                    imported += 1
            print(f"  {speaker}: imported {imported}/{len(files)} files")
    else:
        # Flat directory, use --speaker name
        speaker = args.speaker or input("Enter speaker name: ").strip()
        files = find_audio_files(input_dir)
        if not files:
            print(f"No audio files found in {input_dir}")
            return

        if args.type == "positive":
            output_dir = Path(args.output_dir or config["paths"]["positive_dir"]) / speaker
        else:
            output_dir = Path(args.output_dir or config["paths"]["negative_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        existing = list(output_dir.glob("imported_*.wav"))
        start_idx = len(existing) + 1

        print(f"\n  Importing {len(files)} files for speaker '{speaker}' -> {output_dir}")
        imported = 0
        for i, filepath in enumerate(files):
            if process_file(filepath, output_dir, config, start_idx + i):
                imported += 1
        print(f"\n  Imported {imported}/{len(files)} files")

    print("\nDone! Run validate_dataset.py to verify the imported files.")


if __name__ == "__main__":
    main()
