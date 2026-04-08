"""Download public negative datasets for wake word training.

Downloads background noise, music, and speech that does NOT contain the
wake phrase — used as negative training examples.

Run from IDE or CLI:
    python rnd/wakeword/dataset/download_negatives.py
    python rnd/wakeword/dataset/download_negatives.py --source musan
"""

import argparse
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import yaml
from tqdm import tqdm


def load_config(config_path: str = "rnd/wakeword/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


SOURCES = {
    "musan_noise": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "description": "MUSAN corpus — noise subset (ambient, technical noise)",
        "subdir": "musan/noise",
        "size_gb": 0.4,
    },
    "musan_music": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "description": "MUSAN corpus — music subset (various genres)",
        "subdir": "musan/music",
        "size_gb": 1.7,
    },
    "rir_noises": {
        "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "description": "Room Impulse Responses + point-source noises",
        "subdir": "RIRS_NOISES",
        "size_gb": 6.3,
    },
    "demand": {
        "url": None,  # Manual download required
        "description": "DEMAND noise database (15 environments: kitchen, office, etc.)",
        "subdir": "demand",
        "size_gb": 2.5,
    },
}


def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file with progress bar."""
    if output_path.exists():
        print(f"  Already downloaded: {output_path.name}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {description or url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {output_path.name}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path: Path, output_dir: Path):
    """Extract tar.gz or zip archive."""
    print(f"  Extracting {archive_path.name}...")

    if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(output_dir)
    else:
        print(f"  Unknown archive format: {archive_path.suffix}")


def convert_to_wav(input_dir: Path, output_dir: Path, target_sr: int, max_hours: float):
    """Convert audio files to 16kHz mono WAV, capping at max_hours."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".wav", ".flac", ".mp3", ".ogg", ".opus"}

    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    files = sorted(files)

    total_duration = 0.0
    max_seconds = max_hours * 3600
    converted = 0

    print(f"  Converting {len(files)} files (max {max_hours}h)...")
    for filepath in tqdm(files, desc="  Converting"):
        if total_duration >= max_seconds:
            break

        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Simple resample if needed
            if sr != target_sr:
                ratio = target_sr / sr
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)

            duration = len(audio) / target_sr

            # Split into 3-second chunks
            chunk_samples = 3 * target_sr
            for start in range(0, len(audio) - chunk_samples, chunk_samples):
                if total_duration >= max_seconds:
                    break
                chunk = audio[start : start + chunk_samples]
                out_path = output_dir / f"neg_{converted:06d}.wav"
                sf.write(str(out_path), chunk, target_sr, subtype="PCM_16")
                converted += 1
                total_duration += 3.0

        except Exception:
            continue

    hours = total_duration / 3600
    print(f"  Converted {converted} chunks ({hours:.1f} hours)")


def download_musan(config: dict, output_base: Path, subset: str):
    """Download and process MUSAN corpus."""
    source = SOURCES[f"musan_{subset}"]
    archive_name = "musan.tar.gz"
    archive_path = output_base / "downloads" / archive_name
    extract_dir = output_base / "downloads"

    musan_dir = extract_dir / source["subdir"]

    if not musan_dir.exists():
        download_file(source["url"], archive_path, source["description"])
        extract_archive(archive_path, extract_dir)

    if subset == "noise":
        out_dir = Path(config["paths"]["noise_dir"])
    else:
        out_dir = Path(config["paths"]["negative_dir"]) / f"musan_{subset}"

    max_hours = config["negatives"]["max_negative_hours"] / 3
    convert_to_wav(musan_dir, out_dir, config["audio"]["sample_rate"], max_hours)


def download_rir(config: dict, output_base: Path):
    """Download Room Impulse Responses."""
    source = SOURCES["rir_noises"]
    archive_name = "rirs_noises.zip"
    archive_path = output_base / "downloads" / archive_name
    extract_dir = output_base / "downloads"

    rir_dir = extract_dir / source["subdir"]

    if not rir_dir.exists():
        download_file(source["url"], archive_path, source["description"])
        extract_archive(archive_path, extract_dir)

    out_dir = Path(config["paths"]["rir_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy RIR wav files
    rir_wavs = list(rir_dir.rglob("*.wav"))
    print(f"  Found {len(rir_wavs)} impulse response files")

    copied = 0
    for wav in rir_wavs:
        if "impulse" not in str(wav).lower() and "rir" not in str(wav).lower():
            continue
        try:
            audio, sr = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != config["audio"]["sample_rate"]:
                ratio = config["audio"]["sample_rate"] / sr
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)
            out_path = out_dir / f"rir_{copied:04d}.wav"
            sf.write(str(out_path), audio, config["audio"]["sample_rate"], subtype="PCM_16")
            copied += 1
        except Exception:
            continue

    print(f"  Saved {copied} RIR files to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download negative datasets for wake word training")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml", help="Path to config file")
    parser.add_argument("--source", choices=["musan", "rir", "all"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    output_base = Path(config["paths"]["base_dir"])

    print("--- Negative Dataset Download ---")
    print(f"  Available sources:")
    for name, info in SOURCES.items():
        status = "manual" if info["url"] is None else f"~{info['size_gb']} GB"
        print(f"    {name}: {info['description']} ({status})")
    print()

    if args.source in ("musan", "all"):
        for subset in ("noise", "music"):
            if f"musan_{subset}" in config["negatives"]["ambient_sources"]:
                print(f"\n--- MUSAN {subset} ---")
                download_musan(config, output_base, subset)

    if args.source in ("rir", "all"):
        print(f"\n--- Room Impulse Responses ---")
        download_rir(config, output_base)

    print("\nDone! Negative data ready for augmentation.")


if __name__ == "__main__":
    main()
