"""Record positive wake word samples from microphone.

Run from IDE or CLI:
    python rnd/wakeword/dataset/record_positive.py
    python rnd/wakeword/dataset/record_positive.py --speaker alex --count 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml


def load_config(config_path: str = "rnd/wakeword/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_silence(audio: np.ndarray, threshold: float, duration: float, sample_rate: int) -> bool:
    """Check if the tail of the audio is silence."""
    tail_samples = int(duration * sample_rate)
    if len(audio) < tail_samples:
        return False
    tail = audio[-tail_samples:]
    rms = np.sqrt(np.mean(tail ** 2))
    return rms < threshold


def record_sample(
    sample_rate: int,
    max_duration: float,
    silence_threshold: float,
    silence_duration: float,
) -> np.ndarray:
    """Record a single sample, stopping on silence or max duration."""
    max_samples = int(max_duration * sample_rate)
    chunk_size = int(0.1 * sample_rate)  # 100ms chunks
    frames = []

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while True:
            data, _ = stream.read(chunk_size)
            frames.append(data[:, 0])
            audio = np.concatenate(frames)

            if len(audio) >= max_samples:
                break
            if detect_silence(audio, silence_threshold, silence_duration, sample_rate):
                break

    return np.concatenate(frames)


def trim_silence(audio: np.ndarray, threshold: float, sample_rate: int, pad_ms: int = 50) -> np.ndarray:
    """Trim leading and trailing silence, keep a small pad."""
    abs_audio = np.abs(audio)
    above = abs_audio > threshold
    if not np.any(above):
        return audio

    first = np.argmax(above)
    last = len(above) - np.argmax(above[::-1])

    pad_samples = int(pad_ms / 1000 * sample_rate)
    start = max(0, first - pad_samples)
    end = min(len(audio), last + pad_samples)
    return audio[start:end]


def countdown(seconds: int):
    for i in range(seconds, 0, -1):
        print(f"  Recording in {i}...", end="\r")
        time.sleep(1)
    print("  🎤 Recording NOW — say the wake phrase!   ")


def main():
    parser = argparse.ArgumentParser(description="Record positive wake word samples")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml", help="Path to config file")
    parser.add_argument("--speaker", default=None, help="Speaker name (prompted if not given)")
    parser.add_argument("--count", type=int, default=None, help="Number of samples to record")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    audio_cfg = config["audio"]
    rec_cfg = config["recording"]

    speaker = args.speaker or input("Enter speaker name: ").strip()
    if not speaker:
        print("Error: speaker name is required")
        return

    count = args.count or int(input("How many samples to record? [50]: ").strip() or "50")

    output_dir = Path(args.output_dir or config["paths"]["positive_dir"]) / speaker
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.wav"))
    start_idx = len(existing) + 1

    print(f"\n--- Positive Wake Word Recording ---")
    print(f"  Phrase:   '{config['wake_phrase']}'")
    print(f"  Speaker:  {speaker}")
    print(f"  Samples:  {count}")
    print(f"  Save to:  {output_dir}")
    print(f"  Existing: {len(existing)} files")
    print()
    print("Tips:")
    print("  - Vary volume, speed, distance, and tone between recordings")
    print("  - Speak naturally, as if calling out to someone")
    print("  - Press Ctrl+C to stop early")
    print()
    input("Press Enter to start...")

    recorded = 0
    try:
        for i in range(count):
            print(f"\n[{i + 1}/{count}] Say: \"{config['wake_phrase']}\"")
            countdown(rec_cfg["countdown_sec"])

            audio = record_sample(
                sample_rate=audio_cfg["sample_rate"],
                max_duration=audio_cfg["max_duration_sec"],
                silence_threshold=rec_cfg["silence_threshold"],
                silence_duration=rec_cfg["silence_duration"],
            )

            audio = trim_silence(audio, rec_cfg["silence_threshold"], audio_cfg["sample_rate"])

            duration = len(audio) / audio_cfg["sample_rate"]
            if duration < audio_cfg["min_duration_sec"]:
                print(f"  Too short ({duration:.2f}s) — skipping")
                continue

            peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
            if peak_db < -40:
                print(f"  Too quiet (peak {peak_db:.1f} dB) — skipping")
                continue

            filename = f"positive_{start_idx + recorded:04d}.wav"
            filepath = output_dir / filename
            sf.write(str(filepath), audio, audio_cfg["sample_rate"], subtype="PCM_16")

            print(f"  Saved: {filename} ({duration:.2f}s, peak {peak_db:.1f} dB)")
            recorded += 1

            if i < count - 1:
                time.sleep(rec_cfg["pause_between_sec"])

    except KeyboardInterrupt:
        print("\n\nStopped early.")

    print(f"\nDone! Recorded {recorded} samples in {output_dir}")


if __name__ == "__main__":
    main()
