"""Record negative samples (ambient noise, speech, adversarial phrases).

Run from IDE or CLI:
    python rnd/wakeword/dataset/record_negative.py
    python rnd/wakeword/dataset/record_negative.py --mode ambient --duration 120
    python rnd/wakeword/dataset/record_negative.py --mode adversarial --phrases "hey marcus,hey august"
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


def record_continuous(sample_rate: int, duration_sec: float) -> np.ndarray:
    """Record continuous audio for a fixed duration."""
    print(f"  Recording for {duration_sec:.0f} seconds...")
    samples = int(duration_sec * sample_rate)
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def record_clip(sample_rate: int, max_duration: float, silence_threshold: float, silence_duration: float) -> np.ndarray:
    """Record a single short clip, stopping on silence."""
    chunk_size = int(0.1 * sample_rate)
    frames = []
    max_samples = int(max_duration * sample_rate)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while True:
            data, _ = stream.read(chunk_size)
            frames.append(data[:, 0])
            audio = np.concatenate(frames)

            if len(audio) >= max_samples:
                break

            tail_samples = int(silence_duration * sample_rate)
            if len(audio) >= tail_samples:
                rms = np.sqrt(np.mean(audio[-tail_samples:] ** 2))
                if rms < silence_threshold:
                    break

    return np.concatenate(frames)


def split_into_chunks(audio: np.ndarray, sample_rate: int, chunk_sec: float = 3.0) -> list[np.ndarray]:
    """Split long recording into fixed-length chunks."""
    chunk_samples = int(chunk_sec * sample_rate)
    chunks = []
    for start in range(0, len(audio) - chunk_samples, chunk_samples):
        chunks.append(audio[start : start + chunk_samples])
    return chunks


def record_ambient(config: dict, output_dir: Path, duration_sec: float):
    """Record ambient room noise."""
    audio_cfg = config["audio"]
    print(f"\n--- Ambient Recording ({duration_sec}s) ---")
    print("  Keep the room in its natural state (AC, fridge, etc.)")
    print("  Do NOT speak during this recording.")
    input("  Press Enter to start...")

    audio = record_continuous(audio_cfg["sample_rate"], duration_sec)
    chunks = split_into_chunks(audio, audio_cfg["sample_rate"])

    for i, chunk in enumerate(chunks):
        filepath = output_dir / f"ambient_{i + 1:04d}.wav"
        sf.write(str(filepath), chunk, audio_cfg["sample_rate"], subtype="PCM_16")

    print(f"  Saved {len(chunks)} ambient chunks")


def record_conversation(config: dict, output_dir: Path, duration_sec: float):
    """Record normal speech that does NOT contain the wake word."""
    audio_cfg = config["audio"]
    print(f"\n--- Conversation Recording ({duration_sec}s) ---")
    print(f"  Speak normally but do NOT say \"{config['wake_phrase']}\"")
    print("  Talk about anything — weather, plans, stories, etc.")
    input("  Press Enter to start...")

    audio = record_continuous(audio_cfg["sample_rate"], duration_sec)
    chunks = split_into_chunks(audio, audio_cfg["sample_rate"])

    for i, chunk in enumerate(chunks):
        filepath = output_dir / f"conversation_{i + 1:04d}.wav"
        sf.write(str(filepath), chunk, audio_cfg["sample_rate"], subtype="PCM_16")

    print(f"  Saved {len(chunks)} conversation chunks")


def record_adversarial(config: dict, output_dir: Path, phrases: list[str], count_per_phrase: int):
    """Record adversarial phrases (phonetically similar to wake word)."""
    audio_cfg = config["audio"]
    rec_cfg = config["recording"]

    print(f"\n--- Adversarial Phrase Recording ---")
    print(f"  These phrases sound similar to \"{config['wake_phrase']}\" but aren't it.")
    print(f"  Phrases: {', '.join(phrases)}")
    print(f"  {count_per_phrase} recordings per phrase")
    input("  Press Enter to start...")

    total = 0
    for phrase in phrases:
        print(f"\n  Phrase: \"{phrase}\"")
        for j in range(count_per_phrase):
            print(f"    [{j + 1}/{count_per_phrase}] Say: \"{phrase}\"")
            time.sleep(rec_cfg["countdown_sec"])
            print("    Recording...")

            audio = record_clip(
                audio_cfg["sample_rate"],
                audio_cfg["max_duration_sec"],
                rec_cfg["silence_threshold"],
                rec_cfg["silence_duration"],
            )

            safe_phrase = phrase.replace(" ", "_").lower()
            filepath = output_dir / f"adversarial_{safe_phrase}_{j + 1:03d}.wav"
            sf.write(str(filepath), audio, audio_cfg["sample_rate"], subtype="PCM_16")
            total += 1

            if j < count_per_phrase - 1:
                time.sleep(rec_cfg["pause_between_sec"])

    print(f"\n  Saved {total} adversarial clips")


DEFAULT_ADVERSARIAL_PHRASES = [
    "hey marcus",
    "hey august",
    "hey are us",
    "hey gorgeous",
    "the argus",
    "hey artists",
    "hey",
    "argus",
]


def main():
    parser = argparse.ArgumentParser(description="Record negative samples")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["ambient", "conversation", "adversarial", "all"], default="all")
    parser.add_argument("--duration", type=float, default=120, help="Duration in seconds (ambient/conversation)")
    parser.add_argument("--phrases", default=None, help="Comma-separated adversarial phrases")
    parser.add_argument("--count-per-phrase", type=int, default=10, help="Recordings per adversarial phrase")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["paths"]["negative_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    phrases = args.phrases.split(",") if args.phrases else DEFAULT_ADVERSARIAL_PHRASES

    print(f"--- Negative Sample Recording ---")
    print(f"  Wake phrase: \"{config['wake_phrase']}\"")
    print(f"  Output:      {output_dir}")
    print(f"  Mode:        {args.mode}")

    if args.mode in ("ambient", "all"):
        record_ambient(config, output_dir, args.duration)

    if args.mode in ("conversation", "all"):
        record_conversation(config, output_dir, args.duration)

    if args.mode in ("adversarial", "all"):
        record_adversarial(config, output_dir, phrases, args.count_per_phrase)

    print(f"\nAll done! Files saved in {output_dir}")


if __name__ == "__main__":
    main()
