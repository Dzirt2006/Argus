"""Validate the wake word dataset: format, duration, quality, balance.

Checks all audio files for correct format, detects issues (clipping, silence,
wrong sample rate), and reports dataset statistics.

Run from IDE or CLI:
    python rnd/wakeword/dataset/validate_dataset.py
    python rnd/wakeword/dataset/validate_dataset.py --fix
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def load_config(config_path: str = "rnd/wakeword/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class DatasetValidator:
    def __init__(self, config: dict):
        self.config = config
        self.audio_cfg = config["audio"]
        self.issues: list[dict] = []
        self.stats: dict = defaultdict(lambda: {"count": 0, "total_duration": 0.0, "issues": 0})

    def validate_file(self, filepath: Path, category: str) -> bool:
        """Validate a single audio file. Returns True if valid."""
        try:
            info = sf.info(str(filepath))
        except Exception as e:
            self.issues.append({"file": str(filepath), "issue": f"Cannot read: {e}", "severity": "error"})
            self.stats[category]["issues"] += 1
            return False

        valid = True

        # Check sample rate
        if info.samplerate != self.audio_cfg["sample_rate"]:
            self.issues.append({
                "file": str(filepath),
                "issue": f"Wrong sample rate: {info.samplerate}Hz (expected {self.audio_cfg['sample_rate']}Hz)",
                "severity": "error",
            })
            valid = False

        # Check channels
        if info.channels != self.audio_cfg["channels"]:
            self.issues.append({
                "file": str(filepath),
                "issue": f"Wrong channels: {info.channels} (expected {self.audio_cfg['channels']})",
                "severity": "error",
            })
            valid = False

        # Check duration
        duration = info.duration
        if duration < self.audio_cfg["min_duration_sec"]:
            self.issues.append({
                "file": str(filepath),
                "issue": f"Too short: {duration:.2f}s (min {self.audio_cfg['min_duration_sec']}s)",
                "severity": "warning",
            })
            valid = False

        if duration > self.audio_cfg["max_duration_sec"] + 1.0:
            self.issues.append({
                "file": str(filepath),
                "issue": f"Too long: {duration:.2f}s (max {self.audio_cfg['max_duration_sec']}s)",
                "severity": "warning",
            })

        # Load audio for content checks
        try:
            audio, _ = sf.read(str(filepath), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
        except Exception:
            return valid

        # Check for clipping
        peak = np.max(np.abs(audio))
        if peak > 0.99:
            self.issues.append({
                "file": str(filepath),
                "issue": f"Clipped audio (peak: {peak:.4f})",
                "severity": "warning",
            })

        # Check for silence
        peak_db = 20 * np.log10(peak + 1e-10)
        if peak_db < -40:
            self.issues.append({
                "file": str(filepath),
                "issue": f"Silent audio (peak: {peak_db:.1f} dB)",
                "severity": "error",
            })
            valid = False

        # Check for DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.01:
            self.issues.append({
                "file": str(filepath),
                "issue": f"DC offset: {dc_offset:.4f}",
                "severity": "warning",
            })

        self.stats[category]["count"] += 1
        self.stats[category]["total_duration"] += duration
        if not valid:
            self.stats[category]["issues"] += 1

        return valid

    def validate_directory(self, directory: Path, category: str) -> int:
        """Validate all WAV files in a directory. Returns count of valid files."""
        if not directory.exists():
            return 0

        files = sorted(directory.rglob("*.wav"))
        valid_count = 0
        for f in files:
            if self.validate_file(f, category):
                valid_count += 1

        return valid_count

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 70)
        print("DATASET VALIDATION REPORT")
        print("=" * 70)

        # Stats per category
        print("\n--- Dataset Statistics ---\n")
        print(f"  {'Category':<20} {'Files':>8} {'Duration':>12} {'Issues':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*8}")

        total_files = 0
        total_duration = 0.0
        total_issues = 0

        for category, stats in sorted(self.stats.items()):
            duration_str = f"{stats['total_duration']:.1f}s"
            if stats["total_duration"] > 3600:
                duration_str = f"{stats['total_duration']/3600:.1f}h"
            elif stats["total_duration"] > 60:
                duration_str = f"{stats['total_duration']/60:.1f}m"

            print(f"  {category:<20} {stats['count']:>8} {duration_str:>12} {stats['issues']:>8}")
            total_files += stats["count"]
            total_duration += stats["total_duration"]
            total_issues += stats["issues"]

        print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*8}")
        dur_str = f"{total_duration:.1f}s"
        if total_duration > 3600:
            dur_str = f"{total_duration/3600:.1f}h"
        elif total_duration > 60:
            dur_str = f"{total_duration/60:.1f}m"
        print(f"  {'TOTAL':<20} {total_files:>8} {dur_str:>12} {total_issues:>8}")

        # Issues
        if self.issues:
            errors = [i for i in self.issues if i["severity"] == "error"]
            warnings = [i for i in self.issues if i["severity"] == "warning"]

            if errors:
                print(f"\n--- Errors ({len(errors)}) ---\n")
                for issue in errors[:20]:
                    print(f"  ERROR: {Path(issue['file']).name}: {issue['issue']}")
                if len(errors) > 20:
                    print(f"  ... and {len(errors) - 20} more errors")

            if warnings:
                print(f"\n--- Warnings ({len(warnings)}) ---\n")
                for issue in warnings[:20]:
                    print(f"  WARN:  {Path(issue['file']).name}: {issue['issue']}")
                if len(warnings) > 20:
                    print(f"  ... and {len(warnings) - 20} more warnings")
        else:
            print("\n  No issues found!")

        # Quality checklist
        print("\n--- Quality Checklist ---\n")
        positive_count = self.stats.get("positive", {}).get("count", 0)
        synthetic_count = self.stats.get("synthetic", {}).get("count", 0)
        negative_count = self.stats.get("negative", {}).get("count", 0)
        augmented_count = self.stats.get("augmented", {}).get("count", 0)

        checks = [
            (positive_count >= 40, f"Real positive recordings: {positive_count} (min 40)"),
            (synthetic_count >= 3000, f"Synthetic samples: {synthetic_count} (min 3,000)"),
            (negative_count >= 100, f"Negative samples: {negative_count} (min 100)"),
            (augmented_count > 0, f"Augmented samples: {augmented_count}"),
            (total_issues == 0, f"No errors: {total_issues} issues"),
        ]

        for passed, label in checks:
            marker = "[x]" if passed else "[ ]"
            print(f"  {marker} {label}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Validate wake word dataset")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    validator = DatasetValidator(config)

    print("--- Validating Wake Word Dataset ---")

    dirs_to_check = [
        (config["paths"]["positive_dir"], "positive"),
        (config["paths"]["synthetic_dir"], "synthetic"),
        (config["paths"]["negative_dir"], "negative"),
        (config["paths"]["augmented_dir"], "augmented"),
    ]

    for dir_path, category in dirs_to_check:
        path = Path(dir_path)
        print(f"  Checking {category} ({path})...")
        validator.validate_directory(path, category)

    validator.print_report()


if __name__ == "__main__":
    main()
