"""Extract audio embeddings using OpenWakeWord's frozen feature models.

Converts raw audio WAV files into embedding vectors that the classifier
trains on. Uses the same melspectrogram + embedding pipeline as OWW inference.

Run from IDE or CLI:
    python rnd/wakeword/training/extract_features.py
    python rnd/wakeword/training/extract_features.py --input-dir data/positive/ --label positive
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def load_config(
    global_config_path: str = "rnd/wakeword/config.yaml",
    training_config_path: str = "rnd/wakeword/training/config.yaml",
) -> dict:
    with open(global_config_path) as f:
        config = yaml.safe_load(f)
    with open(training_config_path) as f:
        config["model_config"] = yaml.safe_load(f)
    return config


def find_oww_models() -> tuple[Path | None, Path | None]:
    """Find OpenWakeWord's melspectrogram and embedding ONNX models."""
    try:
        import openwakeword

        oww_dir = Path(openwakeword.__file__).parent / "resources"
        mel_model = oww_dir / "models" / "melspectrogram.onnx"
        emb_model = oww_dir / "models" / "embedding_model.onnx"

        if mel_model.exists() and emb_model.exists():
            return mel_model, emb_model
    except ImportError:
        pass

    # Fallback: search common locations
    search_dirs = [
        Path.home() / ".cache/openwakeword",
        Path("/usr/share/openwakeword"),
    ]
    for base in search_dirs:
        mel = base / "melspectrogram.onnx"
        emb = base / "embedding_model.onnx"
        if mel.exists() and emb.exists():
            return mel, emb

    return None, None


class FeatureExtractor:
    """Extract embeddings from audio using OWW's frozen models."""

    def __init__(self, mel_model_path: Path, emb_model_path: Path):
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

        self.mel_session = ort.InferenceSession(
            str(mel_model_path),
            providers=["CPUExecutionProvider"],
        )
        self.emb_session = ort.InferenceSession(
            str(emb_model_path),
            providers=["CPUExecutionProvider"],
        )

        # Get model input/output info
        self.mel_input_name = self.mel_session.get_inputs()[0].name
        self.mel_output_name = self.mel_session.get_outputs()[0].name
        self.emb_input_name = self.emb_session.get_inputs()[0].name
        self.emb_output_name = self.emb_session.get_outputs()[0].name

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """Extract embedding features from audio.

        Returns array of shape (num_frames, embedding_dim) or None on error.
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # OWW expects 16kHz mono
        if len(audio) < sample_rate * 0.3:  # minimum 300ms
            return None

        # Pad to at least 1280 samples (80ms at 16kHz) for one frame
        min_samples = 1280
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        # Step 1: Mel spectrogram
        # OWW mel model expects shape (1, samples)
        mel_input = audio.reshape(1, -1)
        try:
            mel_output = self.mel_session.run(
                [self.mel_output_name],
                {self.mel_input_name: mel_input},
            )[0]
        except Exception:
            return None

        # Step 2: Embedding model
        # The embedding model processes mel frames
        try:
            embeddings = self.emb_session.run(
                [self.emb_output_name],
                {self.emb_input_name: mel_output},
            )[0]
        except Exception:
            return None

        return embeddings.squeeze()


def process_directory(
    extractor: FeatureExtractor,
    input_dir: Path,
    label: int,
    sample_rate: int,
    num_frames: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Process all WAV files in a directory, return features and labels."""
    features = []
    labels = []

    files = sorted(input_dir.rglob("*.wav"))
    if not files:
        return features, labels

    for filepath in tqdm(files, desc=f"  {input_dir.name}"):
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
        except Exception:
            continue

        embedding = extractor.extract(audio, sample_rate)
        if embedding is None:
            continue

        # If embedding is 1D, treat as single frame
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Use the last num_frames frames (or pad if shorter)
        if embedding.shape[0] >= num_frames:
            feature = embedding[-num_frames:]
        else:
            pad_width = num_frames - embedding.shape[0]
            feature = np.pad(embedding, ((pad_width, 0), (0, 0)))

        # Flatten to 1D feature vector
        features.append(feature.flatten())
        labels.append(label)

    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Extract features from audio using OWW embedding models")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml")
    parser.add_argument("--training-config", default="rnd/wakeword/training/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config, args.training_config)
    emb_cfg = config["model_config"]["embeddings"]
    sample_rate = config["audio"]["sample_rate"]
    num_frames = emb_cfg["num_frames"]

    # Find OWW models
    mel_path, emb_path = find_oww_models()
    if mel_path is None:
        print("Error: OpenWakeWord embedding models not found.")
        print("Install openwakeword: pip install openwakeword")
        print("Then run: python -c \"import openwakeword; openwakeword.utils.download_models()\"")
        return

    print(f"--- Feature Extraction ---")
    print(f"  Mel model:       {mel_path}")
    print(f"  Embedding model: {emb_path}")
    print(f"  Num frames:      {num_frames}")

    extractor = FeatureExtractor(mel_path, emb_path)

    all_features = []
    all_labels = []
    all_weights = []

    weight_cfg = config["model_config"]["weights"]

    # Process positive samples (real recordings — weight 3x)
    positive_dir = Path(config["paths"]["positive_dir"])
    if positive_dir.exists():
        print(f"\n  Processing real positives ({positive_dir})...")
        feats, labs = process_directory(extractor, positive_dir, 1, sample_rate, num_frames)
        all_features.extend(feats)
        all_labels.extend(labs)
        all_weights.extend([weight_cfg["real_positive"]] * len(feats))
        print(f"    {len(feats)} samples")

    # Process synthetic samples (weight 1x)
    synthetic_dir = Path(config["paths"]["synthetic_dir"])
    if synthetic_dir.exists():
        print(f"\n  Processing synthetic positives ({synthetic_dir})...")
        feats, labs = process_directory(extractor, synthetic_dir, 1, sample_rate, num_frames)
        all_features.extend(feats)
        all_labels.extend(labs)
        all_weights.extend([weight_cfg["synthetic_positive"]] * len(feats))
        print(f"    {len(feats)} samples")

    # Process augmented samples (weight 1x)
    augmented_dir = Path(config["paths"]["augmented_dir"])
    if augmented_dir.exists():
        print(f"\n  Processing augmented positives ({augmented_dir})...")
        feats, labs = process_directory(extractor, augmented_dir, 1, sample_rate, num_frames)
        all_features.extend(feats)
        all_labels.extend(labs)
        all_weights.extend([weight_cfg["synthetic_positive"]] * len(feats))
        print(f"    {len(feats)} samples")

    # Process negative samples
    negative_dir = Path(config["paths"]["negative_dir"])
    if negative_dir.exists():
        print(f"\n  Processing negatives ({negative_dir})...")
        feats, labs = process_directory(extractor, negative_dir, 0, sample_rate, num_frames)
        all_features.extend(feats)
        all_labels.extend(labs)
        all_weights.extend([weight_cfg["negative"]] * len(feats))
        print(f"    {len(feats)} samples")

    if not all_features:
        print("\nError: no features extracted. Check your dataset directories.")
        return

    # Save features
    features_dir = Path(config["paths"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)
    weights_array = np.array(all_weights, dtype=np.float32)

    np.save(features_dir / "features.npy", features_array)
    np.save(features_dir / "labels.npy", labels_array)
    np.save(features_dir / "weights.npy", weights_array)

    positive_count = int(np.sum(labels_array == 1))
    negative_count = int(np.sum(labels_array == 0))

    print(f"\n--- Summary ---")
    print(f"  Feature shape: {features_array.shape}")
    print(f"  Positive:      {positive_count}")
    print(f"  Negative:      {negative_count}")
    print(f"  Ratio:         1:{negative_count / max(positive_count, 1):.1f}")
    print(f"  Saved to:      {features_dir}")


if __name__ == "__main__":
    main()
