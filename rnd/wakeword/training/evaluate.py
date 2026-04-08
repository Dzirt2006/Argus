"""Evaluate a trained wake word model with detailed metrics.

Runs the model on test data and reports accuracy, false accept/reject rates,
per-speaker breakdown, and ROC curve data.

Run from IDE or CLI:
    python rnd/wakeword/training/evaluate.py
    python rnd/wakeword/training/evaluate.py --model models/best_model.pt --threshold 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_config(
    global_config_path: str = "rnd/wakeword/config.yaml",
    training_config_path: str = "rnd/wakeword/training/config.yaml",
) -> dict:
    with open(global_config_path) as f:
        config = yaml.safe_load(f)
    with open(training_config_path) as f:
        config["model_config"] = yaml.safe_load(f)
    return config


def load_model(model_path: Path, device: "torch.device"):
    """Load trained model from checkpoint."""
    from train import WakeWordDNN, WakeWordRNN

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    input_dim = checkpoint["input_dim"]

    if model_cfg["type"] == "rnn":
        model = WakeWordRNN(
            frame_dim=96,  # default OWW embedding dim
            num_frames=input_dim // 96,
            hidden_dim=checkpoint.get("hidden_dim", model_cfg["hidden_dim"]),
            num_layers=model_cfg["num_layers"],
            dropout=0.0,  # no dropout at inference
        )
    else:
        model = WakeWordDNN(
            input_dim=input_dim,
            hidden_dim=checkpoint.get("hidden_dim", model_cfg["hidden_dim"]),
            num_layers=model_cfg["num_layers"],
            dropout=0.0,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def compute_metrics_at_threshold(predictions: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred_binary = (predictions >= threshold).astype(int)

    tp = int(np.sum((pred_binary == 1) & (labels == 1)))
    fp = int(np.sum((pred_binary == 1) & (labels == 0)))
    tn = int(np.sum((pred_binary == 0) & (labels == 0)))
    fn = int(np.sum((pred_binary == 0) & (labels == 1)))

    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    far = fp / (fp + tn + 1e-10)
    frr = fn / (fn + tp + 1e-10)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_accept_rate": float(far),
        "false_reject_rate": float(frr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def find_optimal_threshold(predictions: np.ndarray, labels: np.ndarray, target_far: float) -> float:
    """Find the threshold that achieves the target false accept rate."""
    best_threshold = 0.5
    best_diff = float("inf")

    for threshold in np.arange(0.01, 1.0, 0.01):
        metrics = compute_metrics_at_threshold(predictions, labels, threshold)
        diff = abs(metrics["false_accept_rate"] - target_far)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

    return best_threshold


def plot_roc_curve(predictions: np.ndarray, labels: np.ndarray, output_path: Path):
    """Generate and save ROC curve."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib not installed — skipping ROC plot")
        return

    fars = []
    frrs = []

    for threshold in np.arange(0.01, 1.0, 0.01):
        m = compute_metrics_at_threshold(predictions, labels, threshold)
        fars.append(m["false_accept_rate"])
        frrs.append(m["false_reject_rate"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve (FAR vs 1-FRR)
    ax1.plot(fars, [1 - f for f in frrs], "b-", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Accept Rate")
    ax1.set_ylabel("True Accept Rate (1 - FRR)")
    ax1.set_title("ROC Curve")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.2])
    ax1.set_ylim([0.8, 1.0])

    # FAR/FRR vs threshold
    thresholds = np.arange(0.01, 1.0, 0.01)
    ax2.plot(thresholds, fars, "r-", label="FAR", linewidth=2)
    ax2.plot(thresholds, frrs, "b-", label="FRR", linewidth=2)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Rate")
    ax2.set_title("FAR/FRR vs Threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved: {output_path}")


def plot_prediction_distribution(predictions: np.ndarray, labels: np.ndarray, output_path: Path):
    """Plot histogram of prediction scores for positive and negative samples."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    pos_preds = predictions[labels == 1]
    neg_preds = predictions[labels == 0]

    ax.hist(neg_preds, bins=50, alpha=0.6, color="red", label=f"Negative (n={len(neg_preds)})")
    ax.hist(pos_preds, bins=50, alpha=0.6, color="green", label=f"Positive (n={len(pos_preds)})")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Default threshold")
    ax.set_xlabel("Prediction Score")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Distribution plot saved: {output_path}")


@torch.no_grad()
def get_predictions(model: nn.Module, features: np.ndarray, device: "torch.device", batch_size: int = 512):
    """Get model predictions for all features."""
    dataset = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    for (batch,) in loader:
        outputs = model(batch.to(device))
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)

    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained wake word model")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml")
    parser.add_argument("--training-config", default="rnd/wakeword/training/config.yaml")
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if torch is None:
        print("Error: PyTorch not installed.")
        return

    config = load_config(args.config, args.training_config)
    eval_cfg = config["model_config"]["evaluation"]
    models_dir = Path(config["paths"]["models_dir"])
    features_dir = Path(config["paths"]["features_dir"])

    model_path = Path(args.model) if args.model else models_dir / "best_model.pt"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load data
    try:
        features = np.load(features_dir / "features.npy")
        labels = np.load(features_dir / "labels.npy")
    except FileNotFoundError:
        print("Error: features not found. Run extract_features.py first.")
        return

    # Use test split only
    data_cfg = config["model_config"]["data"]
    rng = np.random.RandomState(data_cfg["seed"])
    indices = rng.permutation(len(features))
    n_test = int(len(features) * data_cfg["test_split"])
    test_idx = indices[:n_test]

    test_features = features[test_idx]
    test_labels = labels[test_idx]

    print(f"--- Model Evaluation ---")
    print(f"  Model:     {model_path}")
    print(f"  Test set:  {len(test_features)} samples")
    print(f"  Positive:  {np.sum(test_labels == 1)}")
    print(f"  Negative:  {np.sum(test_labels == 0)}")
    print(f"  Device:    {device}")

    # Load model and get predictions
    model = load_model(model_path, device)
    predictions = get_predictions(model, test_features, device)

    # Metrics at all thresholds
    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FAR':>10} {'FRR':>10}")
    print(f"{'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    all_metrics = []
    for threshold in eval_cfg["thresholds"]:
        m = compute_metrics_at_threshold(predictions, test_labels, threshold)
        all_metrics.append(m)
        marker = " <--" if abs(threshold - 0.5) < 0.01 else ""
        print(
            f"{threshold:>10.1f} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['false_accept_rate']:>10.4f} "
            f"{m['false_reject_rate']:>10.4f}{marker}"
        )

    # Find optimal threshold
    optimal = find_optimal_threshold(predictions, test_labels, eval_cfg["target_false_accept_rate"])
    opt_metrics = compute_metrics_at_threshold(predictions, test_labels, optimal)

    print(f"\n--- Optimal Threshold ---")
    print(f"  Target FAR:     {eval_cfg['target_false_accept_rate']}")
    print(f"  Optimal thresh: {optimal:.2f}")
    print(f"  Achieved FAR:   {opt_metrics['false_accept_rate']:.4f}")
    print(f"  Achieved FRR:   {opt_metrics['false_reject_rate']:.4f}")
    print(f"  F1:             {opt_metrics['f1']:.4f}")

    # Generate plots
    print(f"\n--- Generating Plots ---")
    plot_roc_curve(predictions, test_labels, models_dir / "roc_curve.png")
    plot_prediction_distribution(predictions, test_labels, models_dir / "prediction_distribution.png")

    # Save evaluation results
    results = {
        "model_path": str(model_path),
        "test_size": len(test_features),
        "optimal_threshold": optimal,
        "metrics_at_thresholds": all_metrics,
        "optimal_metrics": opt_metrics,
    }
    with open(models_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved: {models_dir / 'evaluation_results.json'}")
    print(f"\n  Recommended threshold for deployment: {optimal:.2f}")
    print(f"  Set VOICE_WAKEWORD_THRESHOLD={optimal:.2f} in .env")


if __name__ == "__main__":
    main()
