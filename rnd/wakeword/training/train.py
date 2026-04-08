"""Train OpenWakeWord classifier on extracted features.

Trains a small DNN/RNN classifier on the embedding features extracted by
extract_features.py. Outputs a PyTorch model and training metrics.

Run from IDE or CLI:
    python rnd/wakeword/training/train.py
    python rnd/wakeword/training/train.py --epochs 200 --hidden-dim 128
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:
    torch = None


def load_config(
    global_config_path: str = "rnd/wakeword/config.yaml",
    training_config_path: str = "rnd/wakeword/training/config.yaml",
) -> dict:
    with open(global_config_path) as f:
        config = yaml.safe_load(f)
    with open(training_config_path) as f:
        config["model_config"] = yaml.safe_load(f)
    return config


class WakeWordDNN(nn.Module):
    """Small DNN classifier for wake word detection."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Output layer (binary classification)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.network(x).squeeze(-1)


class WakeWordRNN(nn.Module):
    """Small RNN classifier for wake word detection."""

    def __init__(self, frame_dim: int, num_frames: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_frames = num_frames
        self.frame_dim = frame_dim

        self.rnn = nn.GRU(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Reshape flat features back to (batch, num_frames, frame_dim)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_frames, self.frame_dim)
        _, hidden = self.rnn(x)
        return self.classifier(hidden[-1]).squeeze(-1)


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> dict:
    """Split data into train/val/test sets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(features))

    n_test = int(len(features) * test_split)
    n_val = int(len(features) * val_split)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    return {
        "train": (features[train_idx], labels[train_idx], weights[train_idx]),
        "val": (features[val_idx], labels[val_idx], weights[val_idx]),
        "test": (features[test_idx], labels[test_idx], weights[test_idx]),
    }


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification metrics."""
    pred_binary = (predictions >= threshold).astype(int)

    tp = np.sum((pred_binary == 1) & (labels == 1))
    fp = np.sum((pred_binary == 1) & (labels == 0))
    tn = np.sum((pred_binary == 0) & (labels == 0))
    fn = np.sum((pred_binary == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    false_accept_rate = fp / (fp + tn + 1e-10)
    false_reject_rate = fn / (fn + tp + 1e-10)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_accept_rate": float(false_accept_rate),
        "false_reject_rate": float(false_reject_rate),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: "torch.optim.Optimizer",
    criterion: nn.Module,
    device: "torch.device",
) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for features, labels, weights in loader:
        features = features.to(device)
        labels = labels.to(device).float()
        weights = weights.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss = (loss * weights).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: "torch.device",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns loss, predictions, labels."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for features, labels, weights in loader:
        features = features.to(device)
        labels = labels.to(device).float()

        outputs = model(features)
        loss = criterion(outputs, labels).mean()

        total_loss += loss.item()
        num_batches += 1

        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    return (
        total_loss / max(num_batches, 1),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def main():
    parser = argparse.ArgumentParser(description="Train wake word classifier")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml")
    parser.add_argument("--training-config", default="rnd/wakeword/training/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    args = parser.parse_args()

    if torch is None:
        print("Error: PyTorch not installed. Run: pip install torch")
        return

    config = load_config(args.config, args.training_config)
    model_cfg = config["model_config"]["model"]
    train_cfg = config["model_config"]["training"]
    data_cfg = config["model_config"]["data"]
    emb_cfg = config["model_config"]["embeddings"]

    epochs = args.epochs or train_cfg["epochs"]
    hidden_dim = args.hidden_dim or model_cfg["hidden_dim"]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load features
    features_dir = Path(config["paths"]["features_dir"])
    try:
        features = np.load(features_dir / "features.npy")
        labels = np.load(features_dir / "labels.npy")
        weights = np.load(features_dir / "weights.npy")
    except FileNotFoundError:
        print("Error: features not found. Run extract_features.py first.")
        return

    print(f"--- Wake Word Training ---")
    print(f"  Features: {features.shape}")
    print(f"  Positive: {np.sum(labels == 1)}")
    print(f"  Negative: {np.sum(labels == 0)}")
    print(f"  Device:   {device}")
    print(f"  Model:    {model_cfg['type']} (hidden={hidden_dim}, layers={model_cfg['num_layers']})")
    print(f"  Epochs:   {epochs}")

    # Split data
    splits = split_data(
        features, labels, weights,
        data_cfg["validation_split"],
        data_cfg["test_split"],
        data_cfg["seed"],
    )

    # Create dataloaders
    def make_loader(data_tuple, shuffle=True):
        f, l, w = data_tuple
        dataset = TensorDataset(
            torch.from_numpy(f),
            torch.from_numpy(l),
            torch.from_numpy(w),
        )
        return DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=shuffle)

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"], shuffle=False)
    test_loader = make_loader(splits["test"], shuffle=False)

    print(f"  Train: {len(splits['train'][0])}, Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")

    # Create model
    input_dim = features.shape[1]

    if model_cfg["type"] == "rnn":
        model = WakeWordRNN(
            frame_dim=emb_cfg["embedding_dim"],
            num_frames=emb_cfg["num_frames"],
            hidden_dim=hidden_dim,
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )
    else:
        model = WakeWordDNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    if train_cfg["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg["lr_step_size"],
            gamma=train_cfg["lr_gamma"],
        )

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val F1':>10} {'Val FAR':>10} {'Val FRR':>10}")
    print(f"{'─'*6} {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        metrics = compute_metrics(val_preds, val_labels)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        })

        print(
            f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} "
            f"{metrics['f1']:>10.4f} {metrics['false_accept_rate']:>10.4f} "
            f"{metrics['false_reject_rate']:>10.4f}"
        )

        # Save best model
        if metrics["f1"] > best_val_f1 + train_cfg["early_stopping_min_delta"]:
            best_val_f1 = metrics["f1"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "input_dim": input_dim,
                "epoch": epoch,
                "val_f1": best_val_f1,
            }, models_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\n  Early stopping at epoch {epoch} (patience={train_cfg['early_stopping_patience']})")
            break

    # Test evaluation
    print(f"\n--- Test Results ---")
    checkpoint = torch.load(models_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    for threshold in config["model_config"]["evaluation"]["thresholds"]:
        metrics = compute_metrics(test_preds, test_labels, threshold)
        marker = " <--" if abs(threshold - 0.5) < 0.01 else ""
        print(
            f"  Threshold {threshold:.1f}: "
            f"F1={metrics['f1']:.4f} "
            f"FAR={metrics['false_accept_rate']:.4f} "
            f"FRR={metrics['false_reject_rate']:.4f}"
            f"{marker}"
        )

    # Save training history
    with open(models_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_cfg,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "best_epoch": checkpoint["epoch"],
        "best_val_f1": best_val_f1,
    }, models_dir / "final_model.pt")

    print(f"\n  Best model: epoch {checkpoint['epoch']}, val F1={best_val_f1:.4f}")
    print(f"  Saved to: {models_dir}")
    print(f"\n  Next step: python rnd/wakeword/training/export_onnx.py")


if __name__ == "__main__":
    main()
