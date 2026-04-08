"""Export trained PyTorch wake word model to ONNX format.

Converts the trained classifier to ONNX for deployment in the Argus voice
pipeline via OpenWakeWord's ONNX runtime.

Run from IDE or CLI:
    python rnd/wakeword/training/export_onnx.py
    python rnd/wakeword/training/export_onnx.py --model models/best_model.pt --output models/hey_argus.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None

try:
    import onnx
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


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
    hidden_dim = checkpoint.get("hidden_dim", model_cfg["hidden_dim"])

    if model_cfg["type"] == "rnn":
        model = WakeWordRNN(
            frame_dim=96,
            num_frames=input_dim // 96,
            hidden_dim=hidden_dim,
            num_layers=model_cfg["num_layers"],
            dropout=0.0,
        )
    else:
        model = WakeWordDNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=model_cfg["num_layers"],
            dropout=0.0,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_dim, checkpoint


def export_to_onnx(model: "torch.nn.Module", input_dim: int, output_path: Path):
    """Export PyTorch model to ONNX."""
    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )


def verify_onnx(onnx_path: Path, input_dim: int, torch_model: "torch.nn.Module"):
    """Verify ONNX model produces same outputs as PyTorch model."""
    if not HAS_ONNX:
        print("  onnx/onnxruntime not installed — skipping verification")
        return False

    # Check model is valid
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # Compare outputs
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_inputs = np.random.randn(10, input_dim).astype(np.float32)

    # ONNX predictions
    onnx_outputs = session.run(["output"], {"input": test_inputs})[0]

    # PyTorch predictions
    with torch.no_grad():
        torch_outputs = torch_model(torch.from_numpy(test_inputs)).numpy()

    # Compare
    max_diff = np.max(np.abs(onnx_outputs.flatten() - torch_outputs.flatten()))
    mean_diff = np.mean(np.abs(onnx_outputs.flatten() - torch_outputs.flatten()))

    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if max_diff < 1e-5:
        print(f"  Verification: PASSED")
        return True
    else:
        print(f"  Verification: FAILED (max diff > 1e-5)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export wake word model to ONNX")
    parser.add_argument("--config", default="rnd/wakeword/config.yaml")
    parser.add_argument("--training-config", default="rnd/wakeword/training/config.yaml")
    parser.add_argument("--model", default=None, help="Path to PyTorch model checkpoint")
    parser.add_argument("--output", default=None, help="Output ONNX path")
    args = parser.parse_args()

    if torch is None:
        print("Error: PyTorch not installed.")
        return

    config = load_config(args.config, args.training_config)
    models_dir = Path(config["paths"]["models_dir"])

    model_path = Path(args.model) if args.model else models_dir / "best_model.pt"

    wake_phrase_slug = config["wake_phrase"].replace(" ", "_").lower()
    output_path = Path(args.output) if args.output else models_dir / f"{wake_phrase_slug}.onnx"

    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        print("Run train.py first.")
        return

    print(f"--- ONNX Export ---")
    print(f"  Input:  {model_path}")
    print(f"  Output: {output_path}")

    # Load model
    model, input_dim, checkpoint = load_model(model_path, torch.device("cpu"))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,}")

    # Export
    print(f"\n  Exporting to ONNX...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model, input_dim, output_path)

    file_size = output_path.stat().st_size
    print(f"  File size: {file_size / 1024:.1f} KB")

    # Verify
    print(f"\n  Verifying ONNX model...")
    verify_onnx(output_path, input_dim, model)

    # Deployment instructions
    print(f"\n--- Deployment ---")
    print(f"  1. Copy the model to your Argus data directory:")
    print(f"     cp {output_path} data/piper/{wake_phrase_slug}.onnx")
    print(f"")
    print(f"  2. Update .env:")
    print(f"     VOICE_WAKEWORD_MODEL={output_path}")
    print(f"")
    print(f"  3. Or reference by path in voice/config.py")
    print(f"")
    print(f"  Model ready for deployment!")


if __name__ == "__main__":
    main()
