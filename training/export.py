"""
Export the best trained model checkpoint to ONNX format.

Usage
-----
    python export.py \
        --checkpoint ./checkpoints/best.pt \
        --dataset_dir /data/pitch_dataset \
        --output_dir ../models

After running this script you will have:
    models/pitch_detector.onnx      – model for the Rust runtime
    models/normalization.json       – mean/std constants to embed in Rust

Verification
------------
The script also runs the exported ONNX model against the PyTorch model on a
batch of random inputs and asserts that outputs match within 1e-4.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import ImprovedPitchDetector, NUM_NOTES, NUM_SEMITONES, BINS_PER_SEMITONE


def export(checkpoint: Path, dataset_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ──────────────────────────────────────────────────────────
    model = ImprovedPitchDetector()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── export to ONNX ──────────────────────────────────────────────────────
    dummy_input = torch.randn(1, 1, 128)
    onnx_path   = output_dir / "pitch_detector.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,   # folds BatchNorm into Conv weights
        input_names=["mel_features"],
        output_names=["count_logits", "note_logits"],
        dynamic_axes={
            "mel_features":  {0: "batch"},
            "count_logits":  {0: "batch"},
            "note_logits":   {0: "batch"},
        },
        training=torch.onnx.TrainingMode.EVAL,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved: {onnx_path}")
    size_kb = onnx_path.stat().st_size / 1024
    print(f"Model size: {size_kb:.1f} KB")

    # ── verification ────────────────────────────────────────────────────────
    session    = ort.InferenceSession(str(onnx_path))
    test_input = np.random.randn(8, 1, 128).astype(np.float32)

    with torch.no_grad():
        pt_count_logits, pt_note_logits = model(torch.from_numpy(test_input))
        pt_note_probs = torch.sigmoid(pt_note_logits).numpy()

    ort_count_logits, ort_note_logits = session.run(
        ["count_logits", "note_logits"], {"mel_features": test_input}
    )
    ort_note_probs = 1.0 / (1.0 + np.exp(-ort_note_logits))  # sigmoid

    max_note_diff  = float(np.abs(pt_note_probs - ort_note_probs).max())
    max_count_diff = float(np.abs(pt_count_logits.numpy() - ort_count_logits).max())
    print(f"Max note  output difference: {max_note_diff:.2e}")
    print(f"Max count output difference: {max_count_diff:.2e}")
    assert max_note_diff  < 1e-4, f"Note ONNX mismatch: {max_note_diff}"
    assert max_count_diff < 1e-4, f"Count ONNX mismatch: {max_count_diff}"
    print("ONNX verification passed.")

    # ── copy normalization constants ─────────────────────────────────────────
    norm_src = dataset_dir / "normalization.json"
    norm_dst = output_dir  / "normalization.json"
    if norm_src.exists():
        shutil.copy(norm_src, norm_dst)
        with open(norm_dst) as f:
            norm = json.load(f)
        print(f"Normalization constants: mean={norm['mean']:.4f}, std={norm['std']:.4f}")

        # Embed model metadata alongside normalization so Rust can read it
        # from one file without hardcoding bin counts in the source.
        norm["num_semitones"]     = NUM_SEMITONES
        norm["bins_per_semitone"] = BINS_PER_SEMITONE
        with open(norm_dst, "w") as f:
            json.dump(norm, f, indent=2)
        print(f"Normalization saved:  {norm_dst}")

        # Print Rust constants for embedding in pitch_detector.rs
        print("\n── Paste these into src/ml/pitch_detector.rs ──────────────")
        print(f"const NORM_MEAN: f32 = {norm['mean']:.6f};")
        print(f"const NORM_STD:  f32 = {norm['std']:.6f};")
        print(f"const NUM_SEMITONES: usize = {NUM_SEMITONES};")
        print(f"const BINS_PER_SEMITONE: usize = {BINS_PER_SEMITONE};")
        print("────────────────────────────────────────────────────────────")
    else:
        print(f"WARNING: normalization.json not found at {norm_src}")

    print("\nExport complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True, help="Path to best.pt from training")
    parser.add_argument("--dataset_dir", required=True, help="Dataset dir containing normalization.json")
    parser.add_argument("--output_dir",  default="../models")
    args = parser.parse_args()

    export(
        Path(args.checkpoint),
        Path(args.dataset_dir),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
