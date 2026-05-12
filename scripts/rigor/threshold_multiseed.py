"""threshold_multiseed.py — Per-seed threshold tuning, aggregated to G4 output (Gap 4).

Runs threshold_sweep logic for each LSTM seed checkpoint.
Aggregates delta macro F1 across seeds.
Writes G4/threshold_tuning.json.

GUARD: threshold search uses val set only. Test set accessed only for final measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def get_lstm_probas(model_path: Path, df, labeller, device):
    """Load LSTM checkpoint and get probabilities for a dataframe."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.window import SMCWindowDataset
    from src.models.lstm import FVGLSTMClassifier

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    hidden_size = state["lstm.weight_ih_l0"].shape[0] // 4
    num_layers = sum(1 for k in state if k.startswith("lstm.weight_ih_l"))
    model = FVGLSTMClassifier(hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(state)
    model.to(device)

    ds = SMCWindowDataset(df, labeller, stride=1, window_size=60, drop_cross_session_windows=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    probas, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probas.append(p)
            labels.append(y.numpy())

    return np.vstack(probas), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/lstm"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 17, 42, 123, 2024])
    args = parser.parse_args()

    ckpt_dir = ROOT / args.checkpoint_dir
    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    from sklearn.metrics import f1_score

    from src.data.labels import LABELLERS
    from src.rigor.threshold import (
        apply_thresholds,
        compute_pr_curves,
        find_f1_optimal_threshold,
    )

    device = torch.device("cpu")
    labeller = LABELLERS["fvg_valid"]()  # instantiate the labeller

    # Load val + test data once (guard: val for threshold search, test for final measurement only)
    print("Loading val and test parquet...")
    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")

    per_seed: dict[str, dict] = {}
    deltas: list[float] = []

    for seed in args.seeds:
        ckpt = ckpt_dir / f"lstm_seed{seed}.pt"
        if not ckpt.exists():
            print(f"WARNING: checkpoint not found: {ckpt} — skipping seed {seed}")
            continue

        print(f"\n--- Seed {seed} ---")

        # Get val probabilities (threshold search — val only, no test leakage)
        print("  Getting val probas...")
        val_proba, val_true = get_lstm_probas(ckpt, val_df, labeller, device)

        # Get test probabilities (final measurement only, after thresholds are fixed)
        print("  Getting test probas...")
        test_proba, test_true = get_lstm_probas(ckpt, test_df, labeller, device)

        # Threshold optimisation on val ONLY
        curves = compute_pr_curves(val_true, val_proba)
        optimal_thresholds: dict[int, float] = {}
        for cls in range(3):
            prec, rec, thr = curves[cls]
            opt_thr, opt_f1 = find_f1_optimal_threshold(prec, rec, thr)
            optimal_thresholds[cls] = opt_thr
            name = {0: "none", 1: "bull", 2: "bear"}[cls]
            print(f"  Class {cls} ({name}): threshold={opt_thr:.3f}, val F1={opt_f1:.4f}")

        # Apply thresholds to test (final measurement — executed exactly once, after threshold fixed)
        argmax_pred = test_proba.argmax(axis=1)
        tuned_pred = apply_thresholds(test_proba, optimal_thresholds)

        argmax_f1 = float(f1_score(test_true, argmax_pred, average="macro", zero_division=0.0))
        tuned_f1 = float(f1_score(test_true, tuned_pred, average="macro", zero_division=0.0))
        delta = tuned_f1 - argmax_f1

        print(f"  Argmax test macro F1: {argmax_f1:.4f}")
        print(f"  Tuned  test macro F1: {tuned_f1:.4f}")
        print(f"  Delta:                {delta:+.4f}")

        per_seed[str(seed)] = {
            "argmax_macro_f1": argmax_f1,
            "tuned_macro_f1": tuned_f1,
            "delta_macro_f1": delta,
            "thresholds": {str(c): float(t) for c, t in optimal_thresholds.items()},
        }
        deltas.append(delta)

    if not deltas:
        print("ERROR: no seeds processed")
        sys.exit(1)

    result = {
        "per_seed": per_seed,
        "mean_delta_macro_f1": float(np.mean(deltas)),
        "std_delta_macro_f1": float(np.std(deltas)),
        "n_seeds": len(deltas),
    }

    out_path = out_dir / "threshold_tuning.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== G4 Summary ===")
    print(f"Mean delta macro F1: {result['mean_delta_macro_f1']:+.4f} ± {result['std_delta_macro_f1']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
