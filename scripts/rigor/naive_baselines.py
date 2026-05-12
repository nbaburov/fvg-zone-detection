"""naive_baselines.py — Compute majority-class and uniform-random naive baselines.

Loads the test split (2023-2025 default) and evaluates two baselines:
  1. Majority-class: always predict class 0 (none).
  2. Uniform random: sample uniformly from {0, 1, 2} over 1000 trials.

Outputs: reports/rigor/2026-05-13/baselines/naive.json

Usage:
    python scripts/rigor/naive_baselines.py [--splits {default|legacy}]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.labels import LABELLERS
from src.data.split import SPLIT_BOUNDARIES_2018_2024, temporal_split
from src.data.window import SMCWindowDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute naive baselines on test split.")
    p.add_argument(
        "--splits",
        choices=["default", "legacy"],
        default="default",
        help="default: pre-split parquets (2016-2025). legacy: SPLIT_BOUNDARIES_2018_2024.",
    )
    p.add_argument(
        "--output",
        default="reports/rigor/2026-05-13/baselines/naive.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of random trials for uniform baseline.",
    )
    return p.parse_args()


def get_test_labels(splits: str) -> np.ndarray:
    """Return window-level labels for the test split."""
    data_dir = REPO / "data" / "processed"
    labeller_key = "fvg_valid"
    labeller = LABELLERS[labeller_key]()

    if splits == "legacy":
        labeled_df = pd.read_parquet(data_dir / "spy_h1_labeled.parquet")
        # Re-apply ValidFVG labeller so label column matches the default-splits task
        _labeller = LABELLERS["fvg_valid"]()
        _valid_labels = _labeller.label(labeled_df)
        labeled_df = labeled_df.copy()
        labeled_df["label"] = _valid_labels.map(_labeller.encoded_map).astype(int)
        _, _, test_df = temporal_split(labeled_df, SPLIT_BOUNDARIES_2018_2024)
        print(f"  Legacy splits: test rows={len(test_df)}")
    else:
        test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")
        print(f"  Default splits: test rows={len(test_df)}")

    test_ds = SMCWindowDataset(
        test_df, labeller, stride=1, window_size=60, drop_cross_session_windows=False
    )
    y_true = np.array([int(test_ds[i][1]) for i in range(len(test_ds))])
    print(f"  Test windows: {len(y_true)}, label dist: {dict(zip(*np.unique(y_true, return_counts=True)))}")
    return y_true


def majority_baseline(y_true: np.ndarray) -> dict:
    majority = int(np.bincount(y_true).argmax())
    y_pred = np.full_like(y_true, majority)
    report = classification_report(
        y_true, y_pred, target_names=["none", "bull", "bear"],
        output_dict=True, zero_division=0,
    )
    return {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "none_f1": float(report["none"]["f1-score"]),
        "bull_f1": float(report["bull"]["f1-score"]),
        "bear_f1": float(report["bear"]["f1-score"]),
        "majority_class": majority,
    }


def uniform_random_baseline(y_true: np.ndarray, n_samples: int, rng_seed: int = 0) -> dict:
    rng = np.random.default_rng(rng_seed)
    f1_scores = []
    for _ in range(n_samples):
        y_pred = rng.integers(0, 3, size=len(y_true))
        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2], output_dict=True, zero_division=0,
        )
        f1_scores.append(float(report["macro avg"]["f1-score"]))
    return {
        "macro_f1_mean": float(np.mean(f1_scores)),
        "macro_f1_std": float(np.std(f1_scores)),
        "n_samples": n_samples,
    }


def main() -> None:
    args = parse_args()

    print(f"\nNaive Baselines — splits={args.splits}")
    y_true = get_test_labels(args.splits)

    print("\nComputing majority-class baseline...")
    maj = majority_baseline(y_true)
    print(f"  Macro-F1: {maj['macro_f1']:.4f}  (majority class={maj['majority_class']})")

    print(f"\nComputing uniform-random baseline ({args.n_samples} trials)...")
    uni = uniform_random_baseline(y_true, args.n_samples)
    print(f"  Macro-F1: {uni['macro_f1_mean']:.4f} ± {uni['macro_f1_std']:.4f}")

    result = {
        "splits": args.splits,
        "majority_class": maj,
        "uniform_random": uni,
    }

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
