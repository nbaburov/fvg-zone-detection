"""naive_baselines.py — Compute majority-class and random-uniform F1 for both label sets.

Reads:
    data/processed/spy_h1_test.parquet         (ValidFVG)
    data/processed/rawfvg/spy_h1_test.parquet  (raw FVG)

Writes:
    reports/rigor/2026-05-13/baselines/naive_baselines.json

Usage:
    .venv/bin/python scripts/eval/naive_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT_DIR = REPO / "reports" / "rigor" / "2026-05-13" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def majority_f1(y: np.ndarray) -> float:
    majority = int(np.bincount(y.astype(int)).argmax())
    preds = np.full_like(y, majority)
    return float(f1_score(y, preds, average="macro", zero_division=0))


def random_f1(y: np.ndarray, n_classes: int = 3, n_trials: int = 100, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_trials):
        preds = rng.integers(0, n_classes, size=len(y))
        scores.append(float(f1_score(y, preds, average="macro", zero_division=0)))
    return float(np.mean(scores))


def load_labels(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    return df["label"].to_numpy()


def main() -> int:
    validfvg_path = REPO / "data" / "processed" / "spy_h1_test.parquet"
    rawfvg_path   = REPO / "data" / "processed" / "rawfvg" / "spy_h1_test.parquet"

    for p in [validfvg_path, rawfvg_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run build_rawfvg_splits.py first.")
            return 1

    y_valid = load_labels(validfvg_path)
    y_raw   = load_labels(rawfvg_path)

    results = {
        "validfvg": {
            "majority_macro_f1": majority_f1(y_valid),
            "random_macro_f1":   random_f1(y_valid),
            "n_test":            int(len(y_valid)),
            "pos_rate":          float((y_valid != 0).mean()),
        },
        "rawfvg": {
            "majority_macro_f1": majority_f1(y_raw),
            "random_macro_f1":   random_f1(y_raw),
            "n_test":            int(len(y_raw)),
            "pos_rate":          float((y_raw != 0).mean()),
        },
    }

    out_path = OUT_DIR / "naive_baselines.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Naive baselines → {out_path}")
    for label, r in results.items():
        print(f"  {label}: majority F1={r['majority_macro_f1']:.4f}, "
              f"random F1={r['random_macro_f1']:.4f}, "
              f"pos_rate={r['pos_rate']:.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
