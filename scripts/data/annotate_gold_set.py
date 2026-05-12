#!/usr/bin/env python3
"""
Gold-set annotation runner.

Usage:
    python scripts/data/annotate_gold_set.py

Loads the processed H1 DataFrame, samples the 75-candle gold set,
opens the interactive Plotly annotation UI, and saves results to
data/gold_labels.csv. Resumes from last annotated candle if interrupted.

After annotation: computes and prints Cohen's kappa. If kappa < 0.6,
Phase 4 is blocked until label quality is resolved.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.annotate import compute_kappa, run_annotation_ui, sample_gold_set

PROCESSED_H1_PATH = ROOT / "data" / "processed" / "spy_h1.parquet"
GOLD_LABELS_PATH = ROOT / "data" / "gold_labels.csv"


def main() -> None:
    if not PROCESSED_H1_PATH.exists():
        print(
            f"ERROR: {PROCESSED_H1_PATH} not found.\n"
            "Run build_labelled_dataset() first to generate the processed H1 data."
        )
        sys.exit(1)

    print(f"Loading H1 data from {PROCESSED_H1_PATH}...")
    df = pd.read_parquet(PROCESSED_H1_PATH)
    print(f"Loaded {len(df)} bars.")

    print("\nSampling 75-candle gold set (25 FVG-rich / 25 low-vol / 25 random)...")
    gold_set_df = sample_gold_set(df, n_fvg_rich=25, n_low_vol=25, n_random=25, seed=42)
    print(f"Gold set sampled: {len(gold_set_df)} candles.")

    already_done = 0
    if GOLD_LABELS_PATH.exists():
        existing = pd.read_csv(GOLD_LABELS_PATH)
        already_done = len(existing)
        print(f"\nResuming: {already_done} candles already annotated.")

    print("\nStarting annotation UI...")
    print("Controls: b=bullish, e=bearish, n=none, a=ambiguous")
    print("Close the chart window after each annotation.")
    print()

    run_annotation_ui(df, gold_set_df, output_path=str(GOLD_LABELS_PATH))

    print("\nComputing Cohen's kappa...")
    kappa = compute_kappa(str(GOLD_LABELS_PATH))
    print(f"\nFinal kappa: {kappa:.3f}")


if __name__ == "__main__":
    main()
