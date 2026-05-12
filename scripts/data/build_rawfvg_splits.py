"""build_rawfvg_splits.py — Generate raw-FVG labelled splits from 2016-2025 OHLCV data.

Outputs to data/processed/rawfvg/ — does NOT overwrite ValidFVG splits in data/processed/.

Usage:
    .venv/bin/python scripts/data/build_rawfvg_splits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.data.labels import LABELLERS
from src.data.process import build_labelled_dataset
from src.data.split import temporal_split
from src.data.pipeline import _compute_class_weights

OUT_DIR = REPO / "data" / "processed" / "rawfvg"
LABELLER_NAME = "fvg"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building raw-FVG splits → {OUT_DIR}")
    print(f"Labeller: '{LABELLER_NAME}' (FVGLabeller, geometric 3-candle @ N+1)")

    if LABELLER_NAME not in LABELLERS:
        print(f"ERROR: labeller '{LABELLER_NAME}' not found. Registered: {list(LABELLERS)}")
        return 1

    # Build labelled dataset — use_cache=True so spy_h1.parquet OHLCV is reused, no Alpaca call.
    # output_path is set to rawfvg/ subdir so the canonical data/processed/spy_h1.parquet
    # (ValidFVG) is NOT overwritten. This is the critical fix for the write-back side effect.
    print("Labelling from cached OHLCV (use_cache=True) ...")
    full_df = build_labelled_dataset(
        labeller_name=LABELLER_NAME,
        start="2016-01-01",
        end="2025-12-31",
        use_cache=True,
        output_path=str(OUT_DIR / "spy_h1_rawfvg_full.parquet"),
    )
    print(f"Full labelled dataset: {len(full_df)} rows")

    # Temporal split
    train_df, val_df, test_df = temporal_split(full_df)
    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Gate A1: check positive rate
    labeller = LABELLERS[LABELLER_NAME]()
    total_train = len(train_df)
    pos_train = int((train_df["label"] != labeller.encoded_map.get("none", 0)).sum())
    pos_rate = pos_train / max(1, total_train)
    print(f"\nPositive rate (train): {pos_rate:.3%} ({pos_train}/{total_train})")

    if pos_rate < 0.05 or pos_rate > 0.40:
        print(
            f"\nGATE A1 FAILED: positive rate {pos_rate:.3%} outside expected 5–40% range.\n"
            "This suggests a data pipeline error. Inspect splits before proceeding."
        )
        return 2

    print(f"Gate A1 PASSED: positive rate {pos_rate:.3%} in expected 5–40% range.")

    # Persist splits
    for name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = OUT_DIR / f"spy_h1_{name}.parquet"
        df_split.to_parquet(path)
        pos = int((df_split["label"] != 0).sum())
        rate = pos / max(1, len(df_split))
        print(f"  {name}: {len(df_split)} rows, positives={pos} ({rate:.1%}) → {path}")

    # Class weights (from train split only)
    num_classes = labeller.num_classes
    class_weights = _compute_class_weights(train_df, num_classes=num_classes)
    weights_dict = {str(i): float(class_weights[i]) for i in range(num_classes)}

    weights_path = OUT_DIR / "class_weights_rawfvg.json"
    with open(weights_path, "w") as fh:
        json.dump(weights_dict, fh, indent=2)
    print(f"\nClass weights → {weights_path}")
    print(f"  {weights_dict}")

    print("\nDone. Raw-FVG splits ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
