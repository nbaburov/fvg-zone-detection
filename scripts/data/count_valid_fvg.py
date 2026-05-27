"""
Sparsity gate script — Phase 4.

Run BEFORE starting gold re-annotation.

Loads data/processed/spy_h1.parquet, runs ValidFVGLabeller, prints per-criterion
breakdown and total positive count.

Exit codes:
  0 — n_pos >= 150  → proceed to gold re-annotation
  2 — 75 <= n_pos < 150 → relax criterion #3 first (increase confluence_atr_mult)
  1 — n_pos < 75   → escalate to user before proceeding

Usage:
    python scripts/data/count_valid_fvg.py

Saves ablation CSV to .nb/analysis/criterion-ablation-<date>.csv.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run as script
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels.valid_fvg import ValidFVGLabeller  # noqa: E402

PARQUET_PATH = ROOT / "data" / "processed" / "spy_h1.parquet"
ANALYSIS_DIR = ROOT / ".nb" / "analysis"

# Decision-tree thresholds (from plan Risk 2)
GATE_PROCEED = 150
GATE_ESCALATE = 75


def main() -> None:  # noqa: C901
    if not PARQUET_PATH.exists():
        print(f"ERROR: Parquet not found at {PARQUET_PATH}")
        print("Run the data pipeline first: python -m src.data.pipeline")
        sys.exit(1)

    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    n_total = len(df)
    print(f"Total candles: {n_total:,}")

    # Ensure OHLCV columns present (parquet may contain extra label columns)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    labeller = ValidFVGLabeller()
    print("\nRunning ValidFVGLabeller (default parameters)...")
    labels, ablation = labeller.label_with_ablation(df[["open", "high", "low", "close", "volume"]])

    n_bull = (labels == 1).sum()
    n_bear = (labels == -1).sum()
    n_pos = n_bull + n_bear
    pos_rate = 100.0 * n_pos / n_total if n_total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  Total candles:      {n_total:>8,}")
    print(f"  Bull valid FVG:     {n_bull:>8,}")
    print(f"  Bear valid FVG:     {n_bear:>8,}")
    print(f"  Total positives:    {n_pos:>8,}")
    print(f"  Positive rate:      {pos_rate:>8.2f}%")
    print(f"{'='*50}\n")

    # Per-criterion pass counts (computed at middle-candle positions)
    crit_cols = {
        "Geometric FVG (bull)":   "geometric_bull",
        "Geometric FVG (bear)":   "geometric_bear",
        "Crit2 strict (bull)":    "crit2_strict_bull",
        "Crit2 strict (bear)":    "crit2_strict_bear",
        "Crit2 loose  (bull)":    "crit2_loose_bull",
        "Crit2 loose  (bear)":    "crit2_loose_bear",
        "Crit3 S/R    (bull)":    "crit3_sr_bull",
        "Crit3 S/R    (bear)":    "crit3_sr_bear",
        "Crit5 Gann   (bull)":    "crit5_gann_bull",
        "Crit5 Gann   (bear)":    "crit5_gann_bear",
        "Crit6 BOS    (bull)":    "crit6_bos_bull",
        "Crit6 BOS    (bear)":    "crit6_bos_bear",
        "Valid bull":              "valid_bull",
        "Valid bear":              "valid_bear",
    }
    print("Per-criterion pass counts (rows where that criterion is True):")
    for label_str, col in crit_cols.items():
        count = ablation[col].sum()
        rate = 100.0 * count / n_total
        print(f"  {label_str:<28}: {count:>6,}  ({rate:5.1f}%)")

    # Rejection funnel: geometric → each criterion
    n_geom_bull = ablation["geometric_bull"].sum()
    n_geom_bear = ablation["geometric_bear"].sum()
    print(f"\nRejection funnel (bull path, starting from {n_geom_bull} geometric FVGs):")
    funnel_cols_bull = [
        ("crit2_strict_bull", "Crit2 strict"),
        ("crit3_sr_bull",     "Crit3 S/R"),
        ("crit5_gann_bull",   "Crit5 Gann"),
        ("crit6_bos_bull",    "Crit6 BOS"),
    ]
    # Compute cumulative AND
    cumulative = ablation["geometric_bull"].copy()
    for col, name in funnel_cols_bull:
        cumulative = cumulative & ablation[col]
        n_left = cumulative.sum()
        print(f"  + {name:<16}: {n_left:>6,} remaining")
    print(f"  = Valid bull total  : {ablation['valid_bull'].sum():>6,}")

    # Save ablation CSV
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"criterion-ablation-{date.today()}.csv"
    ablation.to_csv(out_path)
    print(f"\nAblation CSV saved to: {out_path}")

    # Decision-tree gate
    print(f"\n{'='*50}")
    if n_pos >= GATE_PROCEED:
        print(f"GATE PASSED — n_pos={n_pos} >= {GATE_PROCEED}.")
        print("Proceed to gold re-annotation (Phase 5).")
        sys.exit(0)
    elif n_pos >= GATE_ESCALATE:
        print(f"GATE WARNING — n_pos={n_pos} is in [{GATE_ESCALATE}, {GATE_PROCEED}).")
        print("Action required: relax criterion #3 (increase confluence_atr_mult 0.5 → 1.0).")
        print("Re-run with ValidFVGLabeller(confluence_atr_mult=1.0) and check count again.")
        print("If still < 150 after relaxation, drop criterion #3 entirely.")
        print("Log which relaxation was applied before proceeding.")
        sys.exit(2)
    else:
        print(f"GATE FAILED — n_pos={n_pos} < {GATE_ESCALATE}.")
        print("Action required: relax crit3 AND reduce bos_lookback from 20 to 50.")
        print("Escalate to user before proceeding. Do NOT start gold re-annotation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
