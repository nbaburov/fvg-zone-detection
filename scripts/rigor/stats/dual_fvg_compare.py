"""dual_fvg_compare.py — Produce side-by-side raw-FVG vs ValidFVG comparison report.

Reads:
    checkpoints/lstm_h1_spy/lstm_seed42_rawfvg.meta.json
    checkpoints/lstm_h1_spy/lstm_seed42.meta.json          (ValidFVG — Phase C)
    checkpoints/xgboost_h1_spy/xgb_seed42_rawfvg.meta.json
    checkpoints/xgboost_h1_spy/xgb_seed42.meta.json        (ValidFVG — Phase C)
    reports/rigor/baselines/naive_baselines.json

Writes:
    reports/rigor/baselines/dual_fvg_compare.json
    reports/rigor/baselines/dual_fvg_compare.md

Usage:
    .venv/bin/python scripts/rigor/stats/dual_fvg_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

REPORT_DIR = REPO / "reports" / "rigor" / "baselines"

LSTM_RAWFVG_META  = REPO / "checkpoints" / "lstm_h1_spy"    / "lstm_seed42_rawfvg.meta.json"
LSTM_VALID_META   = REPO / "checkpoints" / "lstm_h1_spy"    / "lstm_seed42.meta.json"
XGB_RAWFVG_META   = REPO / "checkpoints" / "xgboost_h1_spy" / "xgb_seed42_rawfvg.meta.json"
XGB_VALID_META    = REPO / "checkpoints" / "xgboost_h1_spy" / "xgb_seed42.meta.json"
NAIVE_BASELINES   = REPORT_DIR / "naive_baselines.json"

# Hard gate thresholds
HARD_GATE_DELTA = 0.40
HARD_GATE_RAW_F1 = 0.92


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [LSTM_RAWFVG_META, LSTM_VALID_META, XGB_RAWFVG_META, XGB_VALID_META, NAIVE_BASELINES]:
        if not p.exists():
            print(f"ERROR: required input not found: {p}")
            return 1

    lstm_raw  = load_json(LSTM_RAWFVG_META)
    lstm_val  = load_json(LSTM_VALID_META)
    xgb_raw   = load_json(XGB_RAWFVG_META)
    xgb_val   = load_json(XGB_VALID_META)
    naive     = load_json(NAIVE_BASELINES)

    lstm_raw_f1  = lstm_raw["test_macro_f1"]
    lstm_val_f1  = lstm_val["test_macro_f1"]
    xgb_raw_f1   = xgb_raw["test_macro_f1"]
    xgb_val_f1   = xgb_val["test_macro_f1"]

    # Gate A5 (hard gate)
    raw_delta = lstm_raw_f1 - lstm_val_f1
    if raw_delta > HARD_GATE_DELTA and lstm_raw_f1 > HARD_GATE_RAW_F1:
        print(
            f"\nHARD GATE A5 TRIGGERED:\n"
            f"  raw-FVG LSTM F1 = {lstm_raw_f1:.4f}\n"
            f"  ValidFVG LSTM F1 = {lstm_val_f1:.4f}\n"
            f"  Delta = {raw_delta:.4f} > {HARD_GATE_DELTA} AND raw-FVG F1 > {HARD_GATE_RAW_F1}\n"
            "This suggests a labelling pipeline error in rawfvg splits. "
            "Inspect before migrating to Part B."
        )
        return 2

    # Build output dict
    compare = {
        "gate_a5_passed": True,
        "lstm_rawfvg": {
            "test_macro_f1": lstm_raw_f1,
            "test_bull_f1":  lstm_raw.get("test_bull_f1", 0.0),
            "test_bear_f1":  lstm_raw.get("test_bear_f1", 0.0),
            "pos_rate":      naive["rawfvg"]["pos_rate"],
        },
        "lstm_validfvg": {
            "test_macro_f1": lstm_val_f1,
            "test_bull_f1":  lstm_val.get("test_bull_f1", 0.0),
            "test_bear_f1":  lstm_val.get("test_bear_f1", 0.0),
            "pos_rate":      naive["validfvg"]["pos_rate"],
        },
        "xgb_rawfvg": {
            "test_macro_f1": xgb_raw_f1,
            "test_bull_f1":  xgb_raw.get("test_bull_f1", 0.0),
            "test_bear_f1":  xgb_raw.get("test_bear_f1", 0.0),
            "pos_rate":      naive["rawfvg"]["pos_rate"],
        },
        "xgb_validfvg": {
            "test_macro_f1": xgb_val_f1,
            "test_bull_f1":  xgb_val.get("test_bull_f1", 0.0),
            "test_bear_f1":  xgb_val.get("test_bear_f1", 0.0),
            "pos_rate":      naive["validfvg"]["pos_rate"],
        },
        "naive_rawfvg": {
            "test_macro_f1": naive["rawfvg"]["majority_macro_f1"],
            "pos_rate":      naive["rawfvg"]["pos_rate"],
        },
        "naive_validfvg": {
            "test_macro_f1": naive["validfvg"]["majority_macro_f1"],
            "pos_rate":      naive["validfvg"]["pos_rate"],
        },
    }

    json_path = REPORT_DIR / "dual_fvg_compare.json"
    with open(json_path, "w") as fh:
        json.dump(compare, fh, indent=2)
    print(f"JSON report → {json_path}")

    # Markdown table
    def fmt(v: float) -> str:
        return f"{v:.4f}"

    def pct(v: float) -> str:
        return f"{v:.1%}"

    md_lines = [
        "# Dual-FVG Baseline Comparison",
        "",
        "> **Note:** raw FVG and ValidFVG measure different problems (different label difficulty).",
        "> Raw FVG higher F1 is expected — not a regression.",
        "",
        f"| Model | Label | Pos rate | Test Macro F1 | Bull F1 | Bear F1 |",
        f"|-------|-------|----------|---------------|---------|---------|",
        f"| LSTM seed42  | raw FVG  | {pct(compare['lstm_rawfvg']['pos_rate'])}  | {fmt(compare['lstm_rawfvg']['test_macro_f1'])}  | {fmt(compare['lstm_rawfvg']['test_bull_f1'])}  | {fmt(compare['lstm_rawfvg']['test_bear_f1'])}  |",
        f"| LSTM seed42  | ValidFVG | {pct(compare['lstm_validfvg']['pos_rate'])}   | {fmt(compare['lstm_validfvg']['test_macro_f1'])}  | {fmt(compare['lstm_validfvg']['test_bull_f1'])}  | {fmt(compare['lstm_validfvg']['test_bear_f1'])}  |",
        f"| XGB seed42   | raw FVG  | {pct(compare['xgb_rawfvg']['pos_rate'])}  | {fmt(compare['xgb_rawfvg']['test_macro_f1'])}  | {fmt(compare['xgb_rawfvg']['test_bull_f1'])}  | {fmt(compare['xgb_rawfvg']['test_bear_f1'])}  |",
        f"| XGB seed42   | ValidFVG | {pct(compare['xgb_validfvg']['pos_rate'])}   | {fmt(compare['xgb_validfvg']['test_macro_f1'])}  | {fmt(compare['xgb_validfvg']['test_bull_f1'])}  | {fmt(compare['xgb_validfvg']['test_bear_f1'])}  |",
        f"| Naive majority | raw FVG  | {pct(compare['naive_rawfvg']['pos_rate'])}  | {fmt(compare['naive_rawfvg']['test_macro_f1'])}  | —  | —  |",
        f"| Naive majority | ValidFVG | {pct(compare['naive_validfvg']['pos_rate'])}   | {fmt(compare['naive_validfvg']['test_macro_f1'])}  | —  | —  |",
        "",
        f"*Gate A5 passed. raw-FVG LSTM delta vs ValidFVG: {raw_delta:+.4f} (threshold: >{HARD_GATE_DELTA} AND raw F1 >{HARD_GATE_RAW_F1}).*",
    ]

    md_path = REPORT_DIR / "dual_fvg_compare.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"MD report  → {md_path}")

    print("\n=== PART A RESULTS ===")
    print(f"  LSTM   raw-FVG  F1: {fmt(lstm_raw_f1)}")
    print(f"  LSTM   ValidFVG F1: {fmt(lstm_val_f1)}  (Phase C)")
    print(f"  XGB    raw-FVG  F1: {fmt(xgb_raw_f1)}")
    print(f"  XGB    ValidFVG F1: {fmt(xgb_val_f1)}  (Phase C)")
    print(f"  Naive  raw-FVG  F1: {fmt(compare['naive_rawfvg']['test_macro_f1'])}")
    print(f"  Naive  ValidFVG F1: {fmt(compare['naive_validfvg']['test_macro_f1'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
