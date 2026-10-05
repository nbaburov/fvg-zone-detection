"""rule_window_baseline.py — the labelling rule itself as a baseline, limited to the models' window.

Usage:
    python scripts/rigor/eval/rule_window_baseline.py

Runs ValidFVGLabeller on only the last W candles before each SPY H1 test position, the same input a
W-candle model window holds, and scores it against the full-history labels with macro-F1.

    exact_W60   every position a 60-candle window can end on (the models' test set, n = 5,198)
    W20..W120   one common position set (t >= 119), so the truth is identical across W

Needs data/processed/spy_h1_test.parquet (build the dataset first, see README).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.labels.valid_fvg import ValidFVGLabeller

WINDOWS = (20, 40, 60, 80, 100, 120)


def main() -> None:
    labeller = ValidFVGLabeller()
    test = pd.read_parquet("data/processed/spy_h1_test.parquet")
    truth = test["label"].to_numpy()
    ohlcv = test[["open", "high", "low", "close", "volume"]]

    def rule_at(t: int, w: int) -> int:
        window = ohlcv.iloc[t - w + 1 : t + 1]
        # The labeller zeroes its last two rows, so pad two copies of the final candle to keep row w - 1 labelled.
        padded = pd.concat([window, window.iloc[[-1, -1]]], ignore_index=True)
        return int(labeller.encode(labeller.label(padded)).to_numpy()[w - 1])

    def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
        return float(f1_score(y, p, average="macro", zero_division=0.0))

    out: dict[str, object] = {}
    pos60 = np.arange(59, len(test))
    pred60 = np.array([rule_at(t, 60) for t in pos60])
    out["exact_W60"] = {
        "n": int(len(pos60)),
        "macro_f1": macro_f1(truth[pos60], pred60),
        "always_none": macro_f1(truth[pos60], np.zeros_like(pos60)),
    }
    common = np.arange(119, len(test))
    for w in WINDOWS:
        out[f"W{w}"] = macro_f1(truth[common], np.array([rule_at(t, w) for t in common]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
