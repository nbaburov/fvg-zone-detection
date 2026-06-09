"""_xgb_eval_spy_worker.py — XGB SPY-only-test eval worker (subprocess, TORCH-FREE).

IMPORTANT: Do NOT import torch or any module that imports torch.
This file is invoked by eval_spy_test.py via subprocess so that torch is not
loaded in this process — loading torch before xgboost.predict() causes a
segfault on macOS arm64 (same root cause as fit()).

Stdin:  JSON with keys: ckpt_paths (list[str]), spy_test_parquet (str),
        class_weights_path (str)
Stdout: JSON with list of per-seed result dicts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# Torch-free imports only
import xgboost as xgb
from sklearn.metrics import f1_score


def main() -> None:
    payload = json.load(sys.stdin)
    ckpt_paths: list[str] = payload["ckpt_paths"]
    spy_test_parquet: str = payload["spy_test_parquet"]
    seeds: list[int] = payload["seeds"]

    # Import torch-free feature extractor
    import pandas as pd
    from src.features.window_features import extract_window_features

    test_df = pd.read_parquet(spy_test_parquet)
    X_test, y_test = extract_window_features(test_df)

    results: list[dict] = []
    for seed, ckpt_path_str in zip(seeds, ckpt_paths):
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            results.append({"seed": seed, "missing": True})
            continue
        try:
            clf = xgb.XGBClassifier()
            clf.load_model(str(ckpt_path))
            y_pred = clf.predict(X_test).astype(int)
            macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0.0))
            per_class = [float(v) for v in f1_score(y_test, y_pred, average=None, zero_division=0.0)]
            results.append({
                "seed": seed,
                "missing": False,
                "macro_f1": macro_f1,
                "none_f1": per_class[0],
                "bull_f1": per_class[1],
                "bear_f1": per_class[2],
                # Raw arrays for bootstrap CI (serialised as lists; caller saves npz)
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
            })
        except Exception as exc:
            results.append({"seed": seed, "missing": False, "error": str(exc)})

    print(json.dumps(results))


if __name__ == "__main__":
    main()
