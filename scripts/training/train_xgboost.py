"""train_xgboost.py — End-to-end XGBoost baseline training for FVG classification.

Usage:
    python scripts/training/train_xgboost.py [--seed 42] [--output-dir checkpoints/xgboost]

Steps:
    1. Set global seed.
    2. Log environment info.
    3. Load train/val/test splits from data/processed/.
    4. Extract 35 tabular features (stride=1 for all splits).
    5. Sanity checks: label distribution, finite features, positive count.
    6. Train XGBoostFVGClassifier with early stopping on val mlogloss.
    7. Evaluate on val set, then test set.
    8. Save model + metadata.
    9. Write evaluation log to .nb-suite/test-logs/11-May-26/xgboost-baseline.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# Ensure repo root is on sys.path
REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.labels import LABELLERS
from src.data.split import SPLIT_BOUNDARIES_2018_2024, temporal_split
from src.features.window_features import FEATURE_NAMES, extract_window_features
from src.models.xgboost_baseline import XGBoostFVGClassifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost FVG baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints/xgboost")
    parser.add_argument(
        "--splits",
        default="default",
        help=(
            "default: use pre-split parquets from data/processed/. "
            "legacy: re-split spy_h1_labeled.parquet with SPLIT_BOUNDARIES_2018_2024. "
            "Or a path to a directory containing spy_h1_{train,val,test}.parquet."
        ),
    )
    parser.add_argument(
        "--label",
        choices=["validfvg", "rawfvg"],
        default="validfvg",
        help=(
            "validfvg (default): load class_weights.json, save as xgb_seed{N}.ubj. "
            "rawfvg: load class_weights_rawfvg.json from splits dir, save as xgb_seed{N}_rawfvg.ubj."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=str(REPO)
        ).strip()
    except Exception:
        return "unknown"


def majority_baseline_f1(y_true: np.ndarray) -> float:
    """Macro-F1 of always-predict-majority-class predictor."""
    majority = np.bincount(y_true).argmax()
    preds = np.full_like(y_true, majority)
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)
    return float(report["macro avg"]["f1-score"])


def format_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> str:
    header = "             " + "  ".join(f"{n:>8}" for n in class_names)
    rows = [header]
    for i, row in enumerate(cm):
        rows.append(f"  {class_names[i]:>10}  " + "  ".join(f"{v:>8}" for v in row))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed = args.seed
    output_dir = REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    sha = git_sha()
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"XGBoost FVG Baseline — Phase 4 Step 0")
    print(f"{'='*60}")
    print(f"Git SHA:       {sha}")
    print(f"Python:        {sys.version.split()[0]}")
    print(f"XGBoost:       {xgb.__version__}")
    print(f"Seed:          {seed}")
    print(f"Splits:        {args.splits}")
    print(f"Timestamp:     {ts}")
    print()

    # ------------------------------------------------------------------
    # 1. Load splits
    # ------------------------------------------------------------------
    data_dir = REPO / "data" / "processed"
    print("Loading splits...")
    if args.splits == "legacy":
        labeled_df = pd.read_parquet(data_dir / "spy_h1_labeled.parquet")
        # Re-apply ValidFVG labeller so label column matches the default-splits task
        _labeller = LABELLERS["fvg_valid"]()
        _valid_labels = _labeller.label(labeled_df)
        labeled_df = labeled_df.copy()
        labeled_df["label"] = _valid_labels.map(_labeller.encoded_map).astype(int)
        train_df, val_df, test_df = temporal_split(labeled_df, SPLIT_BOUNDARIES_2018_2024)
        print(f"  Using SPLIT_BOUNDARIES_2018_2024 + ValidFVG re-label (train 2018-2021, test 2023-2024)")
        splits_data_dir = data_dir
    elif args.splits == "default":
        train_df = pd.read_parquet(data_dir / "spy_h1_train.parquet")
        val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
        test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")
        splits_data_dir = data_dir
    else:
        # Treat as a directory path containing pre-split parquets
        splits_data_dir = Path(args.splits)
        train_df = pd.read_parquet(splits_data_dir / "spy_h1_train.parquet")
        val_df = pd.read_parquet(splits_data_dir / "spy_h1_val.parquet")
        test_df = pd.read_parquet(splits_data_dir / "spy_h1_test.parquet")
        print(f"  Using custom splits dir: {splits_data_dir}")

    cw_filename = "class_weights_rawfvg.json" if args.label == "rawfvg" else "class_weights.json"
    with open(splits_data_dir / cw_filename) as fh:
        cw_raw = json.load(fh)
    class_weights = {int(k): float(v) for k, v in cw_raw.items()}
    print(f"Class weights: {class_weights}")

    # ------------------------------------------------------------------
    # 2. Extract features (stride=1 for all splits)
    # ------------------------------------------------------------------
    print("\nExtracting features (stride=1, window=60)...")
    X_train, y_train = extract_window_features(train_df, window_size=60, stride=1)
    X_val, y_val = extract_window_features(val_df, window_size=60, stride=1)
    X_test, y_test = extract_window_features(test_df, window_size=60, stride=1)

    # ------------------------------------------------------------------
    # 3. Sanity checks
    # ------------------------------------------------------------------
    def _label_dist(y: np.ndarray, name: str) -> dict:
        counts = dict(zip(*np.unique(y, return_counts=True)))
        pos = int(sum(v for k, v in counts.items() if k != 0))
        total = len(y)
        print(f"  {name}: {total} windows | labels={counts} | positives={pos} ({100.*pos/max(1,total):.1f}%)")
        return counts

    print("\nLabel distribution:")
    _label_dist(y_train, "Train")
    _label_dist(y_val, "Val  ")
    _label_dist(y_test, "Test ")

    test_pos = int((y_test != 0).sum())
    if test_pos < 200:
        print(f"\nWARNING: test positives={test_pos} < 200. F1 may be unreliable.")
    else:
        print(f"\nPositive count sanity: {test_pos} test positives (>= 200 OK)")

    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        assert np.isfinite(X).all(), f"NaN/inf in {name} features"
    print("Feature matrix sanity: all finite OK")

    # ------------------------------------------------------------------
    # 4. Sample weights
    # ------------------------------------------------------------------
    sample_weight = np.array([class_weights[int(yi)] for yi in y_train], dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\nTraining XGBoostFVGClassifier...")
    model = XGBoostFVGClassifier()
    model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)

    n_used = model.n_estimators_used
    evals = model.evals_result
    val_mlogloss_history = evals.get("validation_0", {}).get("mlogloss", [])
    best_mlogloss = min(val_mlogloss_history) if val_mlogloss_history else float("nan")
    val_merror_history = evals.get("validation_0", {}).get("merror", [])
    best_merror_val = val_merror_history[n_used - 1] if val_merror_history and n_used else float("nan")

    print(f"  Final n_estimators used: {n_used}")
    print(f"  Best val mlogloss:       {best_mlogloss:.6f}")
    print(f"  Val merror at best:      {best_merror_val:.4f}")

    # Risk mitigation: if early-stop fires at < 50 rounds, fall back to fixed n_estimators=527
    # (best from archived Optuna). This was seen on ValidFVG ~97% none; rawfvg ~25% pos should
    # not trigger it, but guard just in case.
    FALLBACK_N_ESTIMATORS = 527
    if n_used < 50:
        print(
            f"\nWARNING: XGBoost stopped at {n_used} < 50 estimators — early-stop likely fired "
            f"before convergence. Falling back to fixed n_estimators={FALLBACK_N_ESTIMATORS}."
        )
        model_fb = XGBoostFVGClassifier(params={"n_estimators": FALLBACK_N_ESTIMATORS})
        model_fb.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
        model = model_fb
        n_used = FALLBACK_N_ESTIMATORS
        evals = model.evals_result
        val_mlogloss_history = evals.get("validation_0", {}).get("mlogloss", [])
        best_mlogloss = min(val_mlogloss_history) if val_mlogloss_history else float("nan")
        print(f"  Fallback n_estimators={FALLBACK_N_ESTIMATORS}, best val mlogloss={best_mlogloss:.6f}")

    # ------------------------------------------------------------------
    # 6. Naive baseline
    # ------------------------------------------------------------------
    naive_f1 = majority_baseline_f1(y_test)
    print(f"\nNaive majority baseline macro-F1 (test): {naive_f1:.4f}")

    # ------------------------------------------------------------------
    # 7. Validation evaluation
    # ------------------------------------------------------------------
    class_names = ["none", "bull", "bear"]

    print("\n" + "="*40)
    print("VALIDATION RESULTS")
    print("="*40)
    y_val_pred = model.predict(X_val)
    val_report_str = classification_report(y_val, y_val_pred, target_names=class_names,
                                           digits=4, zero_division=0)
    print(val_report_str)
    val_cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1, 2])
    print("Confusion matrix (val):")
    print(format_confusion_matrix(val_cm, class_names))

    val_report_dict = classification_report(y_val, y_val_pred, target_names=class_names,
                                            output_dict=True, zero_division=0)

    # ------------------------------------------------------------------
    # 8. Test evaluation
    # ------------------------------------------------------------------
    print("\n" + "="*40)
    print("TEST RESULTS (PRIMARY)")
    print("="*40)
    y_test_pred = model.predict(X_test)
    test_report_str = classification_report(y_test, y_test_pred, target_names=class_names,
                                            digits=4, zero_division=0)
    print(test_report_str)
    test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2])
    print("Confusion matrix (test):")
    print(format_confusion_matrix(test_cm, class_names))

    test_report_dict = classification_report(y_test, y_test_pred, target_names=class_names,
                                             output_dict=True, zero_division=0)

    macro_f1_test = test_report_dict["macro avg"]["f1-score"]
    bull_f1_test = test_report_dict.get("bull", {}).get("f1-score", float("nan"))
    bear_f1_test = test_report_dict.get("bear", {}).get("f1-score", float("nan"))

    print(f"\n  Macro-F1:   {macro_f1_test:.4f}")
    print(f"  Bull F1:    {bull_f1_test:.4f}")
    print(f"  Bear F1:    {bear_f1_test:.4f}")

    # ------------------------------------------------------------------
    # 9. Feature importances
    # ------------------------------------------------------------------
    importances = model.feature_importances_
    top10_idx = np.argsort(importances)[::-1][:10]
    top10 = [(FEATURE_NAMES[i], float(importances[i])) for i in top10_idx]

    print("\nTop-10 feature importances:")
    for rank, (fname, imp) in enumerate(top10, 1):
        print(f"  {rank:2}. {fname:<20} {imp:.6f}")

    # ------------------------------------------------------------------
    # 10. Save model
    # ------------------------------------------------------------------
    label_tag = f"_{args.label}" if args.label != "validfvg" else ""
    model_path = output_dir / f"xgb_seed{seed}{label_tag}.ubj"
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # Write meta.json sidecar
    import platform
    meta_path = output_dir / f"xgb_seed{seed}{label_tag}.meta.json"
    meta_out = {
        "seed": seed,
        "splits": args.splits,
        "label": args.label,
        "git_sha": sha,
        "timestamp": ts,
        "python": sys.version.split()[0],
        "xgboost": xgb.__version__,
        "platform": platform.platform(),
        "n_estimators_used": int(n_used),
        "best_val_mlogloss": float(best_mlogloss),
        "val_macro_f1": float(val_report_dict["macro avg"]["f1-score"]),
        "test_macro_f1": float(macro_f1_test),
        "test_bull_f1": float(bull_f1_test),
        "test_bear_f1": float(bear_f1_test),
        "naive_majority_macro_f1": float(naive_f1),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta_out, fh, indent=2)
    print(f"Meta saved:  {meta_path}")

    # ------------------------------------------------------------------
    # 11. Write evaluation log
    # ------------------------------------------------------------------
    log_dir = REPO / ".nb-suite" / "test-logs" / "11-May-26"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "xgboost-baseline.md"

    n_train_win = len(y_train)
    n_val_win = len(y_val)
    n_test_win = len(y_test)
    train_pos = int((y_train != 0).sum())
    val_pos = int((y_val != 0).sum())

    # Val mlogloss curve excerpt
    curve_head = val_mlogloss_history[:5] if val_mlogloss_history else []
    curve_tail = val_mlogloss_history[-5:] if val_mlogloss_history else []

    log_content = f"""# XGBoost Baseline Evaluation — Phase 4 Step 0

**Date:** {ts[:10]}
**Git SHA:** {sha}
**Python:** {sys.version.split()[0]}
**XGBoost:** {xgb.__version__}
**Seed:** {seed}

## Dataset summary

| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|--------------|
| Train | {n_train_win} | {train_pos} | {100.*train_pos/max(1,n_train_win):.1f}% |
| Val   | {n_val_win} | {val_pos} | {100.*val_pos/max(1,n_val_win):.1f}% |
| Test  | {n_test_win} | {test_pos} | {100.*test_pos/max(1,n_test_win):.1f}% |

Note: stride=1, window=60 for all splits. Overlapping windows are expected.
Test positives sanity: {test_pos} (threshold: >= 200).

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 (max) |
| max_depth | 4 |
| learning_rate | 0.05 |
| min_child_weight | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | multi:softprob |
| num_class | 3 |
| early_stopping_rounds | 30 |
| tree_method | hist |
| seed | {seed} |

## Training

- Final n_estimators used: {n_used} (early stopping on val mlogloss)
- Best val mlogloss: {best_mlogloss:.6f}
- Val merror at best: {best_merror_val:.4f}
- Val mlogloss curve (first 5): {[f'{v:.4f}' for v in curve_head]}
- Val mlogloss curve (last 5):  {[f'{v:.4f}' for v in curve_tail]}

## Naive baseline (always-majority, test set)

F1 macro: {naive_f1:.4f}  ← majority-class predictor for reference

## Validation results

```
{val_report_str}
```

Confusion matrix (val) — rows=actual, cols=predicted (none/bull/bear):
```
{format_confusion_matrix(val_cm, class_names)}
```

## Test results (primary)

```
{test_report_str}
```

Confusion matrix (test) — rows=actual, cols=predicted (none/bull/bear):
```
{format_confusion_matrix(test_cm, class_names)}
```

- **Macro-F1: {macro_f1_test:.4f}**
- Minority class F1 (bull): {bull_f1_test:.4f}
- Minority class F1 (bear): {bear_f1_test:.4f}
- Positive count sanity: {test_pos} windows (>= 200 {"OK" if test_pos >= 200 else "WARNING: below threshold"})

## Comparison to naive baseline

| Metric | Naive (majority) | XGBoost | Delta |
|--------|-----------------|---------|-------|
| Macro-F1 | {naive_f1:.4f} | {macro_f1_test:.4f} | {macro_f1_test - naive_f1:+.4f} |
| Bull F1  | 0.0000 | {bull_f1_test:.4f} | {bull_f1_test:+.4f} |
| Bear F1  | 0.0000 | {bear_f1_test:.4f} | {bear_f1_test:+.4f} |

## Top-10 feature importances

| Rank | Feature | Importance |
|------|---------|-----------|
{"".join(f"| {r} | {n} | {v:.6f} |\n" for r, (n, v) in enumerate(top10, 1))}

## Phase 4 handoff

- **XGBoost test macro-F1: {macro_f1_test:.4f}** — this is the floor DL models must beat.
- Bull F1 floor: {bull_f1_test:.4f}
- Bear F1 floor: {bear_f1_test:.4f}
- Model saved at: checkpoints/xgboost/xgb_seed42.ubj
- Naive baseline macro-F1: {naive_f1:.4f} (delta to XGBoost: {macro_f1_test - naive_f1:+.4f})

Note on overlapping windows: stride=1 produces highly overlapping windows (59/60 shared bars between adjacent windows). This is standard for XGBoost tabular evaluation and matches the distribution DL models will train on. F1 is computed on a held-out temporal test split (2023-2024), so temporal leakage is not a concern despite overlap.
"""

    with log_path.open("w") as fh:
        fh.write(log_content)
    print(f"\nEvaluation log: {log_path}")

    # ------------------------------------------------------------------
    # 12. Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("HANDOFF SUMMARY")
    print(f"{'='*60}")
    print(f"XGBoost test macro-F1:    {macro_f1_test:.4f}")
    print(f"Bull F1:                  {bull_f1_test:.4f}")
    print(f"Bear F1:                  {bear_f1_test:.4f}")
    print(f"Naive baseline macro-F1:  {naive_f1:.4f}")
    print(f"Delta vs naive:           {macro_f1_test - naive_f1:+.4f}")
    print()
    if macro_f1_test > naive_f1 + 0.05:
        print("DL RECOMMENDATION: XGBoost beats naive by >5pp. DL chain is worth pursuing.")
    else:
        print("DL RECOMMENDATION: XGBoost improvement over naive is marginal. DL must show clear gain.")
    print(f"\nF1 floor for DL models to beat: macro-F1={macro_f1_test:.4f}, bull-F1={bull_f1_test:.4f}, bear-F1={bear_f1_test:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
