"""shap_xgb.py — SHAP feature importance analysis for XGBoost (Gap 5).

Usage:
  python scripts/rigor/shap_xgb.py --checkpoint checkpoints/xgboost/xgb_seed42.ubj
      [--config reports/rigor/<ts>/best_xgb_config.json]
      [--drop-threshold 1e-4] [--output-dir reports/rigor]

Outputs:
  reports/rigor/<ts>/shap_summary_xgb.html
  reports/rigor/<ts>/shap_dep_<feature>.html  (top 5)
  reports/rigor/<ts>/shap_xgb_report.md

Runs in subprocess to avoid Python 3.14 XGBoost segfault.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis for XGBoost FVG classifier")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None,
                        help="best_xgb_config.json (for pruned retrain)")
    parser.add_argument("--drop-threshold", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rigor"))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    from src.rigor.report_utils import timestamped_dir
    ts_dir = timestamped_dir(ROOT / args.output_dir)

    # Load val data (for SHAP) and test data (for pruned retrain comparison)
    import pandas as pd
    from src.features.window_features import extract_window_features, FEATURE_NAMES

    data_dir = ROOT / "data" / "processed"
    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")
    X_val, y_val = extract_window_features(val_df)
    X_test, y_test = extract_window_features(test_df)

    # Run SHAP in subprocess
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        data_path = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        results_path = Path(f.name)

    np.savez(data_path, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    worker_path = ROOT / "scripts" / "rigor" / "_workers" / "_shap_worker.py"
    _write_shap_worker(worker_path)

    config_arg = str(args.config) if args.config else ""
    cmd = [
        sys.executable, str(worker_path),
        "--checkpoint", str(ckpt_path),
        "--data-npz", str(data_path),
        "--results-json", str(results_path),
        "--drop-threshold", str(args.drop_threshold),
        "--output-dir", str(ts_dir),
    ]
    if config_arg:
        cmd += ["--config", config_arg]

    subprocess.run(cmd, check=True, cwd=str(ROOT))

    data_path.unlink(missing_ok=True)

    # Load and display results
    if results_path.exists():
        with results_path.open() as fh:
            results = json.load(fh)
        results_path.unlink(missing_ok=True)

        print("\nTop 10 features by mean |SHAP|:")
        for feat, val in sorted(results["mean_shap"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {feat:<30} {val:.6f}")

        dropped = results.get("dropped_features", [])
        print(f"\nFeatures below threshold {args.drop_threshold}: {dropped}")
        if "pruned_test_f1" in results:
            print(f"Full model test macro F1:   {results['full_test_f1']:.4f}")
            print(f"Pruned model test macro F1: {results['pruned_test_f1']:.4f}")
            print(f"Delta: {results['pruned_test_f1'] - results['full_test_f1']:+.4f}")

        print(f"\nOutputs: {ts_dir}")
    else:
        print("WARNING: results JSON not found — SHAP worker may have failed")


def _write_shap_worker(path: Path) -> None:
    """Write SHAP subprocess worker."""
    code = '''"""_shap_worker.py — SHAP analysis subprocess worker."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-npz", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--drop-threshold", type=float, default=1e-4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: pip install shap>=0.46")

    from src.models.xgboost_baseline import XGBoostFVGClassifier
    from src.features.window_features import FEATURE_NAMES
    from sklearn.metrics import f1_score

    clf = XGBoostFVGClassifier.load(args.checkpoint)
    data = np.load(args.data_npz)
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SHAP values on val set
    explainer = shap.TreeExplainer(clf._model)
    shap_vals = explainer.shap_values(X_val)  # list of 3 arrays or 3D array

    # Handle both formats
    if isinstance(shap_vals, list):
        mean_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
    else:
        mean_shap = np.abs(shap_vals).mean(axis=(0, 2)) if shap_vals.ndim == 3 else np.abs(shap_vals).mean(axis=0)

    mean_shap_dict = {FEATURE_NAMES[i]: float(mean_shap[i]) for i in range(len(FEATURE_NAMES))}

    # Identify features to drop
    dropped = [f for f, v in mean_shap_dict.items() if v < args.drop_threshold]
    keep_mask = np.array([v >= args.drop_threshold for v in mean_shap.tolist()])

    # Full model test F1
    y_pred_full = clf.predict(X_test)
    full_f1 = float(f1_score(y_test, y_pred_full, average="macro", zero_division=0.0))

    # Pruned retrain (if config provided)
    pruned_f1 = None
    if args.config and Path(args.config).exists():
        import xgboost as xgb
        with open(args.config) as fh:
            hp = json.load(fh)
        hp_clean = {k: v for k, v in hp.items()
                    if k not in ("val_macro_f1", "trial_number", "study_name", "storage")}
        train_df_path = ROOT / "data" / "processed" / "spy_h1_train.parquet"
        import pandas as pd
        from src.features.window_features import extract_window_features
        train_df = pd.read_parquet(train_df_path)
        X_train, y_train = extract_window_features(train_df)

        with open(ROOT / "data" / "processed" / "class_weights.json") as fh:
            cw = json.load(fh)
        w_arr = [float(cw[str(i)]) for i in range(3)]
        sw = np.array([w_arr[int(y)] for y in y_train], dtype=np.float32)

        xgb_metrics = ["mlogloss"]
        pruned_clf = xgb.XGBClassifier(
            **{k: v for k, v in hp_clean.items()},
            eval_metric=xgb_metrics,
            verbosity=0,
        )
        pruned_clf.fit(
            X_train[:, keep_mask], y_train,
            sample_weight=sw,
            eval_set=[(X_val[:, keep_mask], y_val)],
            verbose=False,
        )
        pruned_pred = pruned_clf.predict(X_test[:, keep_mask])
        pruned_f1 = float(f1_score(y_test, pruned_pred, average="macro", zero_division=0.0))

    # Save HTML summary (bar chart)
    try:
        import plotly.graph_objects as go
        ranked = sorted(mean_shap_dict.items(), key=lambda x: x[1], reverse=True)
        names_sorted, vals_sorted = zip(*ranked)
        fig = go.Figure(go.Bar(x=list(vals_sorted), y=list(names_sorted), orientation="h"))
        fig.update_layout(title="Mean |SHAP| per feature (XGBoost)", height=900)
        fig.write_html(str(out_dir / "shap_summary_xgb.html"))
    except Exception:
        pass

    results = {
        "mean_shap": mean_shap_dict,
        "dropped_features": dropped,
        "full_test_f1": full_f1,
    }
    if pruned_f1 is not None:
        results["pruned_test_f1"] = pruned_f1

    with open(args.results_json, "w") as fh:
        json.dump(results, fh, indent=2)

    # Write markdown report
    lines = ["# SHAP Feature Importance — XGBoost", "",
             f"Full model test macro F1: {full_f1:.4f}"]
    if pruned_f1 is not None:
        lines.append(f"Pruned model test macro F1: {pruned_f1:.4f}")
        lines.append(f"Delta: {pruned_f1 - full_f1:+.4f}")
    lines += ["", f"Features dropped (mean |SHAP| < {args.drop_threshold}): {dropped}",
              "", "| Feature | Mean |SHAP| |", "|---------|------------|"]
    for f, v in sorted(mean_shap_dict.items(), key=lambda x: -x[1]):
        lines.append(f"| {f} | {v:.6f} |")

    (out_dir / "shap_xgb_report.md").write_text("\\n".join(lines))


if __name__ == "__main__":
    main()
'''
    path.write_text(code)


if __name__ == "__main__":
    main()
