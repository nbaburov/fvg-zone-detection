"""_shap_worker.py — SHAP analysis subprocess worker."""
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
    parser.add_argument("--timeframe", default="h1",
                        help="Timeframe token (h1|5m|15m).")
    parser.add_argument("--scope", default="spy",
                        help="Dataset scope (spy|multisym). Determines class_weights filename.")
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
        tf = args.timeframe
        # shap_xgb.py always operates on the SPY single-symbol train set
        train_df_path = ROOT / "data" / "processed" / f"spy_{tf}_train.parquet"
        import pandas as pd
        from src.features.window_features import extract_window_features
        train_df = pd.read_parquet(train_df_path)
        X_train, y_train = extract_window_features(train_df)

        cw_name = f"class_weights_spy_{tf}.json"
        with open(ROOT / "data" / "processed" / cw_name) as fh:
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
        fig.write_html(str(out_dir / "shap_summary_xgb.html"), include_plotlyjs="cdn")
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

    (out_dir / "shap_xgb_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
