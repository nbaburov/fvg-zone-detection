"""threshold_sweep.py — Per-class F1-optimal threshold tuning (Gap 4).

Usage:
  python scripts/rigor/threshold_sweep.py --model-path checkpoints/lstm/lstm_seed42.pt
      --model-type lstm [--output-dir reports/rigor]

Val is used for threshold selection. Test is the final measurement only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class threshold tuning")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-type", required=True, choices=["lstm", "xgboost"])
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to experiments/foo.yaml (optional — sets data_dir, eval params)")
    parser.add_argument("--set", dest="set_overrides", nargs="+", default=[],
                        metavar="key=value",
                        help='Override config fields: --set "data.data_dir=data/processed"')
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    # Resolve config (optional)
    cfg = None
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        from src.config.loader import load_experiment, parse_set_args
        overrides = parse_set_args(args.set_overrides)
        cfg = load_experiment(config_path, overrides or None)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    from src.rigor.report_utils import timestamped_dir
    from src.rigor.threshold import (
        apply_thresholds,
        compute_pr_curves,
        find_f1_optimal_threshold,
    )

    output_dir = args.output_dir or (cfg.runtime.output_dir if cfg else Path("reports/rigor"))
    ts_dir = timestamped_dir(ROOT / output_dir)
    data_dir = ROOT / (cfg.data.data_dir if cfg else Path("data/processed"))

    if args.model_type == "lstm":
        val_proba, val_true, test_proba, test_true = _get_lstm_probas(model_path, data_dir)
    else:
        val_proba, val_true, test_proba, test_true = _get_xgb_probas(model_path, data_dir)

    # Compute PR curves on val (not test)
    assert val_proba is not test_proba, "LOOKAHEAD: thresholds must be tuned on val, not test"
    assert not np.array_equal(val_true, test_true), "LOOKAHEAD: val and test labels are identical — wrong split"
    curves = compute_pr_curves(val_true, val_proba)

    optimal_thresholds: dict[int, float] = {}
    print("\nPer-class F1-optimal thresholds (from val set):")
    for cls in range(3):
        prec, rec, thr = curves[cls]
        opt_thr, opt_f1 = find_f1_optimal_threshold(prec, rec, thr)
        optimal_thresholds[cls] = opt_thr
        name = {0: "none", 1: "bull", 2: "bear"}[cls]
        print(f"  Class {cls} ({name}): threshold={opt_thr:.3f}, val F1={opt_f1:.4f}")

    from sklearn.metrics import f1_score

    argmax_test_preds = test_proba.argmax(axis=1)
    argmax_test_f1 = float(f1_score(test_true, argmax_test_preds, average="macro", zero_division=0.0))

    # Apply thresholds on test (final measurement)
    tuned_test_preds = apply_thresholds(test_proba, optimal_thresholds)
    tuned_test_f1 = float(f1_score(test_true, tuned_test_preds, average="macro", zero_division=0.0))

    per_class_argmax = f1_score(test_true, argmax_test_preds, average=None, zero_division=0.0, labels=[0, 1, 2])
    per_class_tuned = f1_score(test_true, tuned_test_preds, average=None, zero_division=0.0, labels=[0, 1, 2])

    print(f"\n{'Metric':<20} {'Argmax':<10} {'Tuned':<10} {'Delta':<10}")
    print("-" * 50)
    print(f"{'Macro F1':<20} {argmax_test_f1:<10.4f} {tuned_test_f1:<10.4f} {tuned_test_f1 - argmax_test_f1:+.4f}")
    for i, name in enumerate(["none_f1", "bull_f1", "bear_f1"]):
        delta = float(per_class_tuned[i]) - float(per_class_argmax[i])
        print(f"{name:<20} {per_class_argmax[i]:<10.4f} {per_class_tuned[i]:<10.4f} {delta:+.4f}")

    thresholds_out = {str(cls): float(thr) for cls, thr in optimal_thresholds.items()}
    thr_path = ts_dir / f"thresholds_{args.model_type}.json"
    with thr_path.open("w") as fh:
        json.dump({
            "thresholds": thresholds_out,
            "argmax_test_macro_f1": argmax_test_f1,
            "tuned_test_macro_f1": tuned_test_f1,
            "delta_macro_f1": tuned_test_f1 - argmax_test_f1,
        }, fh, indent=2)
    print(f"\nThresholds saved: {thr_path}")

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        class_names = {0: "none", 1: "bull", 2: "bear"}
        colors = {0: "gray", 1: "green", 2: "red"}
        for cls in range(3):
            prec, rec, _ = curves[cls]
            fig.add_trace(go.Scatter(
                x=rec[:-1], y=prec[:-1],
                mode="lines",
                name=f"Class {cls} ({class_names[cls]})",
                line=dict(color=colors[cls]),
            ))
        fig.update_layout(
            title=f"PR Curves — {args.model_type} (val set)",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        pr_path = ts_dir / f"pr_curves_{args.model_type}.html"
        fig.write_html(str(pr_path))
        print(f"PR curves HTML: {pr_path}")
    except ImportError:
        print("plotly not available — PR curve HTML not saved")


def _get_lstm_probas(model_path, data_dir):
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.models.lstm import FVGLSTMClassifier

    device = torch.device("cpu")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    hidden_size = state["lstm.weight_ih_l0"].shape[0] // 4
    num_layers = sum(1 for k in state if k.startswith("lstm.weight_ih_l"))

    model = FVGLSTMClassifier(hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(state)
    model.eval()
    labeller = LABELLERS["fvg_valid"]()

    def _get_proba(split):
        df = pd.read_parquet(data_dir / f"spy_h1_{split}.parquet")
        ds = SMCWindowDataset(df, labeller, stride=1, window_size=60, drop_cross_session_windows=False)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        probas_list, true_list = [], []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x)
                proba = torch.softmax(logits, dim=-1).numpy()
                probas_list.append(proba)
                true_list.extend(y.numpy().tolist() if isinstance(y, torch.Tensor) else list(y))
        return np.concatenate(probas_list), np.array(true_list, dtype=np.int64)

    val_proba, val_true = _get_proba("val")
    test_proba, test_true = _get_proba("test")
    return val_proba, val_true, test_proba, test_true


def _get_xgb_probas(model_path, data_dir):
    """XGBoost inference via subprocess (Python 3.14 segfault workaround)."""
    import subprocess
    import tempfile
    import pandas as pd
    from src.features.window_features import extract_window_features

    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")
    X_val, y_val = extract_window_features(val_df)
    X_test, y_test = extract_window_features(test_df)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f_in:
        in_path = Path(f_in.name)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f_out:
        out_path = Path(f_out.name)

    np.savez(in_path, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    worker_lines = [
        "import sys; from pathlib import Path; import numpy as np",
        "ROOT = Path(sys.argv[0]).resolve().parent.parent",
        "sys.path.insert(0, str(ROOT))",
        "from src.models.xgboost_baseline import XGBoostFVGClassifier",
        "clf = XGBoostFVGClassifier.load(sys.argv[1])",
        "data = np.load(sys.argv[2])",
        "vp = clf.predict_proba(data['X_val'])",
        "tp = clf.predict_proba(data['X_test'])",
        "np.savez(sys.argv[3], val_proba=vp, val_true=data['y_val'], test_proba=tp, test_true=data['y_test'])",
    ]
    worker_path = ROOT / "scripts" / "_xgb_infer_worker.py"
    worker_path.write_text("\n".join(worker_lines) + "\n")

    subprocess.run(
        [sys.executable, str(worker_path), str(model_path), str(in_path), str(out_path)],
        check=True, cwd=str(ROOT),
    )
    result = np.load(out_path)
    val_proba, val_true = result["val_proba"], result["val_true"]
    test_proba, test_true = result["test_proba"], result["test_true"]
    in_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return val_proba, val_true, test_proba, test_true


if __name__ == "__main__":
    main()
