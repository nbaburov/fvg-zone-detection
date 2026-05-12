"""asymmetry_analysis.py — Bull vs Bear FVG asymmetry analysis (Gap 6).

Usage:
  python scripts/rigor/asymmetry_analysis.py
      --pred-dir reports/rigor/<ts>/   (dir with lstm_seed*_preds.npz files)
      [--output-dir reports/rigor]

Loads all multi-seed LSTM WeightedCE prediction npz files.
Computes per-class precision/recall/F1 per seed.
Flags if bull_f1 > bear_f1 by >0.05 consistently.
Structural investigation: compares gap size for FP bull vs TP bull windows.
Writes asymmetry.md + confusion HTML files + gap size distribution HTML.
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
    parser = argparse.ArgumentParser(description="Bull vs Bear asymmetry analysis")
    parser.add_argument("--pred-dir", required=True, type=Path,
                        help="Directory containing lstm_seed*_preds.npz files")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rigor"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir

    from src.rigor.report_utils import timestamped_dir
    ts_dir = timestamped_dir(ROOT / args.output_dir)
    data_dir = ROOT / args.data_dir

    # Find all lstm WeightedCE seed prediction files
    pred_files = sorted(pred_dir.glob("lstm_seed*_preds.npz"))
    if not pred_files:
        print(f"No lstm_seed*_preds.npz files found in {pred_dir}")
        sys.exit(1)

    print(f"Found {len(pred_files)} seed prediction files")

    from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

    seed_results = []
    all_y_true = []
    all_y_pred = []

    for pf in pred_files:
        data = np.load(pf)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0.0
        )
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
        seed_name = pf.stem.replace("_preds", "")

        seed_results.append({
            "seed": seed_name,
            "none_prec": float(prec[0]), "none_rec": float(rec[0]), "none_f1": float(f1[0]),
            "bull_prec": float(prec[1]), "bull_rec": float(rec[1]), "bull_f1": float(f1[1]),
            "bear_prec": float(prec[2]), "bear_rec": float(rec[2]), "bear_f1": float(f1[2]),
            "macro_f1": macro_f1,
        })

    # Aggregate
    bull_f1s = [r["bull_f1"] for r in seed_results]
    bear_f1s = [r["bear_f1"] for r in seed_results]
    bull_bear_gaps = [b - c for b, c in zip(bull_f1s, bear_f1s)]
    systematic = all(g > 0.05 for g in bull_bear_gaps)
    mean_gap = float(np.mean(bull_bear_gaps))

    print(f"\nBull F1: {np.mean(bull_f1s):.4f} ± {np.std(bull_f1s):.4f}")
    print(f"Bear F1: {np.mean(bear_f1s):.4f} ± {np.std(bear_f1s):.4f}")
    print(f"Mean bull-bear gap: {mean_gap:+.4f}")
    print(f"Systematic asymmetry (gap>0.05 all seeds): {systematic}")

    # Confusion HTML per seed
    try:
        import plotly.figure_factory as ff
        import plotly.graph_objects as go

        for i, pf in enumerate(pred_files):
            data = np.load(pf)
            cm = confusion_matrix(data["y_true"], data["y_pred"], labels=[0, 1, 2])
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
            labels = ["none", "bull", "bear"]
            fig = ff.create_annotated_heatmap(
                cm_norm, x=labels, y=labels, colorscale="Blues",
                annotation_text=[[f"{v:.2f}" for v in row] for row in cm_norm],
            )
            fig.update_layout(title=f"Confusion Matrix — {pf.stem}")
            seed_num = pf.stem.split("seed")[1].split("_")[0]
            html_path = ts_dir / f"confusion_seed{seed_num}.html"
            fig.write_html(str(html_path))
    except Exception as e:
        print(f"Confusion HTML skipped: {e}")

    # Structural gap size analysis
    _run_gap_analysis(data_dir, all_y_true, all_y_pred, ts_dir)

    # Write asymmetry.md
    _write_asymmetry_md(seed_results, bull_f1s, bear_f1s, mean_gap, systematic, ts_dir)
    print(f"\nOutputs: {ts_dir}")


def _run_gap_analysis(data_dir, all_y_true, all_y_pred, ts_dir):
    """Compare gap size for FP bull predictions vs TP bull predictions on test split."""
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from src.features.window_features import extract_window_features

        test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")
        X_test, y_test_arr = extract_window_features(test_df)

        # Use first seed's predictions
        y_true_seed0 = all_y_true[0]
        y_pred_seed0 = all_y_pred[0]

        # Feature index 25 = gap_bull, 26 = gap_bear
        n = min(len(y_true_seed0), len(X_test))
        y_true_s = y_true_seed0[:n]
        y_pred_s = y_pred_seed0[:n]
        X_s = X_test[:n]

        tp_bull_mask = (y_true_s == 1) & (y_pred_s == 1)
        fp_bull_mask = (y_true_s == 0) & (y_pred_s == 1)
        tp_bear_mask = (y_true_s == 2) & (y_pred_s == 2)
        fp_bear_mask = (y_true_s == 0) & (y_pred_s == 2)

        gap_tp_bull = X_s[tp_bull_mask, 25]   # gap_bull feature
        gap_fp_bull = X_s[fp_bull_mask, 25]
        gap_tp_bear = X_s[tp_bear_mask, 26]
        gap_fp_bear = X_s[fp_bear_mask, 26]

        print(f"\nGap size analysis (test set, seed 0):")
        print(f"  TP bull gap mean: {gap_tp_bull.mean():.5f}  FP bull gap mean: {gap_fp_bull.mean():.5f}")
        print(f"  TP bear gap mean: {gap_tp_bear.mean():.5f}  FP bear gap mean: {gap_fp_bear.mean():.5f}")

        fig = go.Figure()
        if len(gap_tp_bull) > 0:
            fig.add_trace(go.Histogram(x=gap_tp_bull, name="TP Bull", opacity=0.7, nbinsx=50))
        if len(gap_fp_bull) > 0:
            fig.add_trace(go.Histogram(x=gap_fp_bull, name="FP Bull", opacity=0.7, nbinsx=50))
        if len(gap_tp_bear) > 0:
            fig.add_trace(go.Histogram(x=gap_tp_bear, name="TP Bear", opacity=0.7, nbinsx=50))
        if len(gap_fp_bear) > 0:
            fig.add_trace(go.Histogram(x=gap_fp_bear, name="FP Bear", opacity=0.7, nbinsx=50))
        fig.update_layout(
            title="Gap Size Distribution: TP vs FP (test set)",
            xaxis_title="Gap size (raw, normalised by close)",
            yaxis_title="Count",
            barmode="overlay",
        )
        fig.write_html(str(ts_dir / "gap_size_distribution.html"))
    except Exception as e:
        print(f"Gap analysis skipped: {e}")


def _write_asymmetry_md(seed_results, bull_f1s, bear_f1s, mean_gap, systematic, ts_dir):
    lines = [
        "# Bull vs Bear FVG Asymmetry Analysis",
        "",
        "## Per-Seed Per-Class Metrics",
        "",
        "| Seed | Bull F1 | Bear F1 | Bull-Bear Gap | Macro F1 |",
        "|------|---------|---------|---------------|----------|",
    ]
    for r in seed_results:
        gap = r["bull_f1"] - r["bear_f1"]
        lines.append(
            f"| {r['seed']} | {r['bull_f1']:.4f} | {r['bear_f1']:.4f} | {gap:+.4f} | {r['macro_f1']:.4f} |"
        )
    lines += [
        "",
        f"Mean bull F1: {np.mean(bull_f1s):.4f} ± {np.std(bull_f1s):.4f}",
        f"Mean bear F1: {np.mean(bear_f1s):.4f} ± {np.std(bear_f1s):.4f}",
        f"Mean bull-bear gap: {mean_gap:+.4f}",
        "",
        "## Conclusion",
        "",
        f"Systematic asymmetry (bull>bear by >0.05 across all seeds): **{systematic}**",
        "",
        "If systematic: model likely benefits from bull FVG structural clarity vs bear.",
        "Potential causes: SPY long-term upward bias creates more bull FVG instances in",
        "training data; bull gap features (gap_bull, gap_norm_bull) may be more discriminative.",
        "",
        "## Structural Gap Size Analysis",
        "",
        "See `gap_size_distribution.html` for TP vs FP gap size comparison.",
        "See `confusion_seed*.html` for per-seed confusion matrices.",
    ]
    (ts_dir / "asymmetry.md").write_text("\n".join(lines))
    print(f"asymmetry.md saved: {ts_dir / 'asymmetry.md'}")


if __name__ == "__main__":
    main()
