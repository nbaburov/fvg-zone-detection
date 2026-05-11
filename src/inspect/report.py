"""report.py — Write markdown summary and per-window HTML plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

from src.inspect.outcomes import Trade, compute_trades_for_model, summarise_trades
from src.inspect.runner import InspectionResults
from src.inspect.stats import InspectionStats, ModelMetrics

_CLASS_NAMES = ["none", "bullish", "bearish"]


def write_report(
    results: InspectionResults,
    stats: InspectionStats,
    output_dir: Path,
    top_k: int = 20,
    extra_window_indices: list[int] | None = None,
    df_full: pd.DataFrame | None = None,
    tp_rr: float = 2.0,
) -> Path:
    """Write summary markdown and top-K disagreement window HTML files.

    Parameters
    ----------
    results : InspectionResults
    stats : InspectionStats
    output_dir : Path
        Timestamped output directory (caller is responsible for the timestamp).
        Created if it does not exist.
    top_k : int
        Maximum number of disagreement windows to render as HTML.
    extra_window_indices : list[int] | None
        Additional window indices to render regardless of disagreement rank.

    Returns
    -------
    Path
        Path to the written ``summary.md`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_per_model: dict[str, list[Trade]] = {}
    if results.future_ohlcv.shape[1] > 0 and results.model_names:
        for name in results.model_names:
            trades_per_model[name] = compute_trades_for_model(
                results.windows_raw,
                results.future_ohlcv,
                results.future_timestamps,
                results.preds[name],
                tp_rr=tp_rr,
            )

    summary_path = output_dir / "summary.md"
    _write_summary(results, stats, summary_path, trades_per_model)

    if results.n > 0 and results.model_names:
        plots_dir = output_dir / "plots"
        _render_window_plots(results, stats, plots_dir, top_k, extra_window_indices or [])

        if trades_per_model and df_full is not None:
            _render_model_timelines(df_full, trades_per_model, plots_dir)

    return summary_path


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _write_summary(
    results: InspectionResults,
    stats: InspectionStats,
    path: Path,
    trades_per_model: dict[str, list[Trade]] | None = None,
) -> None:
    lines: list[str] = []

    lines.append("# Model Inspection Summary\n")

    if results.n == 0:
        lines.append("**No windows in range.** The selected slice produced 0 valid windows.\n")
        path.write_text("\n".join(lines))
        return

    lines.append(f"- Windows: {results.n}")
    lines.append(f"- FVG-positive windows: {stats.n_positive} "
                 f"({100 * stats.n_positive / results.n:.1f}%)")
    if stats.warning_no_positives:
        lines.append("\n> **Warning:** No FVG-positive labels in this slice. "
                     "F1 scores are 0.0 by definition.\n")
    lines.append(f"- Models: {', '.join(results.model_names)}\n")

    # Per-model metrics table
    lines.append("## Per-model Metrics\n")
    header = "| Model | F1 Bull | F1 Bear | F1 FVG Macro | F1 Binary FVG |"
    sep = "|-------|---------|---------|--------------|---------------|"
    lines.append(header)
    lines.append(sep)
    for name in results.model_names:
        m = stats.per_model[name]
        lines.append(
            f"| {name} | {m.f1_bull:.4f} | {m.f1_bear:.4f} "
            f"| {m.f1_fvg_macro:.4f} | {m.f1_binary_fvg:.4f} |"
        )
    lines.append("")

    # Precision / recall
    lines.append("## Precision / Recall (FVG classes)\n")
    header2 = "| Model | Prec Bull | Rec Bull | Prec Bear | Rec Bear |"
    sep2 = "|-------|-----------|----------|-----------|----------|"
    lines.append(header2)
    lines.append(sep2)
    for name in results.model_names:
        m = stats.per_model[name]
        lines.append(
            f"| {name} | {m.precision_bull:.4f} | {m.recall_bull:.4f} "
            f"| {m.precision_bear:.4f} | {m.recall_bear:.4f} |"
        )
    lines.append("")

    # Confusion matrices
    lines.append("## Confusion Matrices\n")
    for name in results.model_names:
        m = stats.per_model[name]
        lines.append(f"### {name}\n")
        lines.append("```")
        lines.append(f"{'':12s}  pred_none  pred_bull  pred_bear")
        for i, row_name in enumerate(_CLASS_NAMES):
            row = m.confusion[i]
            lines.append(
                f"true_{row_name:<7s}  {row[0]:9d}  {row[1]:9d}  {row[2]:9d}"
            )
        lines.append("```\n")

    # Agreement matrix
    if len(results.model_names) >= 2:
        lines.append("## Agreement Matrix\n")
        lines.append("Fraction of windows where each pair of models predicts the same class.\n")
        col_header = "| | " + " | ".join(results.model_names) + " |"
        col_sep = "|---|" + "---|" * len(results.model_names)
        lines.append(col_header)
        lines.append(col_sep)
        for i, name_i in enumerate(results.model_names):
            row_vals = " | ".join(
                f"{stats.agreement_matrix[i, j]:.3f}"
                for j in range(len(results.model_names))
            )
            lines.append(f"| **{name_i}** | {row_vals} |")
        lines.append("")

    # Top-K disagreement windows
    if stats.disagreement_index.size > 0:
        top_k_actual = min(20, results.n)
        top_indices = np.argsort(stats.disagreement_index)[::-1][:top_k_actual]
        lines.append("## Top Disagreement Windows\n")
        lines.append("| Rank | Window | Timestamp | Disagreement | GT Label |"
                     + "".join(f" {m} |" for m in results.model_names))
        lines.append("|------|--------|-----------|-------------|----------|"
                     + "".join("---|" for _ in results.model_names))
        for rank, idx in enumerate(top_indices, 1):
            ts = results.timestamps[idx]
            gt = _CLASS_NAMES[int(results.labels[idx])]
            disag = stats.disagreement_index[idx]
            pred_cells = "".join(
                f" {_CLASS_NAMES[int(results.preds[m][idx])]} |"
                for m in results.model_names
            )
            lines.append(
                f"| {rank} | {idx} | {ts.strftime('%Y-%m-%d %H:%M')} "
                f"| {disag:.3f} | {gt} |{pred_cells}"
            )
        lines.append("")

    # Trade outcomes (only if lookahead enabled)
    if trades_per_model:
        lines.append("## Trade Outcomes (simulated)\n")
        lines.append("Each positive prediction → bracket trade: entry at next H1 open, "
                     "SL beyond opposite gap edge, TP = R×R-multiple. Win rate counts only decided trades.\n")
        header = "| Model | Trades | TP | SL | Undecided | Win Rate | Total R | Avg R |"
        sep = "|-------|--------|----|----|-----------|----------|---------|-------|"
        lines.append(header)
        lines.append(sep)
        for name in results.model_names:
            s = summarise_trades(trades_per_model.get(name, []))
            lines.append(
                f"| {name} | {s['n_trades']} | {s['n_tp']} | {s['n_sl']} "
                f"| {s['n_undecided']} | {s['win_rate']:.3f} | "
                f"{s['total_r']:+.2f} | {s['avg_r']:+.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Per-model timeline plots
# ---------------------------------------------------------------------------

def _render_model_timelines(
    df_full: pd.DataFrame,
    trades_per_model: dict[str, list[Trade]],
    plots_dir: Path,
) -> None:
    from src.inspect.viz import plot_model_timeline
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name, trades in trades_per_model.items():
        out = plots_dir / f"timeline_{name}.html"
        plot_model_timeline(df_full, trades, name, out)


# ---------------------------------------------------------------------------
# Window plots
# ---------------------------------------------------------------------------

def _render_window_plots(
    results: InspectionResults,
    stats: InspectionStats,
    plots_dir: Path,
    top_k: int,
    extra_indices: list[int],
) -> None:
    from src.inspect.viz import plot_window  # local import — plotly optional

    # Collect indices to render: top-K disagreement + extras, deduplicated
    if stats.disagreement_index.size > 0:
        sorted_indices = list(
            np.argsort(stats.disagreement_index)[::-1][:top_k]
        )
    else:
        sorted_indices = []

    all_indices = list(dict.fromkeys(sorted_indices + extra_indices))
    all_indices = [i for i in all_indices if 0 <= i < results.n]

    for idx in all_indices:
        model_preds: dict[str, tuple[int, float]] = {}
        for name in results.model_names:
            pred_cls = int(results.preds[name][idx])
            confidence = float(results.probas[name][idx].max())
            model_preds[name] = (pred_cls, confidence)

        plot_window(
            raw_ohlcv=results.windows_raw[idx],
            timestamp=results.timestamps[idx],
            gold_label=int(results.labels[idx]),
            model_preds=model_preds,
            output_path=plots_dir / f"window_{idx:06d}.html",
        )
