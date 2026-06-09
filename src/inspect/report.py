"""report.py — Write markdown summary and per-window HTML plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

from src.inspect.outcomes import Trade, compute_trades_for_model, summarise_trades
from src.inspect.runner import InspectionResults
from src.inspect.stats import InspectionStats, ModelMetrics
from src.strategy.exits import ExitConfig

_CLASS_NAMES = ["none", "bullish", "bearish"]


def write_report(
    results: InspectionResults,
    stats: InspectionStats,
    output_dir: Path,
    top_k: int = 20,
    extra_window_indices: list[int] | None = None,
    df_full: pd.DataFrame | None = None,
    tp_rr: float = 2.0,
    exit_configs: list[ExitConfig] | None = None,
    all_seeds_data: dict | None = None,
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
    exit_configs : list[ExitConfig] | None
        Exit strategies to simulate.  Defaults to ``[ExitConfig("fixed_2r")]``
        for backward-compatibility.  First entry is the primary config.
    all_seeds_data : dict | None
        Pre-aggregated multi-seed results keyed by arch name then strategy.
        When provided, an extra section is appended to the summary report.

    Returns
    -------
    Path
        Path to the written ``summary.md`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve exit configs: default to fixed_2r (backward-compat)
    if exit_configs is None:
        exit_configs = [ExitConfig(strategy="fixed_2r", tp_rr=tp_rr)]

    # Primary config (first in list) used for single-strategy trade table + plots
    primary_config = exit_configs[0]

    trades_per_model: dict[str, list[Trade]] = {}
    if results.future_ohlcv.shape[1] > 0 and results.model_names:
        for name in results.model_names:
            trades_per_model[name] = compute_trades_for_model(
                results.windows_raw,
                results.future_ohlcv,
                results.future_timestamps,
                results.preds[name],
                probas=results.probas.get(name),
                tp_rr=primary_config.tp_rr,
                exit_config=primary_config,
            )

    # Multi-strategy comparison (only when more than one config requested)
    multi_trades: dict[str, dict[str, list[Trade]]] | None = None
    if len(exit_configs) > 1 and results.future_ohlcv.shape[1] > 0 and results.model_names:
        # strategy_name -> model_name -> trades
        multi_trades = {}
        for cfg in exit_configs:
            multi_trades[cfg.strategy] = {}
            for name in results.model_names:
                multi_trades[cfg.strategy][name] = compute_trades_for_model(
                    results.windows_raw,
                    results.future_ohlcv,
                    results.future_timestamps,
                    results.preds[name],
                    probas=results.probas.get(name),
                    tp_rr=cfg.tp_rr,
                    exit_config=cfg,
                )

    summary_path = output_dir / "summary.md"
    _write_summary(
        results, stats, summary_path, trades_per_model,
        multi_trades=multi_trades, primary_config=primary_config,
        all_seeds_data=all_seeds_data,
    )

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
    multi_trades: dict[str, dict[str, list[Trade]]] | None = None,
    primary_config: ExitConfig | None = None,
    all_seeds_data: dict | None = None,
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
        _single_realism_on = (
            primary_config is not None and primary_config.report_realism
        )
        lines.append("## Trade Outcomes (simulated)\n")
        lines.append("Each positive prediction → bracket trade: entry at next H1 open, "
                     "SL beyond opposite gap edge, TP = R×R-multiple. Win rate counts only decided trades.\n")
        if _single_realism_on:
            header = (
                "| Model | Signals | Filled | TP | SL | Undecided | Win Rate "
                "| Total R | Avg R | after_cost_total_R | median_R | n_outlier_R |"
            )
            sep = (
                "|-------|---------|--------|----|----|-----------|---------"
                "|---------|------|--------------------|---------|------------|"
            )
        else:
            header = "| Model | Signals | Filled | TP | SL | Undecided | Win Rate | Total R | Avg R |"
            sep = "|-------|---------|--------|----|----|-----------|---------|---------|------|"
        lines.append(header)
        lines.append(sep)
        for name in results.model_names:
            s = summarise_trades(
                trades_per_model.get(name, []),
                realism_config=primary_config if _single_realism_on else None,
            )
            row = (
                f"| {name} | {s['n_signals']} | {s['n_trades']} | {s['n_tp']} | {s['n_sl']} "
                f"| {s['n_undecided']} | {s['win_rate']:.3f} | "
                f"{s['total_r']:+.2f} | {s['avg_r']:+.3f} |"
            )
            if _single_realism_on:
                row += (
                    f" {s['after_cost_total_r']:+.2f} "
                    f"| {s['median_r']:+.3f} "
                    f"| {s['n_outlier_r']} |"
                )
            lines.append(row)
        lines.append("")

    # Multi-strategy comparison table
    if multi_trades:
        _realism_on = primary_config is not None and primary_config.report_realism
        lines.append("## Exit-Strategy Comparison\n")
        if _realism_on:
            lines.append(
                "**Realistic mode — pre-registered primary metric: `after_cost_total_R`** "
                "(total R after ATR min-stop floor + round-trip costs).  "
                "`winsorized_total_R`, `median_R`, `n_outlier_R` are mandatory context.  "
                "Raw `total_R` shown for continuity with the pre-realism comparison.\n"
            )
        else:
            lines.append(
                "**Pre-registered primary metric: `total_R`** (sum of realised R over all decided "
                "filled trades on the test period).  `avg_R`, `fill_rate`, and `win_rate` are "
                "mandatory context: a limit variant that barely fills can post a high `avg_R` on "
                "tiny `n_trades` — `total_R` + `fill_rate` together guard against crowning a "
                "strategy that almost never trades.\n"
            )
        lines.append(
            "> **Caveat:** limit entries modelled as filled at exact limit price on first bar "
            "touching that level (optimistic OHLC bar-level backtest assumption).  "
            "Overstates fill quality vs live execution."
            + ("" if _realism_on else "  All figures are pre-cost simulation.")
            + "\n"
        )
        for model_name in results.model_names:
            lines.append(f"### {model_name}\n")
            if _realism_on:
                hdr = (
                    "| Strategy | n_signals | n_trades | fill_rate | win_rate "
                    "| n_tp | n_sl | n_undecided | total_R | avg_R "
                    "| after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R "
                    "| winsorized_total_R | swing_fallback_rate |"
                )
                sep = (
                    "|----------|-----------|----------|-----------|----------"
                    "|------|------|-------------|---------|-------"
                    "|--------------------|-----------------|----------|-------------|"
                    "--------------------|---------------------|"
                )
            else:
                hdr = (
                    "| Strategy | n_signals | n_trades | fill_rate | win_rate "
                    "| n_tp | n_sl | n_undecided | total_R | avg_R | swing_fallback_rate |"
                )
                sep = (
                    "|----------|-----------|----------|-----------|----------"
                    "|------|------|-------------|---------|-------|---------------------|"
                )
            lines.append(hdr)
            lines.append(sep)
            for strategy_name, trades_by_model in multi_trades.items():
                s = summarise_trades(
                    trades_by_model.get(model_name, []),
                    strategy_name=strategy_name,
                    realism_config=primary_config if _realism_on else None,
                )
                fill_rate_val = s["fill_rate"]
                fill_rate_str = (
                    "N/A"
                    if isinstance(fill_rate_val, float) and fill_rate_val != fill_rate_val
                    else f"{fill_rate_val:.3f}"
                )
                row = (
                    f"| {strategy_name} "
                    f"| {s['n_signals']} "
                    f"| {s['n_trades']} "
                    f"| {fill_rate_str} "
                    f"| {s['win_rate']:.3f} "
                    f"| {s['n_tp']} "
                    f"| {s['n_sl']} "
                    f"| {s['n_undecided']} "
                    f"| {s['total_r']:+.2f} "
                    f"| {s['avg_r']:+.3f} "
                )
                if _realism_on:
                    row += (
                        f"| {s['after_cost_total_r']:+.2f} "
                        f"| {s['after_cost_avg_r']:+.3f} "
                        f"| {s['median_r']:+.3f} "
                        f"| {s['n_outlier_r']} "
                        f"| {s['winsorized_total_r']:+.2f} "
                    )
                row += f"| {s['swing_fallback_rate']:.3f} |"
                lines.append(row)
            lines.append("")

    # All-seeds aggregate section (Addition 2) — only when data is provided
    if all_seeds_data is not None:
        _write_all_seeds_section(all_seeds_data, lines)

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# All-seeds aggregate section (Addition 2)
# ---------------------------------------------------------------------------

def _write_all_seeds_section(
    all_seeds_data: dict,
    lines: list[str],
) -> None:
    """Append ## All-Seeds Aggregate section to ``lines``.

    Parameters
    ----------
    all_seeds_data : dict
        Structure produced by the all-seeds aggregation block in
        ``inspect_models.py``:

        .. code-block:: python

            {
                arch_name: {
                    "strategy_name": {
                        "mean_after_cost_total_r": float,
                        "std_after_cost_total_r": float,
                        "mean_win_rate": float,
                        "std_win_rate": float,
                        "mean_n_trades": float,
                        "per_seed": [
                            {"seed": int, "after_cost_total_r": float,
                             "win_rate": float, "n_trades": int},
                            ...
                        ],
                    },
                    ...
                },
                ...
            }
    """
    lines.append("## All-Seeds Aggregate\n")
    lines.append(
        "5-seed mean ± std across multisym checkpoints.  "
        "Pre-registered primary metric: `after_cost_total_R` (mean±std).  "
        "Per-seed rows follow each strategy block.\n"
    )

    for arch_name, strat_dict in all_seeds_data.items():
        lines.append(f"### {arch_name}\n")
        hdr = (
            "| Strategy | Seeds | after_cost_total_R (mean±std) "
            "| win_rate (mean±std) | n_trades (mean) |"
        )
        sep = (
            "|----------|-------|------------------------------"
            "--|--------------------|-----------------|"
        )
        lines.append(hdr)
        lines.append(sep)
        for strat_name, agg in strat_dict.items():
            n_seeds = len(agg["per_seed"])
            lines.append(
                f"| {strat_name} "
                f"| {n_seeds} "
                f"| {agg['mean_after_cost_total_r']:+.2f} ± {agg['std_after_cost_total_r']:.2f} "
                f"| {agg['mean_win_rate']:.3f} ± {agg['std_win_rate']:.3f} "
                f"| {agg['mean_n_trades']:.1f} |"
            )
        lines.append("")

        # Per-seed detail
        for strat_name, agg in strat_dict.items():
            lines.append(f"#### {arch_name} / {strat_name} — per-seed\n")
            lines.append("| Seed | after_cost_total_R | win_rate | n_trades |")
            lines.append("|------|-------------------|----------|----------|")
            for row in agg["per_seed"]:
                lines.append(
                    f"| {row['seed']} "
                    f"| {row['after_cost_total_r']:+.2f} "
                    f"| {row['win_rate']:.3f} "
                    f"| {row['n_trades']} |"
                )
            lines.append("")


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
