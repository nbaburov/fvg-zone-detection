"""inspect_models.py — CLI for model inspection and live comparison.

Usage
-----
    python scripts/inspect_models.py --help
    python scripts/inspect_models.py --list
    python scripts/inspect_models.py --models lstm xgboost --dataset test --top-k 30
    python scripts/inspect_models.py --models lstm --dataset data/processed/spy_h1_val.parquet \\
        --start 2023-06 --end 2023-09

    # Per-model checkpoint override (name:path form — mirrors paper_trade.py):
    python scripts/inspect_models.py --dataset test --models \\
        cnn_lstm:checkpoints/cnn_lstm_multisym/cnn_lstm/cnn_lstm_seed0.pt \\
        lstm:checkpoints/lstm_multisym/lstm/lstm_seed0.pt \\
        transformer:checkpoints/transformer_multisym/transformer/transformer_seed0.pt \\
        xgboost:checkpoints/xgb_multisym/xgboost/xgb_seed42.ubj

    Bare names (e.g. --models lstm xgboost) use --checkpoint-dir + default seed,
    exactly as before.  Mixed forms are allowed.

Output is written to reports/inspect/<YYYY-MM-DD_HHMMSS>/.

Notes
-----
- Inference is always live (no precomputed caches).
- XGBoost + PyTorch coexistence is stable on Python 3.12. If you upgrade to
  Python 3.14+, XGBoost segfaults when loaded alongside Torch. Workaround:
  run XGBoost in a subprocess using multiprocessing start method 'spawn'
  with arrays communicated via shared memory or temp files.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="inspect_models",
        description="SMC model inspection / live comparison tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME_OR_NAME:PATH",
        default=None,
        help=(
            "Adapters to load. Each token is either a bare name (e.g. lstm) "
            "or name:path pointing at a specific checkpoint file "
            "(e.g. cnn_lstm:checkpoints/cnn_lstm_multisym/cnn_lstm/cnn_lstm_seed0.pt). "
            "Bare names use --checkpoint-dir + the adapter's default seed. "
            "Default: all discovered adapters."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="test",
        metavar="SPLIT_OR_PATH",
        help=(
            "Dataset to run on. Use 'train', 'val', or 'test' for standard splits, "
            "or supply an absolute/relative path to a .parquet file."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="DATE",
        help="Inclusive start date for filtering (e.g. 2024-01-01 or 2024-01).",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="DATE",
        help="Inclusive end date for filtering (e.g. 2024-03-31 or 2024-03).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help="Number of top-disagreement windows to render as HTML plots. Default: 20.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        metavar="DIR",
        dest="checkpoint_dir",
        help="Directory containing model checkpoints. Default: checkpoints/.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/inspect",
        metavar="DIR",
        dest="output_dir",
        help="Base output directory. A timestamped subdir is created. Default: reports/inspect/.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available adapter names and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Numpy/torch seed for any sampling. Default: 42.",
    )
    parser.add_argument(
        "--lookahead-bars",
        type=int,
        default=20,
        help="H1 bars to walk after each positive prediction to determine "
             "TP/SL/undecided outcome. 0 disables outcome simulation. Default: 20.",
    )
    parser.add_argument(
        "--tp-rr",
        type=float,
        default=2.0,
        help="Take-profit reward multiple (TP = entry + R * tp_rr). Default: 2.0.",
    )
    parser.add_argument(
        "--exit-strategy",
        default="fixed_2r",
        choices=["fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"],
        dest="exit_strategy",
        help=(
            "Exit-strategy variant to simulate.  Default: fixed_2r "
            "(market entry at N+2 open, gap-edge SL, 2R TP — backward-compatible). "
            "Ignored when --all-exit-strategies is set."
        ),
    )
    parser.add_argument(
        "--all-exit-strategies",
        action="store_true",
        dest="all_exit_strategies",
        help=(
            "Run all 4 exit-strategy variants and emit a comparison table.  "
            "Pre-registered primary metric = total_R.  "
            "Overrides --exit-strategy."
        ),
    )
    parser.add_argument(
        "--realistic",
        action="store_true",
        help=(
            "Enable all three realism guards with conservative defaults: "
            "min_stop_atr_k=0.5, slippage_ticks=1, commission_per_share=0.005, "
            "report_realism=True.  Pre-registered primary metric = after_cost_total_R.  "
            "Use --min-stop-atr-k to override the floor multiplier."
        ),
    )
    parser.add_argument(
        "--min-stop-atr-k",
        type=float,
        default=None,
        dest="min_stop_atr_k",
        metavar="FLOAT",
        help=(
            "Override the ATR floor multiplier used by --realistic.  "
            "Ignored unless --realistic is set.  Example: --min-stop-atr-k 1.0."
        ),
    )
    parser.add_argument(
        "--min-stop-atr-k-sweep",
        action="store_true",
        dest="min_stop_atr_k_sweep",
        help=(
            "Run the exit-strategy comparison at k=0.25, 0.5, 1.0 and emit "
            "three side-by-side result blocks in one report.  Implies "
            "--realistic and --all-exit-strategies.  Headline k=0.5.  "
            "Writes its own report and exits; run separately from --all-seeds "
            "and other sweeps.  "
            "Only --min-stop-atr-k is varied; --confidence-threshold and "
            "--fill-mode are held at their defaults and are not applied."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        dest="confidence_threshold",
        metavar="FLOAT",
        help=(
            "Skip predictions where max(P_bull, P_bear) < FLOAT.  "
            "Default 0.0 = no filter (backward-compatible).  "
            "NOTE: raw softmax confidence is NOT calibrated — use as a "
            "relative conviction filter only."
        ),
    )
    parser.add_argument(
        "--confidence-sweep",
        action="store_true",
        dest="confidence_sweep",
        help=(
            "Run one-axis confidence-threshold sweep at {0.5, 0.6, 0.7, 0.8} "
            "with all other params at headline (k=0.5, optimistic fill, "
            "base cost, tp_rr=2.0).  Implies --realistic --all-exit-strategies.  "
            "Writes confidence_sweep.md to the run dir.  "
            "Writes its own report and exits; run separately from --all-seeds "
            "and other sweeps.  "
            "Only --confidence-threshold is varied; --fill-mode and "
            "--min-stop-atr-k are held at their defaults and are not applied."
        ),
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        dest="all_seeds",
        help=(
            "Load all 5 multisym checkpoints per arch (seeds 0/17/42/123/2024) "
            "and report mean±std after_cost_total_R across seeds.  "
            "Archs with no multisym checkpoints are skipped with a warning.  "
            "Implies --realistic --all-exit-strategies."
        ),
    )
    parser.add_argument(
        "--fill-mode",
        default="optimistic",
        choices=["optimistic", "conservative"],
        dest="fill_mode",
        help=(
            "Fill model for limit entries.  "
            "'optimistic' (default): bar.low <= limit fills (bull).  "
            "'conservative': bar.close <= limit fills (bull) — fewer fills, "
            "more realistic."
        ),
    )
    parser.add_argument(
        "--sensitivity-sweep",
        action="store_true",
        dest="sensitivity_sweep",
        help=(
            "Run one-axis-at-a-time sensitivity sweep over cost tiers, "
            "tp_rr values, and fill modes.  Implies --realistic "
            "--all-exit-strategies.  Writes sensitivity_sweep.md.  "
            "Writes its own report and exits; run separately from --all-seeds "
            "and other sweeps.  "
            "Each sub-axis varies only its own parameter; --confidence-threshold "
            "and other realism flags beyond --realistic defaults are not applied."
        ),
    )
    return parser.parse_args(argv)


def _resolve_dataset_path(dataset: str) -> Path:
    """Resolve dataset argument to a parquet file path."""
    split_map = {
        "train": "data/processed/spy_h1_train.parquet",
        "val": "data/processed/spy_h1_val.parquet",
        "test": "data/processed/spy_h1_test.parquet",
    }
    if dataset in split_map:
        return _ROOT / split_map[dataset]
    p = Path(dataset)
    if not p.is_absolute():
        p = _ROOT / p
    return p


_ALL_SEEDS = [0, 17, 42, 123, 2024]

# Maps arch_name -> (multisym_subdir, filename_template)
# xgboost uses .ubj; all DL archs use .pt
_MULTISYM_DIRS: dict[str, tuple[str, str]] = {
    "lstm": ("lstm_multisym/lstm", "lstm_seed{seed}.pt"),
    "cnn_lstm": ("cnn_lstm_multisym/cnn_lstm", "cnn_lstm_seed{seed}.pt"),
    "transformer": ("transformer_multisym/transformer", "transformer_seed{seed}.pt"),
    "xgboost": ("xgb_multisym/xgboost", "xgb_seed{seed}.ubj"),
    "xlstm": ("xlstm_multisym/xlstm", "xlstm_seed{seed}.pt"),
}


def _resolve_multisym_checkpoints(
    checkpoint_dir: Path,
) -> dict[str, list[tuple[int, Path]]]:
    """Resolve available multisym checkpoints per arch.

    Returns a mapping arch_name -> list of (seed, path) for checkpoints
    that exist on disk.  Archs with zero found files are omitted; caller
    should warn and skip them.
    """
    result: dict[str, list[tuple[int, Path]]] = {}
    for arch, (subdir, tmpl) in _MULTISYM_DIRS.items():
        found: list[tuple[int, Path]] = []
        for seed in _ALL_SEEDS:
            fname = tmpl.format(seed=seed)
            p = checkpoint_dir / subdir / fname
            if p.exists():
                found.append((seed, p))
        if found:
            result[arch] = found
    return result


def _aggregate_seed_summaries(
    seed_summaries: list[tuple[int, dict]],
) -> dict:
    """Aggregate per-seed summarise_trades dicts into mean±std.

    Parameters
    ----------
    seed_summaries : list[tuple[int, dict]]
        List of (seed, summary_dict) where summary_dict comes from
        ``summarise_trades(..., report_realism=True)``.

    Returns
    -------
    dict with keys: mean_after_cost_total_r, std_after_cost_total_r,
    mean_win_rate, std_win_rate, mean_n_trades, per_seed.

    Requires per-seed summaries produced with report_realism=True
    (after_cost_total_r present); falls back to 0.0 if absent.
    """
    import numpy as np

    after_costs = [s.get("after_cost_total_r", 0.0) for _, s in seed_summaries]
    win_rates = [s["win_rate"] for _, s in seed_summaries]
    n_trades = [s["n_trades"] for _, s in seed_summaries]
    per_seed = [
        {
            "seed": seed,
            "after_cost_total_r": s.get("after_cost_total_r", 0.0),
            "win_rate": s["win_rate"],
            "n_trades": s["n_trades"],
        }
        for seed, s in seed_summaries
    ]
    return {
        "mean_after_cost_total_r": float(np.mean(after_costs)),
        "std_after_cost_total_r": float(np.std(after_costs, ddof=1)) if len(after_costs) > 1 else 0.0,
        "mean_win_rate": float(np.mean(win_rates)),
        "std_win_rate": float(np.std(win_rates, ddof=1)) if len(win_rates) > 1 else 0.0,
        "mean_n_trades": float(np.mean(n_trades)),
        "per_seed": per_seed,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from src.inspect.registry import list_available, load_adapters

    if args.list:
        names = list_available()
        if not names:
            print("No adapters discovered.")
        else:
            print("Available adapters:")
            for name in names:
                print(f"  {name}")
        return 0

    import numpy as np
    import pandas as pd

    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    dataset_path = _resolve_dataset_path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    # Apply date filter
    if args.start or args.end:
        if args.start:
            df = df[df.index >= pd.Timestamp(args.start, tz=df.index.tz)]
        if args.end:
            # Make end inclusive by going to end of the day/month
            end_ts = pd.Timestamp(args.end, tz=df.index.tz)
            # If only year-month supplied, go to end of month
            if len(args.end) <= 7:
                end_ts = end_ts + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
            else:
                end_ts = end_ts + pd.Timedelta(hours=23, minutes=59)
            df = df[df.index <= end_ts]

    print(f"Bars in slice: {len(df)}")

    # -----------------------------------------------------------------------
    # Check for label column; add if missing
    # -----------------------------------------------------------------------
    if "label" not in df.columns:
        print("No 'label' column found in dataset — computing labels via ValidFVGLabeller...")
        from src.data.labels.valid_fvg import ValidFVGLabeller
        labeller = ValidFVGLabeller()
        raw_labels = labeller.label(df)
        df = df.copy()
        df["label"] = labeller.encode(raw_labels)
    else:
        from src.data.labels.valid_fvg import ValidFVGLabeller
        labeller = ValidFVGLabeller()

    # -----------------------------------------------------------------------
    # Load adapters
    # -----------------------------------------------------------------------
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = _ROOT / checkpoint_dir

    if args.models is None:
        model_names = list_available()
        if not model_names:
            print("ERROR: No adapters discovered. Check src/inspect/adapters/.", file=sys.stderr)
            return 1
        print(f"No --models specified; using all discovered: {model_names}")
        checkpoint_path_overrides: dict[str, str] = {}
    else:
        model_names = []
        checkpoint_path_overrides = {}
        for token in args.models:
            name = token.split(":", 1)[0] if ":" in token else token
            if name in model_names:
                print(
                    f"ERROR: model '{name}' specified more than once in --models. "
                    "Each adapter may appear at most once.",
                    file=sys.stderr,
                )
                return 1
            if ":" in token:
                _, cp_path = token.split(":", 1)
                # Resolve relative paths from project root
                p = Path(cp_path)
                if not p.is_absolute():
                    p = _ROOT / p
                model_names.append(name)
                checkpoint_path_overrides[name] = str(p)
            else:
                model_names.append(name)

    print(f"Loading adapters: {model_names}")
    if checkpoint_path_overrides:
        print(f"Checkpoint overrides: {checkpoint_path_overrides}")

    # When --all-seeds is set, skip adapters whose single-seed checkpoint is
    # missing rather than aborting — the all-seeds block will use multisym
    # checkpoints directly.
    if args.all_seeds:
        adapters = []
        skipped_for_all_seeds: list[str] = []
        for name in model_names:
            try:
                loaded = load_adapters(
                    [name], checkpoint_dir,
                    checkpoint_paths={name: checkpoint_path_overrides[name]}
                    if name in checkpoint_path_overrides else {},
                )
                adapters.extend(loaded)
            except (KeyError, FileNotFoundError) as exc:
                print(
                    f"WARNING: --all-seeds mode: skipping single-seed load for '{name}' "
                    f"({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
                skipped_for_all_seeds.append(name)
        if not adapters and not model_names:
            print("ERROR: No adapters loaded.", file=sys.stderr)
            return 1
    else:
        try:
            adapters = load_adapters(model_names, checkpoint_dir, checkpoint_paths=checkpoint_path_overrides)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            print(f"ERROR: Checkpoint not found — {exc}", file=sys.stderr)
            return 1

    # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------
    from src.inspect.runner import run

    print("Running inference...")
    results = run(
        adapters=adapters, df_slice=df, labeller=labeller,
        lookahead_bars=args.lookahead_bars,
    )

    print(f"Windows built: {results.n}")
    if results.n == 0:
        print("WARNING: 0 windows produced. Slice too short or all cross-session.", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Compute stats
    # -----------------------------------------------------------------------
    from src.inspect.stats import compute_stats

    stats = compute_stats(results)

    if stats.warning_no_positives:
        print("WARNING: No FVG-positive labels in this slice. F1 scores are 0.")

    # Print quick summary to stdout
    print("\n--- Metrics ---")
    for name in results.model_names:
        m = stats.per_model[name]
        print(
            f"  {name:12s}  F1_bull={m.f1_bull:.4f}  F1_bear={m.f1_bear:.4f}"
            f"  F1_fvg_macro={m.f1_fvg_macro:.4f}  F1_binary={m.f1_binary_fvg:.4f}"
        )

    if len(results.model_names) >= 2:
        overall_agree = float(stats.agreement_matrix[0, 1]) if stats.agreement_matrix.size > 1 else 0.0
        print(f"\nOverall agreement (all pairs): {overall_agree:.3f}")

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    from src.inspect.report import write_report

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _ROOT / output_dir
    run_dir = output_dir / timestamp_str

    import dataclasses
    from src.strategy.exits import ExitConfig

    # --min-stop-atr-k-sweep / --confidence-sweep / --all-seeds /
    # --sensitivity-sweep all imply --realistic + --all-exit-strategies
    do_sweep = args.min_stop_atr_k_sweep
    do_conf_sweep = args.confidence_sweep
    do_all_seeds = args.all_seeds
    do_sensitivity = args.sensitivity_sweep
    do_realistic = args.realistic or do_sweep or do_conf_sweep or do_all_seeds or do_sensitivity
    do_all = args.all_exit_strategies or do_sweep or do_conf_sweep or do_all_seeds or do_sensitivity

    exit_strategies = (
        ["fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"]
        if do_all
        else [args.exit_strategy]
    )
    exit_configs = [
        ExitConfig(strategy=s, tp_rr=args.tp_rr) for s in exit_strategies
    ]

    if do_realistic:
        # Headline k=0.5; --min-stop-atr-k overrides
        k_value = args.min_stop_atr_k if args.min_stop_atr_k is not None else 0.5
        exit_configs = [
            dataclasses.replace(
                cfg,
                min_stop_atr_k=k_value,
                slippage_ticks=1,
                commission_per_share=0.005,
                report_realism=True,
                confidence_threshold=args.confidence_threshold,
                fill_mode=args.fill_mode,
            )
            for cfg in exit_configs
        ]
    else:
        # Not realistic mode — still apply confidence_threshold and fill_mode
        if args.confidence_threshold != 0.0 or args.fill_mode != "optimistic":
            exit_configs = [
                dataclasses.replace(
                    cfg,
                    confidence_threshold=args.confidence_threshold,
                    fill_mode=args.fill_mode,
                )
                for cfg in exit_configs
            ]

    if do_sweep:
        # Built-in sweep: run k=0.25, 0.5, 1.0 and emit three result blocks
        sweep_ks = [0.25, 0.5, 1.0]
        print(f"\nWriting sweep report to: {run_dir}")

        sweep_lines: list[str] = [
            "# Exit-Strategy Comparison — ATR Floor Sensitivity Sweep\n",
            "**Pre-registered primary metric: `after_cost_total_R`**  "
            "(after ATR min-stop floor + round-trip costs)\n",
            "Context columns: `winsorized_total_R`, `median_R`, `n_outlier_R`  "
            "Headline k = **0.5**; sweep demonstrates ranking robustness.\n",
        ]

        for k in sweep_ks:
            sweep_configs = [
                dataclasses.replace(
                    ExitConfig(strategy=s, tp_rr=args.tp_rr),
                    min_stop_atr_k=k,
                    slippage_ticks=1,
                    commission_per_share=0.005,
                    report_realism=True,
                )
                for s in exit_strategies
            ]

            from src.inspect.outcomes import compute_trades_for_model, summarise_trades

            sweep_lines.append(f"\n---\n\n## k = {k}\n")
            for model_name in results.model_names:
                sweep_lines.append(f"### {model_name}\n")
                hdr = (
                    "| Strategy | n_trades | win_rate | total_R | after_cost_total_R "
                    "| winsorized_total_R | median_R | n_outlier_R |"
                )
                sep = (
                    "|----------|----------|----------|---------|--------------------"
                    "|--------------------|----------|-------------|"
                )
                sweep_lines.append(hdr)
                sweep_lines.append(sep)
                for cfg in sweep_configs:
                    trades = compute_trades_for_model(
                        results.windows_raw,
                        results.future_ohlcv,
                        results.future_timestamps,
                        results.preds[model_name],
                        probas=results.probas.get(model_name),
                        tp_rr=cfg.tp_rr,
                        exit_config=cfg,
                    )
                    s = summarise_trades(trades, strategy_name=cfg.strategy, realism_config=cfg)
                    sweep_lines.append(
                        f"| {cfg.strategy} "
                        f"| {s['n_trades']} "
                        f"| {s['win_rate']:.3f} "
                        f"| {s['total_r']:+.2f} "
                        f"| {s['after_cost_total_r']:+.2f} "
                        f"| {s['winsorized_total_r']:+.2f} "
                        f"| {s['median_r']:+.3f} "
                        f"| {s['n_outlier_r']} |"
                    )
                sweep_lines.append("")

        run_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = run_dir / "sweep_summary.md"
        sweep_path.write_text("\n".join(sweep_lines))
        print(f"Sweep summary: {sweep_path}")
        print("Done.")
        return 0

    # -----------------------------------------------------------------------
    # --confidence-sweep: one-axis threshold sensitivity
    # -----------------------------------------------------------------------
    if do_conf_sweep:
        from src.inspect.outcomes import compute_trades_for_model, summarise_trades

        conf_thresholds = [0.5, 0.6, 0.7, 0.8]
        k_value = args.min_stop_atr_k if args.min_stop_atr_k is not None else 0.5
        # Print confidence distribution for each model (D9 honesty)
        print("\n--- Confidence Distribution (softmax max(P_bull, P_bear)) ---")
        for model_name in results.model_names:
            proba_m = results.probas.get(model_name)
            if proba_m is None:
                continue
            preds_m = results.preds[model_name]
            pos_mask = preds_m != 0
            if pos_mask.sum() == 0:
                print(f"  {model_name}: no positive predictions")
                continue
            conf_vals = np.maximum(proba_m[pos_mask, 1], proba_m[pos_mask, 2])
            print(
                f"  {model_name}: n_pos={pos_mask.sum()}  "
                f"p25={np.percentile(conf_vals, 25):.3f}  "
                f"p50={np.percentile(conf_vals, 50):.3f}  "
                f"p75={np.percentile(conf_vals, 75):.3f}  "
                f"p90={np.percentile(conf_vals, 90):.3f}"
            )

        conf_lines: list[str] = [
            "# Confidence-Threshold Sensitivity Sweep\n",
            "**NOTE: softmax confidence is NOT calibrated** — "
            "`max(P_bull, P_bear)` is a relative conviction score, "
            "NOT a probability of correctness.  "
            "Threshold 0.7 means 'raw softmax >= 0.7', not '70% likely correct'.  "
            "With ~97% none-class imbalance, directional probas may rarely "
            "exceed ~0.7; high thresholds may yield few or zero eligible trades.\n",
            "Headline config: k=0.5, fill=optimistic, cost=base (1-tick + 0.005), "
            "tp_rr=2.0.  One axis varied at a time.\n",
        ]

        for model_name in results.model_names:
            conf_lines.append(f"## {model_name}\n")
            conf_lines.append(
                "| threshold | n_signals | n_eligible | n_trades "
                "| win_rate | after_cost_total_R | median_R |"
            )
            conf_lines.append(
                "|-----------|-----------|------------|----------"
                "|----------|-------------------|---------|"
            )
            for threshold in conf_thresholds:
                conf_cfg = ExitConfig(
                    strategy="fixed_2r",
                    tp_rr=args.tp_rr,
                    min_stop_atr_k=k_value,
                    slippage_ticks=1,
                    commission_per_share=0.005,
                    report_realism=True,
                    confidence_threshold=threshold,
                )
                preds_m = results.preds[model_name]
                n_signals = int((preds_m != 0).sum())
                trades = compute_trades_for_model(
                    results.windows_raw,
                    results.future_ohlcv,
                    results.future_timestamps,
                    preds_m,
                    probas=results.probas.get(model_name),
                    tp_rr=conf_cfg.tp_rr,
                    exit_config=conf_cfg,
                )
                # fixed_2r is market-entry: every signal that passes the
                # confidence filter enters immediately, so n_eligible cleanly
                # equals "signals that survived the filter".  A limit strategy
                # would conflate filter rejection with unfilled limit orders here.
                n_eligible = len(trades)
                s = summarise_trades(trades, strategy_name="fixed_2r", realism_config=conf_cfg)
                conf_lines.append(
                    f"| {threshold} "
                    f"| {n_signals} "
                    f"| {n_eligible} "
                    f"| {s['n_trades']} "
                    f"| {s['win_rate']:.3f} "
                    f"| {s['after_cost_total_r']:+.2f} "
                    f"| {s['median_r']:+.3f} |"
                )
            conf_lines.append("")

        run_dir.mkdir(parents=True, exist_ok=True)
        conf_sweep_path = run_dir / "confidence_sweep.md"
        conf_sweep_path.write_text("\n".join(conf_lines))
        print(f"\nConfidence sweep: {conf_sweep_path}")
        print("Done.")
        return 0

    # -----------------------------------------------------------------------
    # --sensitivity-sweep: one-axis cost/fill/tp_rr sensitivity
    # -----------------------------------------------------------------------
    if do_sensitivity:
        from src.inspect.outcomes import compute_trades_for_model, summarise_trades

        k_value = args.min_stop_atr_k if args.min_stop_atr_k is not None else 0.5
        HEADLINE_THRESHOLD = 0.5
        HEADLINE_TP_RR = args.tp_rr  # default 2.0

        # Cost tiers for sub-axis A
        cost_tiers = [
            ("cheap",     0, 0.0),
            ("base",      1, 0.005),
            ("expensive", 2, 0.010),
        ]
        # tp_rr values for sub-axis B
        tp_rr_values = [1.0, 2.0, 3.0]
        # fill modes for sub-axis C
        fill_modes = ["optimistic", "conservative"]

        sens_lines: list[str] = [
            "# Sensitivity Sweep — Cost / Fill / TP-RR\n",
            "One-axis-at-a-time sensitivity around headline config: "
            "k=0.5, fill=optimistic, cost=base (1-tick+0.005), "
            "tp_rr=2.0, threshold=0.5.\n",
            "**NOTE: softmax confidence_threshold is NOT calibrated.**\n",
        ]

        def _sens_table_header() -> tuple[str, str]:
            hdr = (
                "| config | strategy | n_trades | win_rate | after_cost_total_R "
                "| winsorized_total_R | median_R |"
            )
            sep = (
                "|--------|----------|----------|----------|--------------------"
                "|--------------------|---------|"
            )
            return hdr, sep

        # Sub-axis A: cost tiers
        sens_lines.append(
            "\n---\n\n## Sensitivity — Cost Tiers  "
            "(fill=optimistic, tp_rr=2.0, threshold=0.5, k=0.5)\n"
        )
        for model_name in results.model_names:
            sens_lines.append(f"### {model_name}\n")
            hdr, sep = _sens_table_header()
            sens_lines.append(hdr)
            sens_lines.append(sep)
            for tier_label, slip_ticks, comm in cost_tiers:
                for strat in exit_strategies:
                    cfg = ExitConfig(
                        strategy=strat,
                        tp_rr=HEADLINE_TP_RR,
                        min_stop_atr_k=k_value,
                        slippage_ticks=slip_ticks,
                        commission_per_share=comm,
                        report_realism=True,
                        confidence_threshold=HEADLINE_THRESHOLD,
                        fill_mode="optimistic",
                    )
                    trades = compute_trades_for_model(
                        results.windows_raw, results.future_ohlcv,
                        results.future_timestamps, results.preds[model_name],
                        probas=results.probas.get(model_name),
                        tp_rr=cfg.tp_rr, exit_config=cfg,
                    )
                    s = summarise_trades(trades, strategy_name=strat, realism_config=cfg)
                    sens_lines.append(
                        f"| {tier_label} | {strat} | {s['n_trades']} "
                        f"| {s['win_rate']:.3f} | {s['after_cost_total_r']:+.2f} "
                        f"| {s['winsorized_total_r']:+.2f} | {s['median_r']:+.3f} |"
                    )
            sens_lines.append("")

        # Sub-axis B: tp_rr
        sens_lines.append(
            "\n---\n\n## Sensitivity — TP Reward Multiple  "
            "(fill=optimistic, cost=base, threshold=0.5, k=0.5)\n"
        )
        for model_name in results.model_names:
            sens_lines.append(f"### {model_name}\n")
            hdr, sep = _sens_table_header()
            sens_lines.append(hdr)
            sens_lines.append(sep)
            for tp_val in tp_rr_values:
                for strat in exit_strategies:
                    cfg = ExitConfig(
                        strategy=strat,
                        tp_rr=tp_val,
                        min_stop_atr_k=k_value,
                        slippage_ticks=1,
                        commission_per_share=0.005,
                        report_realism=True,
                        confidence_threshold=HEADLINE_THRESHOLD,
                        fill_mode="optimistic",
                    )
                    trades = compute_trades_for_model(
                        results.windows_raw, results.future_ohlcv,
                        results.future_timestamps, results.preds[model_name],
                        probas=results.probas.get(model_name),
                        tp_rr=cfg.tp_rr, exit_config=cfg,
                    )
                    s = summarise_trades(trades, strategy_name=strat, realism_config=cfg)
                    sens_lines.append(
                        f"| tp_rr={tp_val} | {strat} | {s['n_trades']} "
                        f"| {s['win_rate']:.3f} | {s['after_cost_total_r']:+.2f} "
                        f"| {s['winsorized_total_r']:+.2f} | {s['median_r']:+.3f} |"
                    )
            sens_lines.append("")

        # Sub-axis C: fill mode (limit strategies only; V1 is market entry)
        limit_strategies = ["ict_iofed", "ce_50pct", "tradinglab"]
        sens_lines.append(
            "\n---\n\n## Sensitivity — Fill Mode  "
            "(cost=base, tp_rr=2.0, threshold=0.5, k=0.5)  "
            "Limit strategies only (V2/V3/V4); V1 fixed_2r is market entry — N/A.\n"
        )
        for model_name in results.model_names:
            sens_lines.append(f"### {model_name}\n")
            fill_hdr = (
                "| fill_mode | strategy | n_signals | n_trades | fill_rate "
                "| win_rate | after_cost_total_R |"
            )
            fill_sep = (
                "|-----------|----------|-----------|----------|----------"
                "|----------|-------------------|"
            )
            sens_lines.append(fill_hdr)
            sens_lines.append(fill_sep)
            for fm in fill_modes:
                for strat in limit_strategies:
                    cfg = ExitConfig(
                        strategy=strat,
                        tp_rr=HEADLINE_TP_RR,
                        min_stop_atr_k=k_value,
                        slippage_ticks=1,
                        commission_per_share=0.005,
                        report_realism=True,
                        confidence_threshold=HEADLINE_THRESHOLD,
                        fill_mode=fm,
                    )
                    trades = compute_trades_for_model(
                        results.windows_raw, results.future_ohlcv,
                        results.future_timestamps, results.preds[model_name],
                        probas=results.probas.get(model_name),
                        tp_rr=cfg.tp_rr, exit_config=cfg,
                    )
                    s = summarise_trades(trades, strategy_name=strat, realism_config=cfg)
                    fr = s["fill_rate"]
                    fr_str = "N/A" if (isinstance(fr, float) and fr != fr) else f"{fr:.3f}"
                    sens_lines.append(
                        f"| {fm} | {strat} | {s['n_signals']} | {s['n_trades']} "
                        f"| {fr_str} | {s['win_rate']:.3f} "
                        f"| {s['after_cost_total_r']:+.2f} |"
                    )
            sens_lines.append("")

        run_dir.mkdir(parents=True, exist_ok=True)
        sens_path = run_dir / "sensitivity_sweep.md"
        sens_path.write_text("\n".join(sens_lines))
        print(f"\nSensitivity sweep: {sens_path}")
        print("Done.")
        return 0

    # -----------------------------------------------------------------------
    # --all-seeds: load 5 multisym checkpoints per arch, aggregate mean±std
    # -----------------------------------------------------------------------
    all_seeds_data: dict | None = None
    if do_all_seeds:
        from src.inspect.adapters.lstm_adapter import LSTMAdapter
        from src.inspect.adapters.cnn_lstm_adapter import CNNLSTMAdapter
        from src.inspect.adapters.transformer_adapter import TransformerAdapter
        from src.inspect.adapters.xgboost_adapter import XGBoostAdapter
        from src.inspect.adapters.xlstm_adapter import XLSTMAdapter
        from src.inspect.outcomes import compute_trades_for_model, summarise_trades
        from src.inspect.runner import run as run_inference

        _ADAPTER_CLASS_MAP = {
            "lstm": LSTMAdapter,
            "cnn_lstm": CNNLSTMAdapter,
            "transformer": TransformerAdapter,
            "xgboost": XGBoostAdapter,
            "xlstm": XLSTMAdapter,
        }

        # Use models requested (or all known archs if --models not specified)
        requested_archs = [
            n.split(":", 1)[0] if ":" in n else n
            for n in (args.models or list(_MULTISYM_DIRS.keys()))
        ]
        # Filter to only archs that have multisym checkpoint templates
        requested_archs = [a for a in requested_archs if a in _MULTISYM_DIRS]

        checkpoint_map = _resolve_multisym_checkpoints(checkpoint_dir)

        all_seeds_data = {}  # arch -> {strategy -> aggregated dict}

        for arch in requested_archs:
            if arch not in checkpoint_map:
                print(
                    f"WARNING: {arch} has no multisym checkpoints under "
                    f"checkpoints/{arch}_multisym/ — skipping from --all-seeds "
                    f"(custom --models name:path entries are not used in all-seeds mode).",
                    file=sys.stderr,
                )
                continue

            seed_entries = checkpoint_map[arch]
            print(f"\n[all-seeds] {arch}: found {len(seed_entries)} checkpoints")

            adapter_cls = _ADAPTER_CLASS_MAP.get(arch)
            if adapter_cls is None:
                print(f"WARNING: {arch} — no adapter class found, skipping.", file=sys.stderr)
                continue

            # Collect per-seed summaries for each strategy
            # strat -> list of (seed, summary_dict)
            strat_seed_summaries: dict[str, list[tuple[int, dict]]] = {
                s: [] for s in exit_strategies
            }

            for seed, ckpt_path in seed_entries:
                print(f"  seed={seed}  {ckpt_path.name}")
                try:
                    # Instantiate adapter directly (bypasses registry name collision)
                    adapter = adapter_cls(
                        checkpoint_dir,
                        checkpoint_path=str(ckpt_path),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  WARNING: could not load {arch} seed={seed}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                # Run inference for this seed
                seed_results = run_inference(
                    adapters=[adapter],
                    df_slice=df,
                    labeller=labeller,
                    lookahead_bars=args.lookahead_bars,
                )

                # seed_results is fresh per-seed; its only preds key is this adapter's name.
                adapter_name = adapter.name

                for cfg in exit_configs:
                    trades = compute_trades_for_model(
                        seed_results.windows_raw,
                        seed_results.future_ohlcv,
                        seed_results.future_timestamps,
                        seed_results.preds[adapter_name],
                        probas=seed_results.probas.get(adapter_name),
                        tp_rr=cfg.tp_rr,
                        exit_config=cfg,
                    )
                    summary = summarise_trades(
                        trades,
                        strategy_name=cfg.strategy,
                        realism_config=cfg,
                    )
                    strat_seed_summaries[cfg.strategy].append((seed, summary))

            # Aggregate across seeds per strategy
            arch_agg: dict[str, dict] = {}
            for strat_name, seed_summaries in strat_seed_summaries.items():
                if not seed_summaries:
                    continue
                arch_agg[strat_name] = _aggregate_seed_summaries(seed_summaries)

            if arch_agg:
                all_seeds_data[arch] = arch_agg

        if not all_seeds_data:
            print("WARNING: --all-seeds produced no data (all archs skipped).", file=sys.stderr)
            all_seeds_data = None

    print(f"\nWriting report to: {run_dir}")
    summary_path = write_report(
        results=results,
        stats=stats,
        output_dir=run_dir,
        top_k=args.top_k,
        df_full=df,
        tp_rr=args.tp_rr,
        exit_configs=exit_configs,
        all_seeds_data=all_seeds_data,
    )
    print(f"Summary: {summary_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
