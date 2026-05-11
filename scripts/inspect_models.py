"""inspect_models.py — CLI for model inspection and live comparison.

Usage
-----
    python scripts/inspect_models.py --help
    python scripts/inspect_models.py --list
    python scripts/inspect_models.py --models lstm xgboost --dataset test --top-k 30
    python scripts/inspect_models.py --models lstm --dataset data/processed/spy_h1_val.parquet \\
        --start 2023-06 --end 2023-09

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
        metavar="NAME",
        default=None,
        help="Adapter names to load (e.g. lstm xgboost). Default: all discovered.",
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
    else:
        model_names = args.models

    print(f"Loading adapters: {model_names}")
    try:
        adapters = load_adapters(model_names, checkpoint_dir)
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

    print(f"\nWriting report to: {run_dir}")
    summary_path = write_report(
        results=results,
        stats=stats,
        output_dir=run_dir,
        top_k=args.top_k,
        df_full=df,
        tp_rr=args.tp_rr,
    )
    print(f"Summary: {summary_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
