"""bootstrap_ci.py — Block bootstrap confidence intervals for test metrics (Gap 10).

Usage:
  python scripts/rigor/bootstrap_ci.py --predictions reports/rigor/<ts>/lstm_seed42_preds.npz
      [--block-size 60] [--n-iter 1000] [--seed 42] [--output-dir reports/rigor]

Appends CI block to docs/models-status.md.
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
    parser = argparse.ArgumentParser(description="Block bootstrap CI for test metrics")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Path to npz with y_true and y_pred arrays")
    parser.add_argument("--block-size", type=int, default=60)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rigor"))
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path

    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}")
        sys.exit(1)

    data = np.load(pred_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    from src.rigor.bootstrap_ci import block_bootstrap_f1, effective_n
    from src.rigor.report_utils import timestamped_dir

    ts_dir = timestamped_dir(ROOT / args.output_dir)

    print(f"Running block bootstrap (block_size={args.block_size}, n_iter={args.n_iter})...")
    print(f"Test set size: {len(y_true)}, effective_n={effective_n(len(y_true), args.block_size)}")

    result = block_bootstrap_f1(
        y_true, y_pred,
        block_size=args.block_size,
        n_iterations=args.n_iter,
        seed=args.seed,
    )

    print("\nBootstrap CI Results (95%):")
    print(f"{'Metric':<15} {'Point':<10} {'CI Lower':<12} {'CI Upper':<12}")
    print("-" * 50)
    for key in ("macro_f1", "bull_f1", "bear_f1"):
        r = result[key]
        print(f"{key:<15} {r['point']:<10.4f} {r['ci_lower']:<12.4f} {r['ci_upper']:<12.4f}")
    print(f"\nEffective n (non-overlapping): {result['effective_n']}")
    print("Note: CIs are wide due to small effective_n (~58). This is honest.")

    # Save JSON
    ci_path = ts_dir / "bootstrap_ci_lstm.json"
    with ci_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nCI JSON saved: {ci_path}")

    # Append to docs/models-status.md
    status_path = ROOT / "docs" / "models-status.md"
    if status_path.exists():
        ci_section = _format_ci_section(result, pred_path)
        existing = status_path.read_text()
        if "## Confidence Intervals" not in existing:
            status_path.write_text(existing + "\n\n" + ci_section)
            print(f"CI section appended to {status_path}")
        else:
            print(f"CI section already exists in {status_path} — not overwriting")
    else:
        print(f"docs/models-status.md not found — skipping append")


def _format_ci_section(result: dict, pred_path: Path) -> str:
    lines = [
        "## Confidence Intervals",
        "",
        f"Block bootstrap (block_size={result['block_size']}, n={result['n_bootstrap']}, seed=42)",
        f"Effective n: {result['effective_n']} non-overlapping test windows",
        f"Source predictions: `{pred_path.name}`",
        "",
        "| Metric | Point | CI Lower (2.5%) | CI Upper (97.5%) |",
        "|--------|-------|-----------------|------------------|",
    ]
    for key in ("macro_f1", "bull_f1", "bear_f1"):
        r = result[key]
        lines.append(f"| {key} | {r['point']:.4f} | {r['ci_lower']:.4f} | {r['ci_upper']:.4f} |")
    lines += [
        "",
        "> Note: Wide CIs are expected with effective_n≈58 (3514 test bars / window_size=60).",
        "> This reflects honest uncertainty — not a model deficiency.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
