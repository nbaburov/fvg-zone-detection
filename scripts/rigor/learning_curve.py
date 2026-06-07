"""learning_curve.py — Model-agnostic learning-curve harness (plan step A2/A2b).

Shells out to scripts/training/train.py for each (fraction, seed) combination,
parses best_smoothed_val_macro_f1 from the meta sidecar written by that script,
and emits:
  reports/rigor/<DD-Mon-YY>/learning_curve_<arch>.csv
  reports/rigor/<DD-Mon-YY>/learning_curve_<arch>.png

Guard (A2b): for every run, the meta's n_train_windows is asserted to be within
±5 % of fraction × N(1.0).  Fails loudly if a dispatch branch ignored
--train-fraction.

Usage:
  python scripts/rigor/learning_curve.py --model lstm --config experiments/lstm_g1.yaml
  python scripts/rigor/learning_curve.py --model cnn_lstm --config experiments/cnn_lstm_g1.yaml
      [--fractions 0.2 0.4 0.6 0.8 1.0]
      [--seeds 42 17 0]
      [--dry-run]           # print commands only, do not execute
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FRACTIONS: list[float] = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_SEEDS: list[int] = [42, 17, 0]
_GUARD_TOL: float = 0.05  # ±5 % relative tolerance for n_train_windows guard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_dir() -> Path:
    """Return today's rigor report directory, creating it if necessary."""
    tag = date.today().strftime("%d-%b-%y")  # e.g. 06-Jun-26
    d = ROOT / "reports" / "rigor" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_path_for_run(cfg_path: Path, arch: str, seed: int) -> Path | None:
    """Infer the .meta.json path written by train.py for a given seed.

    train.py resolves checkpoint_dir from cfg.runtime.checkpoint_dir
    (default: 'checkpoints').  The filename pattern is:
        <arch>_seed<N><label_tag>.meta.json
    where label_tag is '' for fvg_valid (canonical) or '_<labeller>' otherwise.

    We try fvg_valid first (no tag), then fall back to glob to be robust.
    """
    ckpt_root = ROOT / "checkpoints" / arch
    # Canonical: no label tag
    candidate = ckpt_root / f"{arch}_seed{seed}.meta.json"
    if candidate.exists():
        return candidate
    # Fallback: any meta for this seed
    matches = sorted(ckpt_root.glob(f"{arch}_seed{seed}*.meta.json"))
    if matches:
        return matches[-1]  # newest by name sort
    return None


def _parse_meta(meta_path: Path) -> dict[str, Any]:
    with open(meta_path) as f:
        return json.load(f)


def _run_train(
    arch: str,
    cfg_path: Path,
    seed: int,
    fraction: float,
    dry_run: bool,
) -> Path | None:
    """Shell out to train.py for one (seed, fraction) and return the meta path.

    Returns None on dry_run (command is printed but not executed).
    Raises RuntimeError if the subprocess exits non-zero.
    """
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "training" / "train.py"),
        "--config", str(cfg_path),
        "--seed", str(seed),
        "--train-fraction", str(fraction),
    ]
    print(f"  + {' '.join(cmd)}")
    if dry_run:
        return None

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"train.py exited {result.returncode} for arch={arch} seed={seed} "
            f"fraction={fraction}"
        )

    meta = _meta_path_for_run(cfg_path, arch, seed)
    if meta is None:
        raise FileNotFoundError(
            f"Could not find meta sidecar for arch={arch} seed={seed} after "
            f"successful train run (fraction={fraction}). "
            f"Expected in checkpoints/{arch}/{arch}_seed{seed}[*].meta.json"
        )
    return meta


# ---------------------------------------------------------------------------
# Guard (A2b)
# ---------------------------------------------------------------------------

def _guard_n_train_windows(
    meta: dict[str, Any],
    fraction: float,
    n_full: int,
    arch: str,
    seed: int,
) -> None:
    """Assert n_train_windows ≈ fraction × N(1.0) within ±5 %.

    Fails loudly if the dispatch branch ignored --train-fraction.
    n_full is the n_train_windows for the fraction=1.0 baseline seed.
    """
    actual = meta.get("n_train_windows")
    if actual is None:
        raise KeyError(
            f"[A2b GUARD] meta for arch={arch} seed={seed} fraction={fraction} "
            f"is missing 'n_train_windows'.  Cannot verify --train-fraction was honoured."
        )
    expected = fraction * n_full
    if expected == 0:
        raise ValueError(f"[A2b GUARD] n_full=0; cannot guard fraction={fraction}")
    rel_err = abs(actual - expected) / expected
    if rel_err > _GUARD_TOL:
        raise AssertionError(
            f"[A2b GUARD] FAIL: arch={arch} seed={seed} fraction={fraction}: "
            f"n_train_windows={actual}, expected≈{expected:.0f} "
            f"(fraction×N_full={fraction}×{n_full}), "
            f"relative_error={rel_err:.1%} > tol={_GUARD_TOL:.0%}.  "
            f"The dispatch branch likely ignored --train-fraction."
        )
    print(
        f"  [A2b GUARD] OK  fraction={fraction:.1f}  "
        f"n_train_windows={actual} ≈ {expected:.0f}  (err={rel_err:.1%})"
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(
    rows: list[dict[str, Any]],
    arch: str,
    out_path: Path,
) -> None:
    """Mean ± std-dev band of val_macro_f1 vs train fraction."""
    import collections
    import statistics

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_fraction: dict[float, list[float]] = collections.defaultdict(list)
    for r in rows:
        by_fraction[r["fraction"]].append(r["val_macro_f1"])

    fractions = sorted(by_fraction)
    means = [statistics.mean(by_fraction[f]) for f in fractions]
    stds = [statistics.stdev(by_fraction[f]) if len(by_fraction[f]) > 1 else 0.0
            for f in fractions]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fractions, means, marker="o", linewidth=2, label="mean val macro-F1")
    ax.fill_between(
        fractions,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25,
        label="±1 std",
    )
    # Individual seed points
    for r in rows:
        ax.scatter(r["fraction"], r["val_macro_f1"], color="steelblue", alpha=0.5, s=20, zorder=3)

    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Val macro-F1 (best smoothed)")
    ax.set_title(f"Learning curve — {arch}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Model-agnostic learning-curve harness.  "
            "Trains arch across fractions × seeds, guards --train-fraction "
            "compliance (A2b), writes CSV + PNG."
        )
    )
    p.add_argument(
        "--model",
        required=True,
        help="Architecture name — must match cfg.model.arch in the YAML "
             "(e.g. lstm, cnn_lstm, transformer, xlstm, xgb).  "
             "NOT hard-coded; the YAML's arch field is the source of truth.",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML passed verbatim to train.py.",
    )
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        metavar="F",
        help=f"Train fractions to sweep (default: {DEFAULT_FRACTIONS}).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="S",
        help=f"Seeds to run per fraction (default: {DEFAULT_SEEDS}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print all train.py commands but do not execute them.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    arch: str = args.model
    cfg_path = Path(args.config).resolve()
    fractions: list[float] = sorted(set(args.fractions))
    seeds: list[int] = args.seeds

    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure 1.0 is always in the sweep — required for the A2b guard denominator.
    if 1.0 not in fractions:
        print("[WARNING] 1.0 not in --fractions; appending it for A2b guard denominator.")
        fractions = sorted(fractions + [1.0])

    report_dir = _report_dir()
    csv_path = report_dir / f"learning_curve_{arch}.csv"
    png_path = report_dir / f"learning_curve_{arch}.png"

    print(f"Learning curve sweep: arch={arch}  fractions={fractions}  seeds={seeds}")
    print(f"Config: {cfg_path}")
    print(f"Output: {csv_path}")
    if args.dry_run:
        print("[DRY RUN — no training will execute]\n")

    rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1: run fraction=1.0 for all seeds first to get N_full baseline.
    # ------------------------------------------------------------------
    n_full_by_seed: dict[int, int] = {}

    print("\n--- Baseline pass (fraction=1.0) ---")
    for seed in seeds:
        print(f"\n[fraction=1.0  seed={seed}]")
        meta_path = _run_train(arch, cfg_path, seed, 1.0, args.dry_run)

        if args.dry_run:
            continue

        meta = _parse_meta(meta_path)
        n_full = meta.get("n_train_windows")
        if n_full is None:
            raise KeyError(
                f"meta for arch={arch} seed={seed} fraction=1.0 is missing "
                f"'n_train_windows'.  Cannot establish guard denominator."
            )
        n_full_by_seed[seed] = int(n_full)

        f1 = meta.get("best_smoothed_val_macro_f1")
        if f1 is None:
            raise KeyError(
                f"meta for arch={arch} seed={seed} fraction=1.0 is missing "
                f"'best_smoothed_val_macro_f1'."
            )
        rows.append({"arch": arch, "fraction": 1.0, "seed": seed, "val_macro_f1": float(f1)})
        print(f"  best_smoothed_val_macro_f1={f1:.4f}  n_train_windows={n_full}")

    # ------------------------------------------------------------------
    # Step 2: remaining fractions (skip 1.0 — already done above).
    # ------------------------------------------------------------------
    sub_fractions = [f for f in fractions if f != 1.0]
    if sub_fractions:
        print(f"\n--- Sub-fraction passes: {sub_fractions} ---")

    for fraction in sub_fractions:
        for seed in seeds:
            print(f"\n[fraction={fraction}  seed={seed}]")
            meta_path = _run_train(arch, cfg_path, seed, fraction, args.dry_run)

            if args.dry_run:
                continue

            meta = _parse_meta(meta_path)

            # A2b guard
            n_full = n_full_by_seed.get(seed)
            if n_full is None:
                raise RuntimeError(
                    f"[A2b GUARD] No N_full baseline for seed={seed}; "
                    f"fraction=1.0 pass must have succeeded first."
                )
            _guard_n_train_windows(meta, fraction, n_full, arch, seed)

            f1 = meta.get("best_smoothed_val_macro_f1")
            if f1 is None:
                raise KeyError(
                    f"meta for arch={arch} seed={seed} fraction={fraction} is missing "
                    f"'best_smoothed_val_macro_f1'."
                )
            rows.append({
                "arch": arch,
                "fraction": fraction,
                "seed": seed,
                "val_macro_f1": float(f1),
            })
            print(f"  best_smoothed_val_macro_f1={f1:.4f}")

    if args.dry_run:
        print("\n[DRY RUN complete — no outputs written]")
        return

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["arch", "fraction", "seed", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # ------------------------------------------------------------------
    # Write PNG
    # ------------------------------------------------------------------
    if rows:
        _plot(rows, arch, png_path)
    else:
        print("[WARNING] No rows to plot.")


if __name__ == "__main__":
    main()
