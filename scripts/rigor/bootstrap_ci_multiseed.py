"""bootstrap_ci_multiseed.py — Block bootstrap CI of MEAN across multiple seeds (Gap 10).

Usage:
  python scripts/rigor/bootstrap_ci_multiseed.py \
      --pred-dir reports/rigor/2026-05-13/G2 \
      --model lstm \
      --block-size 60 --n-iter 1000 \
      --output-dir reports/rigor/2026-05-13/G10

For each bootstrap resample, computes macro F1 per seed, then averages across seeds.
CI is of the mean — wider than single-seed CI, more honest about model variance.

Effective n = n_bars // window_size (non-overlapping, overlap-aware).
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
    parser = argparse.ArgumentParser(description="Multi-seed block bootstrap CI")
    parser.add_argument("--pred-dir", required=True, type=Path,
                        help="Directory containing <model>_seed*_preds.npz files")
    parser.add_argument("--model", required=True, choices=["lstm", "xgb", "xgboost", "cnn_lstm"],
                        help="Model type prefix to glob for")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to experiments/foo.yaml (optional — sets block_size, n_iter)")
    parser.add_argument("--set", dest="set_overrides", nargs="+", default=[],
                        metavar="key=value",
                        help='Override config fields: --set "eval.bootstrap_n_iter=500"')
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=42,
                        help="RNG seed for bootstrap resampling")
    parser.add_argument("--output-dir", required=True, type=Path)
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

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve bootstrap params: CLI > config > default
    block_size = args.block_size or (cfg.eval.bootstrap_block_size if cfg else 60)
    n_iter = args.n_iter or (cfg.eval.bootstrap_n_iter if cfg else 1000)

    model_prefix = "xgb" if args.model in ("xgb", "xgboost") else args.model
    pred_files = sorted(pred_dir.glob(f"{model_prefix}_seed*_preds.npz"))
    if not pred_files:
        print(f"ERROR: No {model_prefix}_seed*_preds.npz files in {pred_dir}")
        sys.exit(1)

    print(f"Found {len(pred_files)} seed prediction files for model={model_prefix}")

    # Load all seeds
    seeds_data: list[tuple[np.ndarray, np.ndarray]] = []
    for pf in pred_files:
        d = np.load(pf)
        seeds_data.append((d["y_true"], d["y_pred"]))
        print(f"  {pf.name}: n={len(d['y_true'])}")

    # Validate all seeds have same n
    ns = [len(yt) for yt, _ in seeds_data]
    if len(set(ns)) != 1:
        print(f"WARNING: seeds have different test set sizes: {ns}")

    n = ns[0]
    eff_n = n // block_size
    print(f"\nTest set n={n}, block_size={block_size}, effective_n={eff_n}")

    rng = np.random.default_rng(args.rng_seed)
    block_starts = np.arange(0, n - block_size + 1)
    n_blocks = max(1, n // block_size)

    # Point estimates (mean across seeds)
    from sklearn.metrics import f1_score

    def macro_f1(yt, yp):
        return float(f1_score(yt, yp, average="macro", zero_division=0.0))

    def per_class_f1(yt, yp):
        scores = f1_score(yt, yp, average=None, zero_division=0.0, labels=[0, 1, 2])
        result = [0.0, 0.0, 0.0]
        for i, v in enumerate(scores[:3]):
            result[i] = float(v)
        return result

    point_macros = [macro_f1(yt, yp) for yt, yp in seeds_data]
    point_bulls = [per_class_f1(yt, yp)[1] for yt, yp in seeds_data]
    point_bears = [per_class_f1(yt, yp)[2] for yt, yp in seeds_data]

    point_mean_macro = float(np.mean(point_macros))
    point_mean_bull = float(np.mean(point_bulls))
    point_mean_bear = float(np.mean(point_bears))

    print(f"\nPoint estimates (mean across {len(seeds_data)} seeds):")
    print(f"  macro_f1: {point_mean_macro:.4f}")
    print(f"  bull_f1:  {point_mean_bull:.4f}")
    print(f"  bear_f1:  {point_mean_bear:.4f}")

    # Bootstrap: per resample, compute mean across seeds
    boot_macros: list[float] = []
    boot_bulls: list[float] = []
    boot_bears: list[float] = []

    print(f"\nRunning {n_iter} bootstrap iterations...")
    for i in range(n_iter):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        indices = np.concatenate([
            np.arange(start, min(start + block_size, n))
            for start in chosen
        ])

        seed_macros = []
        seed_bulls = []
        seed_bears = []
        for yt, yp in seeds_data:
            bt_true = yt[indices]
            bt_pred = yp[indices]
            seed_macros.append(macro_f1(bt_true, bt_pred))
            pcf = per_class_f1(bt_true, bt_pred)
            seed_bulls.append(pcf[1])
            seed_bears.append(pcf[2])

        boot_macros.append(float(np.mean(seed_macros)))
        boot_bulls.append(float(np.mean(seed_bulls)))
        boot_bears.append(float(np.mean(seed_bears)))

    macro_arr = np.array(boot_macros)
    bull_arr = np.array(boot_bulls)
    bear_arr = np.array(boot_bears)

    result = {
        "model": model_prefix,
        "n_seeds": len(seeds_data),
        "macro_f1_mean": point_mean_macro,
        "ci_95_lower": float(np.percentile(macro_arr, 2.5)),
        "ci_95_upper": float(np.percentile(macro_arr, 97.5)),
        "bull_f1_mean": point_mean_bull,
        "bull_ci_95_lower": float(np.percentile(bull_arr, 2.5)),
        "bull_ci_95_upper": float(np.percentile(bull_arr, 97.5)),
        "bear_f1_mean": point_mean_bear,
        "bear_ci_95_lower": float(np.percentile(bear_arr, 2.5)),
        "bear_ci_95_upper": float(np.percentile(bear_arr, 97.5)),
        "n_bootstrap": n_iter,
        "block_size": block_size,
        "effective_n": eff_n,
        "per_seed_point_macros": point_macros,
        "std_macro_f1": float(np.std(point_macros)),
    }

    print(f"\n95% CI of mean macro F1 across seeds:")
    print(f"  [{result['ci_95_lower']:.4f}, {result['ci_95_upper']:.4f}]")
    print(f"  (point mean: {result['macro_f1_mean']:.4f})")
    print(f"  effective_n: {eff_n} (= {n} bars // {block_size})")

    out_path = out_dir / f"bootstrap_ci_{model_prefix}.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
