"""multiseed_run.py — Multi-seed training sweep (Gap 2: variance, Gap 3: focal ablation).

Usage:
  python scripts/rigor/multiseed_run.py --model lstm --config reports/rigor/<ts>/best_lstm_config.json
      --seeds 42 17 0 123 2024 --loss weighted_ce

  # Focal ablation:
  python scripts/rigor/multiseed_run.py --model lstm --config best_lstm_config.json
      --seeds 42 17 0 123 2024 --loss focal --gamma 2

  # Regularisation ablation (Gap 9) via --no-dropout and/or --no-l2:
  python scripts/rigor/multiseed_run.py --model lstm --config best_lstm_config.json
      --seeds 42 --loss weighted_ce --no-dropout --no-l2

Outputs:
  checkpoints/<model>/<name>_seed<N>.{pt,ubj} + .meta.json
  reports/rigor/<ts>/multiseed_summary_<model>_<loss>.md
  reports/rigor/<ts>/<model>_seed<N>_preds.npz  (for bootstrap CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training sweep")
    parser.add_argument("--model", required=True, choices=["lstm", "xgboost"])
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to best_<model>_config.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 0, 123, 2024])
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal loss gamma (only used when --loss focal)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rigor"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    # Regularisation ablation flags (Gap 9)
    parser.add_argument("--no-dropout", action="store_true",
                        help="Ablation: set dropout=0, head_dropout=0")
    parser.add_argument("--no-l2", action="store_true",
                        help="Ablation: set weight_decay=0")
    args = parser.parse_args()

    # Load hyperparams from config JSON
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    with config_path.open() as fh:
        hyperparams = json.load(fh)

    # Strip non-HP keys that may be in config JSON
    hp_only = {k: v for k, v in hyperparams.items()
               if k not in ("val_macro_f1", "trial_number", "study_name", "storage")}

    # Apply regularisation ablation overrides
    if args.no_dropout:
        hp_only["dropout"] = 0.0
        hp_only["head_dropout"] = 0.0
    if args.no_l2:
        hp_only["weight_decay"] = 0.0

    from src.rigor.report_utils import timestamped_dir, write_rigor_report
    from src.rigor.seed_sweep import SeedSweepConfig, run_seed_sweep

    ts_dir = timestamped_dir(ROOT / args.output_dir)

    cfg = SeedSweepConfig(
        model_type=args.model if args.model != "xgboost" else "xgb",
        hyperparams=hp_only,
        seeds=args.seeds,
        loss_type=args.loss,
        focal_gamma=args.gamma,
        output_dir=ts_dir,
        checkpoint_dir=ROOT / args.checkpoint_dir,
        data_dir=ROOT / args.data_dir,
    )

    # For XGBoost, run in subprocess to avoid Python 3.14 segfault
    if args.model == "xgboost":
        _run_xgb_seed_sweep_subprocess(cfg, ts_dir)
        return

    print(f"\nStarting {len(args.seeds)}-seed sweep: {args.model} / {args.loss}"
          f"{f' gamma={args.gamma}' if args.loss == 'focal' else ''}")
    print(f"Checkpoints: {cfg.checkpoint_dir}")
    print(f"Output: {ts_dir}\n")

    df = run_seed_sweep(cfg)

    # Build summary filename
    suffix = f"{args.loss}"
    if args.loss == "focal":
        suffix += f"_g{args.gamma:.0f}"
    if args.no_dropout and args.no_l2:
        suffix += "_no_reg"
    elif args.no_dropout:
        suffix += "_no_dropout"
    elif args.no_l2:
        suffix += "_no_l2"

    title = f"multiseed_summary_{args.model}_{suffix}"
    report_path = write_rigor_report(df, ts_dir, title)
    print(f"\nSummary written: {report_path}")
    print(df.to_string(index=False))
    print(f"\nMean macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")


def _run_xgb_seed_sweep_subprocess(cfg, ts_dir: Path) -> None:
    """Run XGBoost seed sweep in subprocess due to Python 3.14 segfault."""
    import subprocess
    import tempfile
    import numpy as np

    from src.features.window_features import extract_window_features
    import pandas as pd

    data_dir = cfg.data_dir
    train_df = pd.read_parquet(data_dir / "spy_h1_train.parquet")
    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")

    X_train, y_train = extract_window_features(train_df)
    X_val, y_val = extract_window_features(val_df)
    X_test, y_test = extract_window_features(test_df)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    np.savez(tmp_path, X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    worker_script = ROOT / "scripts" / "rigor" / "_workers" / "_xgb_sweep_worker.py"
    # Note: _xgb_sweep_worker.py must be torch-free. Do NOT overwrite it here.
    # The worker already exists on disk with the correct torch-free implementation.

    config_json = json.dumps(cfg.hyperparams)

    cmd = [
        sys.executable, str(worker_script),
        "--data-npz", str(tmp_path),
        "--config", config_json,
        "--seeds", *[str(s) for s in cfg.seeds],
        "--output-dir", str(ts_dir),
        "--checkpoint-dir", str(cfg.checkpoint_dir),
    ]

    subprocess.run(cmd, check=True, cwd=str(ROOT))
    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
