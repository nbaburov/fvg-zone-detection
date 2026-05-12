"""tune_lstm.py — Optuna hyperparameter search for LSTM FVG classifier.

Usage:
  python scripts/rigor/tune_lstm.py [--n-trials 50] [--study-name lstm_fvg] [--device auto]

Outputs:
  checkpoints/lstm/optuna_<date>.db
  reports/rigor/<ts>/best_lstm_config.json
  reports/rigor/<ts>/lstm_optuna_summary.html  (if optuna-dashboard available)

Test set is NEVER loaded in this script. Train + val only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HP search for LSTM FVG classifier")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to experiments/foo.yaml (optional — sets data_dir, window_size etc.)")
    parser.add_argument("--set", dest="set_overrides", nargs="+", default=[],
                        metavar="key=value",
                        help='Override config fields: --set "data.window_size=60"')
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default="lstm_fvg")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu | mps | cuda | auto")
    parser.add_argument("--max-epochs", type=int, default=50,
                        help="Max epochs per trial (default 50 to cap compute)")
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

    # Resolve device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load data — train + val ONLY
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset

    data_dir = ROOT / (cfg.data.data_dir if cfg else Path("data/processed"))
    window_size = cfg.data.window_size if cfg else 60
    train_df = pd.read_parquet(data_dir / "spy_h1_train.parquet")
    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    # test parquet deliberately NOT loaded here

    labeller_key = cfg.data.labeller if cfg else "fvg_valid"
    labeller = LABELLERS[labeller_key]()
    train_ds = SMCWindowDataset(train_df, labeller, stride=1, window_size=window_size, drop_cross_session_windows=False)
    val_ds = SMCWindowDataset(val_df, labeller, stride=1, window_size=window_size, drop_cross_session_windows=False)

    # Placeholder loaders — batch size overridden per trial in LSTMObjective
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # Class weights
    with (data_dir / "class_weights.json").open() as fh:
        cw = json.load(fh)
    weights = torch.tensor([float(cw[str(i)]) for i in range(3)], dtype=torch.float32)

    # Output dirs
    from src.rigor.report_utils import timestamped_dir
    output_dir = args.output_dir or (cfg.runtime.output_dir if cfg else Path("reports/rigor"))
    ts_dir = timestamped_dir(ROOT / output_dir)
    ckpt_dir = ROOT / "checkpoints" / "lstm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    storage_path = ckpt_dir / f"optuna_{date.today()}.db"
    storage_url = f"sqlite:///{storage_path}"

    from src.rigor.optuna_utils import LSTMObjective, run_study

    objective = LSTMObjective(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=weights,
        device=device,
        max_epochs=args.max_epochs,
        patience=10,
    )

    print(f"\nRunning {args.n_trials} trials. Storage: {storage_path}")
    print(f"Output dir: {ts_dir}\n")

    study = run_study(
        objective,
        n_trials=args.n_trials,
        storage_url=storage_url,
        study_name=args.study_name,
    )

    # Save best config
    best_config = {
        **study.best_trial.params,
        "val_macro_f1": study.best_trial.value,
        "trial_number": study.best_trial.number,
        "study_name": args.study_name,
        "storage": str(storage_path),
    }
    config_path = ts_dir / "best_lstm_config.json"
    with config_path.open("w") as fh:
        json.dump(best_config, fh, indent=2)
    print(f"\nBest config saved: {config_path}")
    print(json.dumps(best_config, indent=2))

    # Save Optuna HTML visualisations if plotly available
    try:
        import optuna.visualization as vis
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(str(ts_dir / "lstm_optuna_history.html"))

        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html(str(ts_dir / "lstm_optuna_importances.html"))

        print(f"Optuna HTML saved to {ts_dir}")
    except Exception as e:
        print(f"Optuna visualisation skipped: {e}")


if __name__ == "__main__":
    main()
