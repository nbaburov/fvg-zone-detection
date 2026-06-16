"""tune_cnn_lstm.py — Optuna hyperparameter search for CNN-LSTM FVG classifier.

Usage:
  python scripts/rigor/tune/tune_cnn_lstm.py [--n-trials 50] [--device cpu] [--max-epochs 50]

Outputs:
  checkpoints/cnn_lstm_h1_spy/optuna_<date>.db
  reports/rigor/2026-05-13/cnn_lstm_G1/best_hp_cnn_lstm.json

Test set is NEVER loaded in this script. Train + val only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HP search for CNN-LSTM FVG classifier")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to experiments/foo.yaml (optional — sets data_dir, window_size etc.)")
    parser.add_argument("--set", dest="set_overrides", nargs="+", default=[],
                        metavar="key=value",
                        help='Override config fields: --set "data.window_size=60"')
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default=None,
                        help="Optuna study name override. Default: auto-namespaced by arch+timeframe.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeframe", default="h1", choices=["h1", "5m", "15m"],
                        help="Data timeframe token (default: h1 = legacy unsuffixed paths)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | cuda  (MPS excluded — CNN-LSTM has same MPS bug as LSTM)")
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

    # Device — CPU forced; MPS excluded for CNN-LSTM
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data — train + val ONLY
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset

    data_dir = ROOT / (cfg.data.data_dir if cfg else Path("data/processed"))
    window_size = cfg.data.window_size if cfg else 60
    from src.rigor.seed_sweep import _load_splits, _load_class_weights
    _timeframe = args.timeframe
    _splits = _load_splits(data_dir, _timeframe)
    train_df, val_df = _splits[0], _splits[1]  # test deliberately NOT loaded
    # test parquet deliberately NOT loaded here

    labeller_key = cfg.data.labeller if cfg else "fvg_valid"
    labeller = LABELLERS[labeller_key]()
    train_ds = SMCWindowDataset(train_df, labeller, stride=1, window_size=window_size, drop_cross_session_windows=False)
    val_ds = SMCWindowDataset(val_df, labeller, stride=1, window_size=window_size, drop_cross_session_windows=False)

    # Placeholder loaders — batch size overridden per trial in CNNLSTMObjective
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # Class weights
    _cw_raw = _load_class_weights(data_dir, _timeframe)
    weights = torch.tensor(_cw_raw, dtype=torch.float32)

    # Output dirs
    from src.rigor.report_utils import timestamped_dir
    output_dir = args.output_dir or (cfg.runtime.output_dir if cfg else Path("reports/rigor"))
    ts_dir = timestamped_dir(ROOT / output_dir)
    from src.rigor.optuna_ns import _optuna_storage_and_study
    storage_path, storage_url, _study_name = _optuna_storage_and_study(
        "cnn_lstm", _timeframe, ROOT
    )
    if args.study_name is not None:   # explicit CLI override
        _study_name = args.study_name
    ckpt_dir = storage_path.parent

    from src.rigor.optuna_utils import CNNLSTMObjective, run_study

    objective = CNNLSTMObjective(
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
        study_name=_study_name,
    )

    # Save best HP
    best_hp = {
        **study.best_trial.params,
        "val_macro_f1": study.best_trial.value,
        "trial_number": study.best_trial.number,
        "n_trials_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "study_name": _study_name,
        "storage": str(storage_path),
    }
    _tf_suffix = f"_{_timeframe}" if _timeframe != "h1" else ""
    hp_path = ts_dir / f"best_hp_cnn_lstm{_tf_suffix}.json"
    with hp_path.open("w") as fh:
        json.dump(best_hp, fh, indent=2)
    print(f"\nBest HP saved: {hp_path}")
    print(json.dumps(best_hp, indent=2))

    # Optuna HTML visualisations if plotly available
    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html(str(ts_dir / "cnn_lstm_optuna_history.html"), include_plotlyjs="cdn")
        vis.plot_param_importances(study).write_html(str(ts_dir / "cnn_lstm_optuna_importances.html"), include_plotlyjs="cdn")
        print(f"Optuna HTML saved to {ts_dir}")
    except Exception as e:
        print(f"Optuna visualisation skipped: {e}")


if __name__ == "__main__":
    main()
