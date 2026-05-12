"""tune_xgboost.py — Optuna hyperparameter search for XGBoost FVG classifier.

Usage:
  python scripts/rigor/tune_xgboost.py [--n-trials 50] [--study-name xgb_fvg]

Outputs:
  checkpoints/xgboost/optuna_<date>.db
  reports/rigor/<ts>/best_xgb_config.json

Note: XGBoost runs in a subprocess to avoid the Python 3.14 segfault.
Each trial spawns a child process via the _xgb_worker pattern.
Test set is NEVER loaded in this script. Train + val only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HP search for XGBoost FVG classifier")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default="xgb_fvg")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rigor"))
    args = parser.parse_args()

    from src.features.window_features import extract_window_features
    from src.rigor.report_utils import timestamped_dir

    data_dir = ROOT / "data" / "processed"

    # Load train + val feature matrices (no test)
    import pandas as pd
    train_df = pd.read_parquet(data_dir / "spy_h1_train.parquet")
    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")

    X_train, y_train = extract_window_features(train_df)
    X_val, y_val = extract_window_features(val_df)

    # Class weights as sample_weight
    with (data_dir / "class_weights.json").open() as fh:
        cw = json.load(fh)
    weights_arr = [float(cw[str(i)]) for i in range(3)]
    sample_weight = np.array([weights_arr[int(y)] for y in y_train], dtype=np.float32)

    ts_dir = timestamped_dir(ROOT / args.output_dir)
    ckpt_dir = ROOT / "checkpoints" / "xgboost"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    storage_path = ckpt_dir / f"optuna_{date.today()}.db"
    storage_url = f"sqlite:///{storage_path}"

    # Persist feature matrices to temp npz so subprocess can load them
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    np.savez(tmp_path, X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, sample_weight=sample_weight)

    print(f"Running {args.n_trials} XGB trials. Storage: {storage_path}")

    # Run tuning in subprocess to avoid Python 3.14 XGB segfault
    worker_script = ROOT / "scripts" / "rigor" / "_workers" / "_xgb_tune_worker.py"
    _write_xgb_tune_worker(worker_script)

    cmd = [
        sys.executable, str(worker_script),
        "--data-npz", str(tmp_path),
        "--storage-url", storage_url,
        "--study-name", args.study_name,
        "--n-trials", str(args.n_trials),
        "--output-config", str(ts_dir / "best_xgb_config.json"),
    ]

    result = subprocess.run(cmd, check=True, cwd=str(ROOT))

    tmp_path.unlink(missing_ok=True)

    # Load and display best config
    config_path = ts_dir / "best_xgb_config.json"
    if config_path.exists():
        with config_path.open() as fh:
            best_config = json.load(fh)
        print(f"\nBest XGB config saved: {config_path}")
        print(json.dumps(best_config, indent=2))

        # Save Optuna HTML if possible
        try:
            import optuna
            import optuna.visualization as vis
            study = optuna.load_study(study_name=args.study_name, storage=storage_url)
            vis.plot_optimization_history(study).write_html(
                str(ts_dir / "xgb_optuna_history.html")
            )
            vis.plot_param_importances(study).write_html(
                str(ts_dir / "xgb_optuna_importances.html")
            )
            print(f"Optuna HTML saved to {ts_dir}")
        except Exception as e:
            print(f"Optuna visualisation skipped: {e}")
    else:
        print("ERROR: best_xgb_config.json not created")
        sys.exit(1)


def _write_xgb_tune_worker(path: Path) -> None:
    """Write the XGB tuning subprocess worker script."""
    path.write_text(
        '"""_xgb_tune_worker.py — XGB Optuna worker running in subprocess."""\n'
        "from __future__ import annotations\n"
        "import argparse\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "\n"
        "ROOT = Path(__file__).resolve().parent.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "\n"
        "def main() -> None:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--data-npz', required=True)\n"
        "    parser.add_argument('--storage-url', required=True)\n"
        "    parser.add_argument('--study-name', required=True)\n"
        "    parser.add_argument('--n-trials', type=int, default=50)\n"
        "    parser.add_argument('--output-config', required=True)\n"
        "    args = parser.parse_args()\n"
        "\n"
        "    data = np.load(args.data_npz)\n"
        "    X_train = data['X_train']\n"
        "    y_train = data['y_train']\n"
        "    X_val = data['X_val']\n"
        "    y_val = data['y_val']\n"
        "    sample_weight = data['sample_weight']\n"
        "\n"
        "    from src.rigor.optuna_xgb import XGBObjective, run_study_xgb\n"
        "    objective = XGBObjective(X_train, y_train, X_val, y_val, sample_weight)\n"
        "    study = run_study_xgb(\n"
        "        objective=objective,\n"
        "        n_trials=args.n_trials,\n"
        "        storage_url=args.storage_url,\n"
        "        study_name=args.study_name,\n"
        "    )\n"
        "\n"
        "    best_config = {\n"
        "        **study.best_trial.params,\n"
        "        'val_macro_f1': study.best_trial.value,\n"
        "        'trial_number': study.best_trial.number,\n"
        "    }\n"
        "    out = Path(args.output_config)\n"
        "    out.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with out.open('w') as fh:\n"
        "        json.dump(best_config, fh, indent=2)\n"
        "    print('Best XGB config:', json.dumps(best_config, indent=2))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


if __name__ == "__main__":
    main()
