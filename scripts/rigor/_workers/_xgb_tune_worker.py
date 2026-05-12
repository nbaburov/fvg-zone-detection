"""_xgb_tune_worker.py — XGB Optuna worker running in subprocess."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-npz', required=True)
    parser.add_argument('--storage-url', required=True)
    parser.add_argument('--study-name', required=True)
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--output-config', required=True)
    args = parser.parse_args()

    data = np.load(args.data_npz)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    sample_weight = data['sample_weight']

    from src.rigor.optuna_xgb import XGBObjective, run_study_xgb
    objective = XGBObjective(X_train, y_train, X_val, y_val, sample_weight)
    study = run_study_xgb(
        objective=objective,
        n_trials=args.n_trials,
        storage_url=args.storage_url,
        study_name=args.study_name,
    )

    best_config = {
        **study.best_trial.params,
        'val_macro_f1': study.best_trial.value,
        'trial_number': study.best_trial.number,
    }
    out = Path(args.output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as fh:
        json.dump(best_config, fh, indent=2)
    print('Best XGB config:', json.dumps(best_config, indent=2))

if __name__ == '__main__':
    main()
