"""optuna_xgb.py — XGBoost Optuna objective (torch-free, subprocess-safe).

Kept separate from optuna_utils.py so it can be imported in XGB subprocess
workers without pulling in torch/LSTM code (which segfaults XGBoost on Python 3.14).
"""

from __future__ import annotations

import numpy as np
import optuna


class XGBObjective:
    """Optuna callable for XGBoost hyperparameter search.

    Uses train + val feature matrices only. No torch dependency.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weights = sample_weights

    def __call__(self, trial: optuna.Trial) -> float:
        import xgboost as xgb
        from sklearn.metrics import f1_score

        n_estimators = trial.suggest_int("n_estimators", 100, 800)
        max_depth = trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.4, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        xgb_metrics = ["mlogloss"]
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric=xgb_metrics,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=30,
            verbosity=0,
        )

        clf.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.sample_weights,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )

        y_pred = clf.predict(self.X_val)
        macro_f1 = float(f1_score(self.y_val, y_pred, average="macro", zero_division=0.0))
        return macro_f1


def run_study_xgb(
    objective: XGBObjective,
    n_trials: int,
    storage_url: str,
    study_name: str,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 10,
) -> optuna.Study:
    """Create or load XGB Optuna study and run trials. Torch-free."""
    import json

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials - n_done)

    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    else:
        print(f"Study '{study_name}' already has {n_done} completed trials — skipping.")

    best = study.best_trial
    print(f"\nBest trial #{best.number}: val Macro F1 = {best.value:.4f}")
    print("Best params:", json.dumps(best.params, indent=2))

    return study
