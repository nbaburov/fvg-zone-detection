"""optuna_utils.py — Optuna objectives + study runner for LSTM and XGBoost HP search.

Design constraints:
  - LSTMObjective: uses train + val only. Test parquet NEVER loaded.
  - XGBObjective: uses train + val only.
  - MedianPruner: fires after each epoch (LSTM) or boosting round (XGB).
  - Fixed seed=42 inside objectives so search variance is hyperparams, not seed.
  - run_study: idempotent — loads existing SQLite study and resumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.lstm import FVGLSTMClassifier
from src.training.early_stop import EarlyStop
from src.training.loss import WeightedCE
from src.training.train_utils import set_seed


# ---------------------------------------------------------------------------
# LSTM Objective
# ---------------------------------------------------------------------------

class LSTMObjective:
    """Optuna callable for LSTM hyperparameter search.

    Uses train + val loaders. Reports intermediate val Macro F1 per epoch so
    MedianPruner can cut bad trials early. Returns best val Macro F1.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        device: torch.device,
        max_epochs: int = 50,
        patience: int = 10,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

    def __call__(self, trial: optuna.Trial) -> float:
        set_seed(42)  # fixed seed: search measures HP variance, not seed variance

        # Sample hyperparameters
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        head_dropout = trial.suggest_float("head_dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Rebuild loaders with trial batch size (reuse same dataset objects)
        train_ds = self.train_loader.dataset
        val_ds = self.val_loader.dataset
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,  # avoid batch of 1 at end
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        model = FVGLSTMClassifier(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            head_dropout=head_dropout,
        ).to(self.device)

        criterion = WeightedCE(self.class_weights.to(self.device))
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        early_stop = EarlyStop(patience=self.patience, mode="max")

        best_val_f1 = 0.0

        for epoch in range(self.max_epochs):
            # Train one epoch
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = torch.as_tensor(y_batch, device=self.device)
                optimiser.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()

            # Evaluate on val
            val_f1 = _eval_macro_f1(model, val_loader, self.device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            # Report intermediate for pruner
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if early_stop.update(val_f1):
                break

        return best_val_f1


# ---------------------------------------------------------------------------
# XGBoost Objective
# ---------------------------------------------------------------------------

class XGBObjective:
    """Optuna callable for XGBoost hyperparameter search.

    Uses train + val feature matrices only.
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

        n_estimators = trial.suggest_int("n_estimators", 100, 600)
        max_depth = trial.suggest_categorical("max_depth", [3, 4, 5, 6])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric=["mlogloss"],
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


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(
    objective: LSTMObjective | XGBObjective,
    n_trials: int,
    storage_url: str,
    study_name: str,
    direction: str = "maximize",
    n_startup_trials: int = 5,
    n_warmup_steps: int = 10,
    timeout: int = 300,
) -> optuna.Study:
    """Create or load Optuna study from SQLite and run n_trials.

    Idempotent: if study already exists, resumes from existing trials.
    Uses MedianPruner to cut bad trials early.
    """
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    # Only run remaining trials
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials - n_done)

    if n_remaining > 0:
        study.optimize(
            objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            timeout=timeout,   # 5 min per trial max — prevents CPU/process hang
            n_jobs=1,          # always serial (XGB subprocess safety + MPS thread-safety)
        )
    else:
        print(f"Study '{study_name}' already has {n_done} completed trials — skipping.")

    best = study.best_trial
    print(f"\nBest trial #{best.number}: val Macro F1 = {best.value:.4f}")
    print("Best params:", json.dumps(best.params, indent=2))

    return study


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eval_macro_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Single-pass macro F1 on loader. No loss computation."""
    from sklearn.metrics import f1_score

    model.eval()
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_pred_all.extend(preds.tolist())
            if isinstance(y_batch, torch.Tensor):
                y_true_all.extend(y_batch.numpy().tolist())
            else:
                y_true_all.extend(list(y_batch))

    y_true = np.array(y_true_all, dtype=np.int64)
    y_pred = np.array(y_pred_all, dtype=np.int64)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
