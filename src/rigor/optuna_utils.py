"""optuna_utils.py — Optuna objectives + study runner for torch models (LSTM/CNN-LSTM/Transformer).

(XGBoost objective lives in optuna_xgb.py — subprocess-isolated to dodge the arm64 torch+libgomp segfault.)

Design constraints:
  - LSTMObjective: uses train + val only. Test parquet NEVER loaded.
  - MedianPruner: fires after each epoch.
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

from src.models.cnn_lstm import FVGCNNLSTMClassifier
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
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
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
# CNN-LSTM Objective
# ---------------------------------------------------------------------------

class CNNLSTMObjective:
    """Optuna callable for CNN-LSTM hyperparameter search (11-dimensional).

    Uses train + val loaders only. Test parquet NEVER loaded here.
    Reports intermediate val Macro F1 per epoch so MedianPruner can prune.
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

        # Sample hyperparameters — 11-dimensional space
        n_conv_layers = trial.suggest_categorical("n_conv_layers", [1, 2, 3])
        conv_filters = trial.suggest_categorical("conv_filters", [16, 32, 64])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        use_pool = trial.suggest_categorical("use_pool", [False, True])
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
        lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        head_dropout = trial.suggest_float("head_dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Rebuild loaders with trial batch size
        train_ds = self.train_loader.dataset
        val_ds = self.val_loader.dataset
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        model = FVGCNNLSTMClassifier(
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            n_conv_layers=n_conv_layers,
            use_pool=bool(use_pool),
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            head_dropout=head_dropout,
        ).to(self.device)

        criterion = WeightedCE(self.class_weights.to(self.device))
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        early_stop = EarlyStop(patience=self.patience, mode="max")

        best_val_f1 = 0.0

        for epoch in range(self.max_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = torch.as_tensor(y_batch, device=self.device)
                optimiser.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()

            val_f1 = _eval_macro_f1(model, val_loader, self.device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if early_stop.update(val_f1):
                break

        return best_val_f1


# ---------------------------------------------------------------------------
# Transformer Objective
# ---------------------------------------------------------------------------

class TransformerObjective:
    """Optuna callable for Transformer-encoder hyperparameter search.

    Uses train + val loaders only. Test parquet NEVER loaded here.
    Reports intermediate val Macro F1 per epoch so MedianPruner can prune.

    Applies linear LR warmup + gradient clipping inside the trial loop — the
    same stabilisation the gated transformer-only path in seed_sweep uses — so
    the search measures HP under the regime the 5-seed run will train in.
    nhead is constrained to divide d_model (TransformerEncoderLayer requirement).
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
        from src.models.transformer import FVGTransformerClassifier

        set_seed(42)  # fixed seed: search measures HP variance, not seed variance

        # Sample hyperparameters
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        # nhead must divide d_model — restrict the candidate set per d_model.
        valid_heads = [h for h in (2, 4, 8) if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", valid_heads)
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        head_dropout = trial.suggest_float("head_dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        warmup_steps = trial.suggest_categorical("warmup_steps", [0, 100, 500])
        pool = trial.suggest_categorical("pool", ["mean", "cls"])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_ds = self.train_loader.dataset
        val_ds = self.val_loader.dataset
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        model = FVGTransformerClassifier(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            head_dropout=head_dropout,
            pool=pool,
        ).to(self.device)

        criterion = WeightedCE(self.class_weights.to(self.device))
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        early_stop = EarlyStop(patience=self.patience, mode="max")

        max_grad_norm = 1.0  # fixed clip — matches seed_sweep transformer default
        base_lr = lr
        global_step = 0
        best_val_f1 = 0.0

        for epoch in range(self.max_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = torch.as_tensor(y_batch, device=self.device)
                optimiser.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if warmup_steps > 0 and global_step < warmup_steps:
                    warm_lr = base_lr * float(global_step + 1) / float(warmup_steps)
                    for pg in optimiser.param_groups:
                        pg["lr"] = warm_lr
                elif warmup_steps > 0:
                    for pg in optimiser.param_groups:
                        pg["lr"] = base_lr
                optimiser.step()
                global_step += 1

            val_f1 = _eval_macro_f1(model, val_loader, self.device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if early_stop.update(val_f1):
                break

        return best_val_f1


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(
    objective: LSTMObjective | CNNLSTMObjective | TransformerObjective,
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
        # timeout here is per-trial budget in seconds; total budget = n_remaining * timeout.
        # Pass None to disable total timeout and rely solely on n_trials count.
        # Individual trial runtime is bounded by max_epochs * epoch_time.
        study.optimize(
            objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            timeout=None,      # no total wall-clock limit; rely on n_trials count
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
