"""tests/rigor/test_optuna_utils.py — Smoke tests for src/rigor/optuna_utils.py.

Runs 2 trials for both LSTMObjective and XGBObjective using tiny synthetic data.
Verifies study create/load idempotency and best params dict keys.

XGB tests use subprocess workers to match production behaviour and avoid the
macOS arm64 libgomp segfault when XGB.fit runs in a process that has imported
torch. The same pattern is used by scripts/rigor/_workers/_xgb_tune_worker.py
in production.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.rigor.optuna_utils import LSTMObjective, run_study

# Path to the project root (two levels up from tests/rigor/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_XGB_WORKER = _PROJECT_ROOT / "scripts" / "rigor" / "_workers" / "_xgb_tune_worker.py"


# ---------------------------------------------------------------------------
# Tiny synthetic datasets
# ---------------------------------------------------------------------------

def _make_lstm_loaders(n: int = 120, window: int = 60) -> tuple[DataLoader, DataLoader]:
    rng = torch.Generator().manual_seed(0)
    X = torch.randn(n, window, 5, generator=rng)
    y = torch.randint(0, 3, (n,), generator=rng)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    return loader, loader  # use same for train + val in smoke test


def _make_xgb_arrays(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 35)).astype(np.float32)
    y = rng.integers(0, 3, size=n)
    return X, y


def _run_xgb_study_subprocess(
    tmp_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    study_name: str,
) -> optuna.Study:
    """Run XGBObjective trials via subprocess worker (matches production pattern).

    Avoids the macOS arm64 libgomp segfault that occurs when xgb.fit() is
    called in-process after torch has been imported.
    """
    # Save arrays to npz (worker expects X_train, y_train, X_val, y_val, sample_weight)
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    npz_path = tmp_path / "data.npz"
    np.savez(npz_path, X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, sample_weight=sample_weight)

    db_url = f"sqlite:///{tmp_path}/{study_name}.db"
    output_config = tmp_path / f"{study_name}_best.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            str(_XGB_WORKER),
            "--data-npz", str(npz_path),
            "--storage-url", db_url,
            "--study-name", study_name,
            "--n-trials", str(n_trials),
            "--output-config", str(output_config),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_PROJECT_ROOT),
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"XGB worker failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    # Load study from the SQLite db that the worker wrote
    study = optuna.load_study(study_name=study_name, storage=db_url)
    return study


# ---------------------------------------------------------------------------
# LSTMObjective smoke test
# ---------------------------------------------------------------------------

def test_lstm_objective_runs_2_trials(tmp_path: Path) -> None:
    train_loader, val_loader = _make_lstm_loaders()
    weights = torch.tensor([1.0, 5.0, 5.0])
    device = torch.device("cpu")

    objective = LSTMObjective(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=weights,
        device=device,
        max_epochs=2,  # tiny for speed
        patience=5,
    )

    db_url = f"sqlite:///{tmp_path}/lstm_test.db"
    study = run_study(
        objective,
        n_trials=2,
        storage_url=db_url,
        study_name="lstm_smoke",
    )

    assert len(study.trials) >= 2
    best = study.best_trial
    assert best.value is not None
    assert 0.0 <= best.value <= 1.0


def test_lstm_objective_best_params_have_expected_keys(tmp_path: Path) -> None:
    train_loader, val_loader = _make_lstm_loaders()
    weights = torch.tensor([1.0, 5.0, 5.0])
    device = torch.device("cpu")

    objective = LSTMObjective(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=weights,
        device=device,
        max_epochs=1,
        patience=3,
    )

    db_url = f"sqlite:///{tmp_path}/lstm_keys.db"
    study = run_study(objective, n_trials=2, storage_url=db_url, study_name="lstm_keys")

    expected_keys = {
        "hidden_size", "num_layers", "dropout", "head_dropout",
        "lr", "weight_decay", "batch_size",
    }
    assert expected_keys.issubset(set(study.best_trial.params.keys()))


# ---------------------------------------------------------------------------
# XGBObjective smoke tests — use subprocess worker (matches production pattern)
# Avoids macOS arm64 libgomp segfault when xgb.fit runs after torch import.
# ---------------------------------------------------------------------------

def test_xgb_objective_runs_2_trials(tmp_path: Path) -> None:
    X_train, y_train = _make_xgb_arrays(100)
    X_val, y_val = _make_xgb_arrays(40)

    study = _run_xgb_study_subprocess(
        tmp_path, X_train, y_train, X_val, y_val,
        n_trials=2, study_name="xgb_smoke",
    )

    assert len(study.trials) >= 2
    best = study.best_trial
    assert best.value is not None
    assert 0.0 <= best.value <= 1.0


def test_xgb_objective_best_params_have_expected_keys(tmp_path: Path) -> None:
    X_train, y_train = _make_xgb_arrays(100)
    X_val, y_val = _make_xgb_arrays(40)

    study = _run_xgb_study_subprocess(
        tmp_path, X_train, y_train, X_val, y_val,
        n_trials=2, study_name="xgb_keys",
    )

    expected_keys = {
        "n_estimators", "max_depth", "learning_rate",
        "min_child_weight", "subsample", "colsample_bytree",
    }
    assert expected_keys.issubset(set(study.best_trial.params.keys()))


# ---------------------------------------------------------------------------
# Study idempotency (create / load)
# ---------------------------------------------------------------------------

def test_run_study_loads_existing_study(tmp_path: Path) -> None:
    """Running XGB worker twice on same db should NOT add more trials."""
    X_train, y_train = _make_xgb_arrays(60)
    X_val, y_val = _make_xgb_arrays(20)

    study1 = _run_xgb_study_subprocess(
        tmp_path, X_train, y_train, X_val, y_val,
        n_trials=2, study_name="idem_test",
    )
    n_after_first = len(study1.trials)

    # Re-run — worker's run_study_xgb is idempotent, should skip since n_done >= n_trials
    study2 = _run_xgb_study_subprocess(
        tmp_path, X_train, y_train, X_val, y_val,
        n_trials=2, study_name="idem_test",
    )
    n_after_second = len(study2.trials)

    assert n_after_second == n_after_first


def test_run_study_idempotency_lstm(tmp_path: Path) -> None:
    """LSTM variant: Running run_study twice should not add more trials."""
    train_loader, val_loader = _make_lstm_loaders()
    weights = torch.tensor([1.0, 5.0, 5.0])
    device = torch.device("cpu")

    objective = LSTMObjective(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=weights,
        device=device,
        max_epochs=1,
        patience=3,
    )

    db_url = f"sqlite:///{tmp_path}/lstm_idem.db"
    study1 = run_study(objective, n_trials=2, storage_url=db_url, study_name="lstm_idem")
    n1 = len(study1.trials)

    study2 = run_study(objective, n_trials=2, storage_url=db_url, study_name="lstm_idem")
    n2 = len(study2.trials)

    assert n2 == n1
