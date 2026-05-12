"""tests/rigor/test_optuna_utils.py — Smoke tests for src/rigor/optuna_utils.py.

Runs 2 trials for both LSTMObjective and XGBObjective using tiny synthetic data.
Verifies study create/load idempotency and best params dict keys.

Note: XGBObjective tests are skipped on Python 3.14 due to the known XGBoost
segfault on Python 3.14 (in-process inference crashes). The real tune_xgboost.py
script uses the _xgb_worker.py subprocess workaround. See CLAUDE.md.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.rigor.optuna_utils import LSTMObjective, XGBObjective, run_study

_XGB_SKIP = sys.version_info >= (3, 14)


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
# XGBObjective smoke test — skipped on Python 3.14 (XGB segfault, use subprocess)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_XGB_SKIP, reason="XGBoost segfaults in-process on Python 3.14")
def test_xgb_objective_runs_2_trials(tmp_path: Path) -> None:
    X_train, y_train = _make_xgb_arrays(100)
    X_val, y_val = _make_xgb_arrays(40)

    objective = XGBObjective(X_train, y_train, X_val, y_val)

    db_url = f"sqlite:///{tmp_path}/xgb_test.db"
    study = run_study(
        objective,
        n_trials=2,
        storage_url=db_url,
        study_name="xgb_smoke",
    )

    assert len(study.trials) >= 2
    best = study.best_trial
    assert best.value is not None
    assert 0.0 <= best.value <= 1.0


@pytest.mark.skipif(_XGB_SKIP, reason="XGBoost segfaults in-process on Python 3.14")
def test_xgb_objective_best_params_have_expected_keys(tmp_path: Path) -> None:
    X_train, y_train = _make_xgb_arrays(100)
    X_val, y_val = _make_xgb_arrays(40)

    objective = XGBObjective(X_train, y_train, X_val, y_val)

    db_url = f"sqlite:///{tmp_path}/xgb_keys.db"
    study = run_study(objective, n_trials=2, storage_url=db_url, study_name="xgb_keys")

    expected_keys = {
        "n_estimators", "max_depth", "learning_rate",
        "min_child_weight", "subsample", "colsample_bytree",
    }
    assert expected_keys.issubset(set(study.best_trial.params.keys()))


# ---------------------------------------------------------------------------
# Study idempotency (create / load)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_XGB_SKIP, reason="XGBoost segfaults in-process on Python 3.14")
def test_run_study_loads_existing_study(tmp_path: Path) -> None:
    """Running run_study twice on same db should NOT duplicate to > n_trials."""
    X_train, y_train = _make_xgb_arrays(60)
    X_val, y_val = _make_xgb_arrays(20)

    objective = XGBObjective(X_train, y_train, X_val, y_val)
    db_url = f"sqlite:///{tmp_path}/idempotent.db"

    study1 = run_study(objective, n_trials=2, storage_url=db_url, study_name="idem_test")
    n_after_first = len([t for t in study1.trials])

    # Re-run — should skip since n_done >= n_trials
    study2 = run_study(objective, n_trials=2, storage_url=db_url, study_name="idem_test")
    n_after_second = len([t for t in study2.trials])

    # Trial count should not grow beyond n_trials on second call
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
