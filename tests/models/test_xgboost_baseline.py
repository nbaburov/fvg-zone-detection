"""Tests for src/models/xgboost_baseline.py.

Run separately from the rest of the suite:

    .venv/bin/pytest tests/models/test_xgboost_baseline.py

Default `pytest` invocation skips this file via `pytest.ini` (`addopts`).
Reason: on macOS arm64, XGBoost.fit + torch in the same process segfaults
(libgomp interaction). Once any torch-importing test module is collected
in the same session, this file crashes. Process isolation via
pytest-forked / pytest-isolate fails because pytest's parent is
multi-threaded (fork unsafe). Cleanest solution: separate invocation.

The Makefile target `make test` runs both halves sequentially.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.xgboost_baseline import XGBoostFVGClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    """200-sample synthetic 3-class dataset."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 35)).astype(np.float32)
    y = rng.integers(0, 3, size=200)
    # Ensure at least one sample per class in both splits
    X_train, y_train = X[:160], y[:160]
    X_val, y_val = X[160:], y[160:]
    # Force class presence
    for cls in [0, 1, 2]:
        y_train[cls] = cls
        y_val[cls] = cls
    return X_train, y_train, X_val, y_val


@pytest.fixture
def trained_model(small_dataset):
    X_train, y_train, X_val, y_val = small_dataset
    model = XGBoostFVGClassifier()
    model.fit(X_train, y_train, X_val, y_val)
    return model, X_val


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation_no_error():
    model = XGBoostFVGClassifier()
    assert model is not None


def test_instantiation_custom_params():
    model = XGBoostFVGClassifier(params={"n_estimators": 10, "max_depth": 2})
    assert model is not None


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def test_fit_completes(small_dataset):
    X_train, y_train, X_val, y_val = small_dataset
    model = XGBoostFVGClassifier(params={"n_estimators": 20})
    model.fit(X_train, y_train, X_val, y_val)
    assert model.n_estimators_used is not None
    assert model.n_estimators_used > 0


def test_fit_with_sample_weight(small_dataset):
    X_train, y_train, X_val, y_val = small_dataset
    sw = np.ones(len(y_train), dtype=np.float32)
    sw[y_train == 1] = 2.0
    model = XGBoostFVGClassifier(params={"n_estimators": 10})
    model.fit(X_train, y_train, X_val, y_val, sample_weight=sw)
    assert model is not None


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def test_predict_shape_and_dtype(trained_model):
    model, X_val = trained_model
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert preds.dtype in (np.int32, np.int64, int)
    assert set(preds).issubset({0, 1, 2})


def test_predict_proba_shape(trained_model):
    model, X_val = trained_model
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 3)


def test_predict_proba_sums_to_one(trained_model):
    model, X_val = trained_model
    proba = model.predict_proba(X_val)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def test_save_creates_files(trained_model, tmp_path):
    model, _ = trained_model
    save_path = tmp_path / "xgb_test.ubj"
    model.save(save_path)
    assert save_path.exists()
    assert (tmp_path / "xgb_test.ubj.meta.json").exists()


def test_load_round_trip(trained_model, tmp_path):
    model, X_val = trained_model
    save_path = tmp_path / "xgb_roundtrip.ubj"
    model.save(save_path)

    loaded = XGBoostFVGClassifier.load(save_path)
    preds_original = model.predict(X_val)
    preds_loaded = loaded.predict(X_val)
    np.testing.assert_array_equal(preds_original, preds_loaded)


def test_meta_json_keys(trained_model, tmp_path):
    model, _ = trained_model
    save_path = tmp_path / "xgb_meta.ubj"
    model.save(save_path)

    with open(tmp_path / "xgb_meta.ubj.meta.json") as fh:
        meta = json.load(fh)

    required_keys = {"params", "git_sha", "python_version", "xgboost_version", "timestamp", "n_estimators_used"}
    assert required_keys.issubset(meta.keys()), f"Missing keys: {required_keys - meta.keys()}"
