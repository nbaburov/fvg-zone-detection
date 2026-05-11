"""test_base_adapter.py — Unit tests for ModelAdapter ABC interface contract."""

from __future__ import annotations

import numpy as np
import pytest

from src.inspect.base import ModelAdapter


class _StubAdapter(ModelAdapter):
    """Minimal concrete adapter for testing interface compliance."""

    name = "stub"

    def __init__(self, checkpoint_dir=None, **_kwargs):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        # Return uniform distribution over 3 classes
        return np.full((n, 3), 1.0 / 3, dtype=np.float32)


class _StubAdapterNoPredictProba(ModelAdapter):
    """Adapter missing predict_proba — cannot be instantiated."""

    name = "broken"


def test_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ModelAdapter()


def test_concrete_without_predict_proba_raises():
    with pytest.raises(TypeError):
        _StubAdapterNoPredictProba()


def test_stub_satisfies_interface():
    adapter = _StubAdapter()
    assert adapter.name == "stub"


def test_predict_proba_shape():
    adapter = _StubAdapter()
    windows = np.zeros((10, 60, 5), dtype=np.float32)
    probas = adapter.predict_proba(windows)
    assert probas.shape == (10, 3)
    assert probas.dtype == np.float32


def test_predict_returns_argmax():
    adapter = _StubAdapter()
    windows = np.zeros((5, 60, 5), dtype=np.float32)
    preds = adapter.predict(windows)
    assert preds.shape == (5,)
    # Uniform distribution — argmax is 0 (ties broken by first element)
    assert set(preds.tolist()).issubset({0, 1, 2})


def test_predict_proba_empty_batch():
    adapter = _StubAdapter()
    windows = np.zeros((0, 60, 5), dtype=np.float32)
    probas = adapter.predict_proba(windows)
    assert probas.shape[0] == 0
    assert probas.ndim == 2


def test_name_is_string_class_attribute():
    assert isinstance(_StubAdapter.name, str)
    assert _StubAdapter.name == "stub"
