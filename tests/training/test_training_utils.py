"""Unit tests for src/training/ utilities — EarlyStop, WeightedCE, set_seed.

Run separately from feature/data tests to avoid Python 3.14 + xgboost segfault:
    pytest tests/training/test_training_utils.py -v

Moved out of tests/models/test_lstm.py so the tests/ tree mirrors src/
(these target src/training/, not src/models/).
"""

from __future__ import annotations

import torch

from src.training.early_stop import EarlyStop
from src.training.loss import WeightedCE
from src.training.train_utils import set_seed


# ---------------------------------------------------------------------------
# EarlyStop tests
# ---------------------------------------------------------------------------


def test_early_stop_triggers():
    """EarlyStop triggers after patience epochs with no improvement."""
    es = EarlyStop(patience=3, min_delta=1e-4, ema_alpha=0.3)
    stopped = False
    for _ in range(4):
        stopped = es.update(0.5)
    assert stopped, "EarlyStop should trigger after patience exhausted"


def test_early_stop_does_not_trigger_too_early():
    """EarlyStop does not trigger before patience is exhausted."""
    es = EarlyStop(patience=5, min_delta=1e-4, ema_alpha=0.3)
    for i in range(4):
        stopped = es.update(0.5)
        assert not stopped, f"EarlyStop triggered too early at step {i+1}"


def test_early_stop_resets_on_improvement():
    """EarlyStop does not trigger when metric keeps improving."""
    es = EarlyStop(patience=3, min_delta=1e-4, ema_alpha=0.3)
    stopped = False
    for i in range(10):
        stopped = es.update(0.5 + i * 0.01)
    assert not stopped, "EarlyStop should not trigger on steady improvement"


def test_early_stop_smoothed_property():
    """smoothed property returns a float."""
    es = EarlyStop(patience=3, ema_alpha=0.5)
    es.update(0.4)
    es.update(0.6)
    assert isinstance(es.smoothed, float)
    assert 0.0 <= es.smoothed <= 1.0


def test_early_stop_best_property():
    """best property is non-negative."""
    es = EarlyStop(patience=5, ema_alpha=0.5, mode="max")
    es.update(0.5)
    assert es.best >= 0.0


def test_early_stop_min_mode():
    """EarlyStop works for loss minimisation."""
    es = EarlyStop(patience=3, min_delta=1e-4, ema_alpha=0.3, mode="min")
    stopped = False
    for _ in range(4):
        stopped = es.update(1.0)
    assert stopped, "EarlyStop (min mode) should trigger when loss is not decreasing"


# ---------------------------------------------------------------------------
# WeightedCE tests
# ---------------------------------------------------------------------------


def test_weighted_ce_returns_scalar():
    """WeightedCE returns a scalar tensor."""
    class_weights = torch.tensor([0.217, 1.097, 1.686])
    criterion = WeightedCE(class_weights)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    loss = criterion(logits, targets)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


def test_weighted_ce_finite():
    """WeightedCE loss is finite."""
    class_weights = torch.tensor([0.217, 1.097, 1.686])
    criterion = WeightedCE(class_weights)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    loss = criterion(logits, targets)
    assert torch.isfinite(loss), f"WeightedCE loss is not finite: {loss.item()}"


def test_weighted_ce_positive():
    """WeightedCE loss is positive."""
    class_weights = torch.tensor([0.217, 1.097, 1.686])
    criterion = WeightedCE(class_weights)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    loss = criterion(logits, targets)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# set_seed tests
# ---------------------------------------------------------------------------


def test_set_seed_runs():
    """set_seed(42) runs without error."""
    set_seed(42)


def test_set_seed_deterministic_output():
    """Two runs with same seed produce identical torch output."""
    set_seed(42)
    x1 = torch.randn(10)
    set_seed(42)
    x2 = torch.randn(10)
    assert torch.allclose(x1, x2), "set_seed should produce deterministic torch output"
