"""Unit tests for LSTM model class and training infrastructure.

Run separately from feature/data tests to avoid Python 3.14 + xgboost segfault:
    pytest tests/models/test_lstm.py -v

Tests cover:
  - FVGLSTMClassifier: shape, finite, batch=1, nn.LSTM check, param count, MPS
  - EarlyStop: triggers after patience, resets on improvement
  - WeightedCE: returns scalar, finite
  - set_seed: runs without error
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.lstm import FVGLSTMClassifier
from src.training.early_stop import EarlyStop
from src.training.loss import WeightedCE
from src.training.train_utils import set_seed


# ---------------------------------------------------------------------------
# FVGLSTMClassifier tests
# ---------------------------------------------------------------------------


def test_forward_shape_cpu():
    """Forward pass on CPU returns (B, 3)."""
    model = FVGLSTMClassifier()
    model.eval()
    x = torch.randn(8, 60, 5)
    logits = model(x)
    assert logits.shape == (8, 3), f"Expected (8, 3), got {logits.shape}"


def test_forward_finite():
    """All logits are finite (no NaN or Inf)."""
    model = FVGLSTMClassifier()
    model.eval()
    x = torch.randn(8, 60, 5)
    logits = model(x)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


def test_forward_single_sample():
    """Works with batch size 1."""
    model = FVGLSTMClassifier()
    model.eval()
    x = torch.randn(1, 60, 5)
    logits = model(x)
    assert logits.shape == (1, 3), f"Expected (1, 3), got {logits.shape}"


def test_uses_nn_lstm_not_cell():
    """Confirms nn.LSTM is used, not nn.LSTMCell (slower on MPS per CLAUDE.md)."""
    model = FVGLSTMClassifier()
    assert isinstance(model.lstm, nn.LSTM), "Must use nn.LSTM not nn.LSTMCell"


def test_parameter_count_under_limit():
    """Total params < 100k."""
    model = FVGLSTMClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, f"Parameter count {n_params} exceeds 100k limit"


def test_parameter_count_reasonable_minimum():
    """Total params > 1k — confirm the model is non-trivial."""
    model = FVGLSTMClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 1_000, f"Parameter count {n_params} suspiciously low"


def test_output_not_softmax():
    """Output is raw logits, not softmax probabilities — check rows don't sum to 1."""
    torch.manual_seed(0)
    model = FVGLSTMClassifier()
    model.eval()
    x = torch.randn(16, 60, 5)
    logits = model(x)
    # If softmax, every row would sum to 1.0 with non-negative entries. Logits won't.
    row_sums = logits.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        "Outputs look like softmax probabilities (rows sum to 1) — expected raw logits"
    )


def test_forward_large_batch():
    """Handles batch size 256 (val/test DataLoader batch size)."""
    model = FVGLSTMClassifier()
    model.eval()
    x = torch.randn(256, 60, 5)
    logits = model(x)
    assert logits.shape == (256, 3)


def test_bidirectional_false():
    """Confirms LSTM is unidirectional."""
    model = FVGLSTMClassifier()
    assert not model.lstm.bidirectional, "LSTM must be unidirectional"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine",
)
def test_forward_mps():
    """Forward pass on MPS device returns finite output."""
    device = torch.device("mps")
    model = FVGLSTMClassifier().to(device)
    model.eval()
    x = torch.randn(8, 60, 5, device=device)
    logits = model(x)
    assert logits.shape == (8, 3)
    assert torch.isfinite(logits).all(), "MPS forward pass produced NaN or Inf"


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
