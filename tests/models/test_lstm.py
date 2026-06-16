"""Unit tests for the FVGLSTMClassifier model class.

Run separately from feature/data tests to avoid Python 3.14 + xgboost segfault:
    pytest tests/models/test_lstm.py -v

Tests cover:
  - FVGLSTMClassifier: shape, finite, batch=1, nn.LSTM check, param count, MPS
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.lstm import FVGLSTMClassifier


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


