"""Unit tests for FVGCNNLSTMClassifier."""

from __future__ import annotations

import pytest
import torch

from src.models.cnn_lstm import FVGCNNLSTMClassifier


@pytest.fixture
def default_model() -> FVGCNNLSTMClassifier:
    return FVGCNNLSTMClassifier(
        conv_filters=32,
        kernel_size=3,
        n_conv_layers=2,
        use_pool=False,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.318,
        head_dropout=0.526,
    )


def test_forward_shape(default_model: FVGCNNLSTMClassifier) -> None:
    """Input (B, 60, 5) -> output (B, 3) — no error."""
    x = torch.randn(2, 60, 5)
    out = default_model(x)
    assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"


def test_gradient_flows(default_model: FVGCNNLSTMClassifier) -> None:
    """loss.backward() without NaN; all params should have gradients."""
    default_model.train()
    x = torch.randn(4, 60, 5)
    y = torch.randint(0, 3, (4,))
    logits = default_model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    for name, param in default_model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    assert not torch.isnan(loss), "NaN loss detected"


def test_num_params_default(default_model: FVGCNNLSTMClassifier) -> None:
    """Default HP (F=32, k=3, H=64, n_layers=2) should be ~29k params."""
    n_params = sum(p.numel() for p in default_model.parameters())
    assert 20_000 <= n_params <= 200_000, (
        f"Param count {n_params:,} outside expected range [20k, 200k]"
    )


def test_single_sample_forward() -> None:
    """batch_size=1 should work — BN handles single sample in inference mode."""
    model = FVGCNNLSTMClassifier()
    model.eval()
    x = torch.randn(1, 60, 5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3), f"Expected (1, 3), got {out.shape}"
