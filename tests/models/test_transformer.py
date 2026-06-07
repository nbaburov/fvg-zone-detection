"""Unit tests for FVGTransformerClassifier."""

from __future__ import annotations

import pytest
import torch

from src.models.transformer import FVGTransformerClassifier


@pytest.fixture
def default_model() -> FVGTransformerClassifier:
    return FVGTransformerClassifier(
        input_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        head_dropout=0.3,
    )


def test_forward_shape(default_model: FVGTransformerClassifier) -> None:
    """Input (B, 60, 5) -> output (B, 3) — no error."""
    x = torch.randn(2, 60, 5)
    out = default_model(x)
    assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"


def test_no_nan_on_random_input(default_model: FVGTransformerClassifier) -> None:
    """Output must be finite for random input — no NaN or Inf."""
    default_model.eval()
    x = torch.randn(4, 60, 5)
    with torch.no_grad():
        out = default_model(x)
    assert not torch.isnan(out).any(), "NaN detected in output"
    assert not torch.isinf(out).any(), "Inf detected in output"


def test_deterministic_under_fixed_seed() -> None:
    """Same seed -> identical outputs on two forward passes."""
    torch.manual_seed(0)
    model = FVGTransformerClassifier()
    model.eval()
    x = torch.randn(3, 60, 5)
    with torch.no_grad():
        out1 = model(x)
    with torch.no_grad():
        out2 = model(x)
    assert torch.allclose(out1, out2), "Outputs differ across identical forward passes"


def test_num_params_default(default_model: FVGTransformerClassifier) -> None:
    """Default HP (d_model=64, nhead=4, num_layers=2, ff=128) should be in [10k, 500k]."""
    n_params = sum(p.numel() for p in default_model.parameters())
    assert 10_000 <= n_params <= 500_000, (
        f"Param count {n_params:,} outside expected range [10k, 500k]"
    )


def test_single_sample_forward() -> None:
    """batch_size=1 should work in eval mode."""
    model = FVGTransformerClassifier()
    model.eval()
    x = torch.randn(1, 60, 5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3), f"Expected (1, 3), got {out.shape}"


def test_cls_pool_forward_shape() -> None:
    """pool='cls' variant produces the same output shape."""
    model = FVGTransformerClassifier(pool="cls")
    model.eval()
    x = torch.randn(2, 60, 5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3), f"Expected (2, 3) with cls pool, got {out.shape}"


def test_invalid_pool_raises() -> None:
    """pool must be 'mean' or 'cls'; anything else raises ValueError."""
    with pytest.raises(ValueError, match="pool must be"):
        FVGTransformerClassifier(pool="last")
