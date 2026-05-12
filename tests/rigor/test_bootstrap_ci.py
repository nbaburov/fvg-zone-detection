"""tests/rigor/test_bootstrap_ci.py — Unit tests for src/rigor/bootstrap_ci.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.rigor.bootstrap_ci import block_bootstrap_f1, effective_n


# ---------------------------------------------------------------------------
# effective_n
# ---------------------------------------------------------------------------

def test_effective_n_standard() -> None:
    assert effective_n(7056, 60, 1) == 117


def test_effective_n_test_split() -> None:
    # 3514 test bars, window=60 -> 58 non-overlapping windows
    assert effective_n(3514, 60, 1) == 58


def test_effective_n_exact_multiple() -> None:
    assert effective_n(120, 60) == 2


def test_effective_n_smaller_than_window() -> None:
    assert effective_n(30, 60) == 0


# ---------------------------------------------------------------------------
# block_bootstrap_f1
# ---------------------------------------------------------------------------

def _make_predictions(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.choice([0, 1, 2], size=n, p=[0.8, 0.1, 0.1])
    # Predictions mostly correct with some noise
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, size=n // 5, replace=False)
    y_pred[flip_idx] = rng.choice([0, 1, 2], size=len(flip_idx))
    return y_true, y_pred


def test_block_bootstrap_returns_all_keys() -> None:
    y_true, y_pred = _make_predictions()
    result = block_bootstrap_f1(y_true, y_pred, block_size=60, n_iterations=100, seed=42)
    for key in ("macro_f1", "bull_f1", "bear_f1"):
        assert key in result
        assert "point" in result[key]
        assert "ci_lower" in result[key]
        assert "ci_upper" in result[key]
    assert "n_bootstrap" in result
    assert "block_size" in result
    assert "effective_n" in result


def test_block_bootstrap_ci_ordered() -> None:
    """CI lower <= point estimate <= CI upper."""
    y_true, y_pred = _make_predictions(n=600)
    result = block_bootstrap_f1(y_true, y_pred, block_size=60, n_iterations=200, seed=42)
    for key in ("macro_f1", "bull_f1", "bear_f1"):
        r = result[key]
        assert r["ci_lower"] <= r["point"] + 1e-9, f"{key}: lower > point"
        assert r["point"] <= r["ci_upper"] + 1e-9, f"{key}: point > upper"


def test_block_bootstrap_reproducible() -> None:
    y_true, y_pred = _make_predictions(n=360)
    r1 = block_bootstrap_f1(y_true, y_pred, block_size=60, n_iterations=50, seed=7)
    r2 = block_bootstrap_f1(y_true, y_pred, block_size=60, n_iterations=50, seed=7)
    assert r1["macro_f1"]["ci_lower"] == r2["macro_f1"]["ci_lower"]
    assert r1["macro_f1"]["ci_upper"] == r2["macro_f1"]["ci_upper"]


def test_block_bootstrap_raises_on_length_mismatch() -> None:
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="same length"):
        block_bootstrap_f1(y_true, y_pred)
