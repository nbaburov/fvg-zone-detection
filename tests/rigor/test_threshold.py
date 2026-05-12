"""tests/rigor/test_threshold.py — Unit tests for src/rigor/threshold.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.rigor.threshold import (
    apply_thresholds,
    compute_pr_curves,
    find_f1_optimal_threshold,
)


# ---------------------------------------------------------------------------
# compute_pr_curves
# ---------------------------------------------------------------------------

def _make_proba_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """Make a small synthetic dataset where class 1 is separable."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = np.array([0] * 120 + [1] * 50 + [2] * 30)
    # Class 1 scores high on column 1, others score low
    y_proba = rng.dirichlet(alpha=[5, 1, 1], size=n)
    # Force class 1 samples to have high class-1 proba
    y_proba[120:170] = rng.dirichlet(alpha=[1, 10, 1], size=50)
    # Renormalise
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    return y_true, y_proba


def test_compute_pr_curves_returns_all_classes() -> None:
    y_true, y_proba = _make_proba_and_labels()
    curves = compute_pr_curves(y_true, y_proba)
    assert set(curves.keys()) == {0, 1, 2}


def test_compute_pr_curves_shapes_consistent() -> None:
    y_true, y_proba = _make_proba_and_labels()
    curves = compute_pr_curves(y_true, y_proba)
    for cls, (prec, rec, thr) in curves.items():
        # sklearn convention: len(prec) == len(rec) == len(thr) + 1
        assert len(prec) == len(rec), f"class {cls}: precision/recall length mismatch"
        assert len(prec) == len(thr) + 1, f"class {cls}: thresholds should be 1 shorter"


def test_compute_pr_curves_no_positive_class_returns_trivial() -> None:
    """Class with zero positives should not crash — returns trivial arrays."""
    y_true = np.zeros(50, dtype=int)  # only class 0
    y_proba = np.zeros((50, 3), dtype=float)
    y_proba[:, 0] = 1.0
    curves = compute_pr_curves(y_true, y_proba)
    # Classes 1 and 2 have no positives — must still return something
    assert 1 in curves
    assert 2 in curves


# ---------------------------------------------------------------------------
# find_f1_optimal_threshold
# ---------------------------------------------------------------------------

def test_find_f1_optimal_threshold_returns_valid_range() -> None:
    from sklearn.metrics import precision_recall_curve
    y_true, y_proba = _make_proba_and_labels()
    binary = (y_true == 1).astype(int)
    prec, rec, thr = precision_recall_curve(binary, y_proba[:, 1])
    optimal_thr, optimal_f1 = find_f1_optimal_threshold(prec, rec, thr)
    assert 0.0 <= optimal_thr <= 1.0
    assert 0.0 <= optimal_f1 <= 1.0


def test_find_f1_optimal_threshold_correct_on_perfect_separator() -> None:
    """When class 1 has perfect score gap, optimal threshold should be < 0.9."""
    rng = np.random.default_rng(1)
    y_true = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.uniform(0, 0.3, 50), rng.uniform(0.7, 1.0, 50)])
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, scores)
    opt_thr, opt_f1 = find_f1_optimal_threshold(prec, rec, thr)
    assert opt_f1 > 0.8, f"Expected high F1 on perfect separator, got {opt_f1}"


# ---------------------------------------------------------------------------
# apply_thresholds
# ---------------------------------------------------------------------------

def test_apply_thresholds_basic() -> None:
    """Class 0 blocked by high threshold; class 1 and 2 exceed their thresholds."""
    y_proba = np.array([
        [0.8, 0.6, 0.1],  # class 0: 0.8 < 0.9 → blocked; class 1: 0.6 >= 0.3 → wins
        [0.1, 0.8, 0.1],  # class 1: 0.8 >= 0.3 → wins (highest eligible)
        [0.1, 0.1, 0.8],  # class 2: 0.8 >= 0.3 → wins
    ], dtype=float)
    thresholds = {0: 0.9, 1: 0.3, 2: 0.3}
    preds = apply_thresholds(y_proba, thresholds)
    assert preds[0] == 1, f"Expected class 1 (class 0 blocked by threshold), got {preds[0]}"
    assert preds[1] == 1, f"Expected class 1, got {preds[1]}"
    assert preds[2] == 2, f"Expected class 2, got {preds[2]}"


def test_apply_thresholds_fallback_to_class0_when_all_below() -> None:
    """If no class exceeds threshold, prediction should be 0 (none)."""
    y_proba = np.array([[0.4, 0.3, 0.3]], dtype=float)
    thresholds = {0: 0.9, 1: 0.9, 2: 0.9}
    preds = apply_thresholds(y_proba, thresholds)
    assert preds[0] == 0


def test_apply_thresholds_output_shape() -> None:
    rng = np.random.default_rng(42)
    y_proba = rng.dirichlet([1, 1, 1], size=100)
    thresholds = {0: 0.5, 1: 0.5, 2: 0.5}
    preds = apply_thresholds(y_proba, thresholds)
    assert preds.shape == (100,)
    assert set(preds).issubset({0, 1, 2})
