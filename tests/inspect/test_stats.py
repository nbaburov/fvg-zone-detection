"""test_stats.py — Unit tests for stats computation with known arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inspect.runner import InspectionResults
from src.inspect.stats import compute_stats, InspectionStats


def _make_results(
    y_true: list[int],
    preds_by_model: dict[str, list[int]],
) -> InspectionResults:
    """Build a minimal InspectionResults from label lists."""
    n = len(y_true)
    windows_raw = np.zeros((n, 60, 5), dtype=np.float32)
    windows_norm = np.zeros((n, 60, 5), dtype=np.float32)
    labels = np.array(y_true, dtype=np.int64)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="America/New_York")

    probas: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    for name, p in preds_by_model.items():
        pred_arr = np.array(p, dtype=np.int64)
        # Build one-hot probas from predictions
        proba_arr = np.zeros((n, 3), dtype=np.float32)
        for i, cls in enumerate(pred_arr):
            proba_arr[i, cls] = 1.0
        probas[name] = proba_arr
        preds[name] = pred_arr

    return InspectionResults(
        windows_raw=windows_raw,
        windows_norm=windows_norm,
        labels=labels,
        timestamps=pd.DatetimeIndex(timestamps),
        probas=probas,
        preds=preds,
    )


def test_empty_results_returns_zero_stats():
    results = InspectionResults(
        windows_raw=np.empty((0, 60, 5), dtype=np.float32),
        windows_norm=np.empty((0, 60, 5), dtype=np.float32),
        labels=np.empty(0, dtype=np.int64),
        timestamps=pd.DatetimeIndex([]),
    )
    stats = compute_stats(results)
    assert stats.n_windows == 0
    assert stats.per_model == {}
    assert stats.disagreement_index.shape == (0,)


def test_perfect_predictions_f1_is_one():
    """When predictions perfectly match labels, all F1 scores should be 1.0."""
    y = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
    preds = {"model_a": y[:]}
    results = _make_results(y, preds)
    stats = compute_stats(results)
    m = stats.per_model["model_a"]
    assert m.f1_bull == pytest.approx(1.0, abs=1e-5)
    assert m.f1_bear == pytest.approx(1.0, abs=1e-5)
    assert m.f1_fvg_macro == pytest.approx(1.0, abs=1e-5)
    assert m.f1_binary_fvg == pytest.approx(1.0, abs=1e-5)


def test_all_predictions_wrong_f1_is_zero():
    """When model always predicts none (0) on all-positive labels, binary FVG F1 = 0."""
    y = [1, 1, 2, 2, 1]
    preds = {"model_b": [0, 0, 0, 0, 0]}
    results = _make_results(y, preds)
    stats = compute_stats(results)
    m = stats.per_model["model_b"]
    assert m.f1_binary_fvg == pytest.approx(0.0, abs=1e-5)


def test_confusion_matrix_shape_and_values():
    """Spot-check confusion matrix values with known ground truth."""
    y = [0, 0, 1, 1, 2, 2]
    # Model gets 0s right, but confuses 1s with 2s
    p = [0, 0, 2, 2, 1, 1]
    results = _make_results(y, {"m": p})
    stats = compute_stats(results)
    cm = stats.per_model["m"].confusion
    assert cm.shape == (3, 3)
    # true=0, pred=0: 2 correct
    assert cm[0, 0] == 2
    # true=1, pred=2: 2 misclassified
    assert cm[1, 2] == 2
    # true=2, pred=1: 2 misclassified
    assert cm[2, 1] == 2


def test_agreement_matrix_two_models_identical():
    """Two models with identical predictions should have agreement = 1.0."""
    y = [0, 1, 2, 0, 1]
    p = [0, 1, 2, 0, 1]
    results = _make_results(y, {"m1": p, "m2": p[:]})
    stats = compute_stats(results)
    assert stats.agreement_matrix.shape == (2, 2)
    assert stats.agreement_matrix[0, 1] == pytest.approx(1.0)
    assert stats.agreement_matrix[1, 0] == pytest.approx(1.0)


def test_agreement_matrix_two_models_fully_different():
    """Models that always disagree should have off-diagonal = 0.0."""
    y = [0, 1, 2, 0, 1]
    p1 = [0, 0, 0, 0, 0]
    p2 = [1, 2, 1, 2, 2]
    results = _make_results(y, {"m1": p1, "m2": p2})
    stats = compute_stats(results)
    assert stats.agreement_matrix[0, 1] == pytest.approx(0.0)


def test_agreement_matrix_partial():
    """Models that agree on 3/5 windows → off-diagonal = 0.6."""
    y = [0, 1, 2, 0, 1]
    p1 = [0, 1, 2, 1, 0]
    p2 = [0, 1, 2, 0, 1]  # agree on indices 0, 1, 2 → 3/5 = 0.6
    results = _make_results(y, {"m1": p1, "m2": p2})
    stats = compute_stats(results)
    assert stats.agreement_matrix[0, 1] == pytest.approx(0.6, abs=1e-5)


def test_disagreement_index_all_agree_is_zero():
    y = [0, 1, 2]
    p = [0, 1, 2]
    results = _make_results(y, {"m1": p, "m2": p[:]})
    stats = compute_stats(results)
    assert np.all(stats.disagreement_index == 0.0)


def test_disagreement_index_all_differ_is_one():
    """For 2 models with 2 unique classes, disagreement = 1.0 when they differ."""
    y = [0, 1, 2]
    p1 = [0, 0, 0]
    p2 = [1, 2, 1]  # always different from p1
    results = _make_results(y, {"m1": p1, "m2": p2})
    stats = compute_stats(results)
    # All windows: 2 unique classes out of max 2 → (2-1)/(2-1) = 1.0
    assert np.all(stats.disagreement_index == pytest.approx(1.0))


def test_no_positive_labels_sets_warning():
    y = [0, 0, 0, 0, 0]
    p = [0, 0, 0, 0, 0]
    results = _make_results(y, {"m": p})
    stats = compute_stats(results)
    assert stats.warning_no_positives is True
    assert stats.n_positive == 0


def test_single_model_no_agreement_matrix():
    y = [0, 1, 2]
    results = _make_results(y, {"solo": [0, 1, 2]})
    stats = compute_stats(results)
    # No agreement matrix for single model
    assert stats.agreement_matrix.size == 0 or stats.agreement_matrix.shape == (0, 0)
