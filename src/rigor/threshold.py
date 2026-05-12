"""threshold.py — Per-class decision threshold optimisation for FVG classifier.

Functions:
  compute_pr_curves   — one-vs-rest PR curves per class
  find_f1_optimal_threshold — F1-maximising threshold from PR curve
  apply_thresholds    — apply per-class thresholds to probability matrix
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


def compute_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute one-vs-rest precision-recall curves for each class.

    Parameters
    ----------
    y_true : (N,) integer labels in {0, 1, 2}
    y_proba : (N, 3) float probability matrix (rows sum to ~1)

    Returns
    -------
    dict[class_id -> (precision, recall, thresholds)]
      Note: precision and recall have one more element than thresholds
      (sklearn convention — last element is precision=1, recall=0).
    """
    n_classes = y_proba.shape[1]
    curves: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for cls in range(n_classes):
        binary_true = (y_true == cls).astype(int)
        scores = y_proba[:, cls]

        if binary_true.sum() == 0:
            logger.warning(
                "Class %d has no positive samples in y_true — "
                "PR curve is undefined. Returning trivial arrays.",
                cls,
            )
            curves[cls] = (
                np.array([1.0, 1.0]),
                np.array([0.0, 0.0]),
                np.array([0.5]),
            )
            continue

        precision, recall, thresholds = precision_recall_curve(binary_true, scores)
        curves[cls] = (precision, recall, thresholds)

    return curves


def find_f1_optimal_threshold(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """Find threshold that maximises F1 score on the PR curve.

    Parameters
    ----------
    precision, recall, thresholds : outputs of precision_recall_curve
      (precision/recall have one extra element vs thresholds — sklearn convention)

    Returns
    -------
    (optimal_threshold, f1_at_threshold)
    """
    # Use only the paired elements (drop the final precision=1, recall=0 point)
    p = precision[:-1]
    r = recall[:-1]

    denom = p + r
    # Avoid div-by-zero where both precision and recall are 0
    f1_scores = np.where(denom > 0, 2.0 * p * r / denom, 0.0)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def apply_thresholds(
    y_proba: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Apply per-class thresholds to a probability matrix and return integer labels.

    Decision rule:
      1. Mask out any class whose probability is below its threshold (set to -inf).
      2. Take argmax of masked probabilities.
      3. If all classes are masked (none exceed threshold), fall back to class 0 (none).

    Parameters
    ----------
    y_proba : (N, C) float
    thresholds : {class_id: threshold_value} for each class

    Returns
    -------
    (N,) integer label array
    """
    n_samples, n_classes = y_proba.shape
    masked = np.copy(y_proba)

    for cls, thr in thresholds.items():
        if thr <= 0:
            continue
        below = y_proba[:, cls] < thr
        masked[below, cls] = -np.inf

    # Check which samples have at least one class exceeding threshold
    any_valid = np.any(masked > -np.inf, axis=1)

    # Argmax on masked — returns highest-proba eligible class
    preds = masked.argmax(axis=1).astype(int)

    # Fallback: if no class exceeds threshold, assign class 0
    preds[~any_valid] = 0
    return preds
