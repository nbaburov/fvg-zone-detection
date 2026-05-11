"""stats.py — Metric computation for model inspection results."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.inspect.runner import InspectionResults

CLASS_NAMES = ["none", "bullish", "bearish"]


@dataclass
class ModelMetrics:
    """Per-model metrics."""

    name: str
    f1_bull: float          # F1 for class 1 (bullish)
    f1_bear: float          # F1 for class 2 (bearish)
    f1_fvg_macro: float     # macro F1 across classes 1+2 (FVG-present classes)
    f1_binary_fvg: float    # binary F1: FVG present (1|2) vs none (0)
    precision_bull: float
    precision_bear: float
    recall_bull: float
    recall_bear: float
    confusion: np.ndarray   # (3, 3) confusion matrix


@dataclass
class InspectionStats:
    """Aggregated stats across all models."""

    per_model: dict[str, ModelMetrics] = field(default_factory=dict)
    agreement_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    model_names: list[str] = field(default_factory=list)
    disagreement_index: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_windows: int = 0
    n_positive: int = 0
    warning_no_positives: bool = False


def compute_stats(results: InspectionResults) -> InspectionStats:
    """Compute all metrics from inspection results.

    Parameters
    ----------
    results : InspectionResults

    Returns
    -------
    InspectionStats
    """
    stats = InspectionStats(n_windows=results.n)

    if results.n == 0:
        return stats

    y_true = results.labels
    stats.n_positive = int(np.sum(y_true != 0))

    if stats.n_positive == 0:
        stats.warning_no_positives = True

    model_names = results.model_names
    stats.model_names = model_names

    for name in model_names:
        y_pred = results.preds[name]
        stats.per_model[name] = _compute_model_metrics(name, y_true, y_pred)

    if len(model_names) >= 2:
        stats.agreement_matrix = _compute_agreement_matrix(
            model_names, results.preds
        )

    stats.disagreement_index = _compute_disagreement_index(
        model_names, results.preds, results.n
    )

    return stats


def _compute_model_metrics(
    name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> ModelMetrics:
    labels_all = [0, 1, 2]

    def _safe_f1(average, labels=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
            )

    def _safe_precision(label):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                precision_score(
                    y_true, y_pred, labels=[label], average="micro", zero_division=0
                )
            )

    def _safe_recall(label):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                recall_score(
                    y_true, y_pred, labels=[label], average="micro", zero_division=0
                )
            )

    f1_per = _safe_f1("weighted", labels=None)
    # Per-class F1
    f1_all = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_f1 = f1_score(y_true, y_pred, labels=labels_all, average=None, zero_division=0)
    for idx, cls in enumerate(labels_all):
        f1_all[cls] = float(raw_f1[idx]) if idx < len(raw_f1) else 0.0

    f1_bull = f1_all.get(1, 0.0)
    f1_bear = f1_all.get(2, 0.0)
    f1_fvg_macro = float(np.mean([f1_bull, f1_bear]))

    # Binary FVG-vs-none
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_binary_fvg = float(
            f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    return ModelMetrics(
        name=name,
        f1_bull=f1_bull,
        f1_bear=f1_bear,
        f1_fvg_macro=f1_fvg_macro,
        f1_binary_fvg=f1_binary_fvg,
        precision_bull=_safe_precision(1),
        precision_bear=_safe_precision(2),
        recall_bull=_safe_recall(1),
        recall_bear=_safe_recall(2),
        confusion=cm,
    )


def _compute_agreement_matrix(
    model_names: list[str], preds: dict[str, np.ndarray]
) -> np.ndarray:
    """(n_models, n_models) fraction of windows where model_i and model_j agree."""
    n = len(model_names)
    matrix = np.ones((n, n), dtype=np.float32)
    n_windows = len(next(iter(preds.values())))
    if n_windows == 0:
        return matrix

    for i in range(n):
        for j in range(i + 1, n):
            agree_frac = float(
                np.mean(preds[model_names[i]] == preds[model_names[j]])
            )
            matrix[i, j] = agree_frac
            matrix[j, i] = agree_frac
    return matrix


def _compute_disagreement_index(
    model_names: list[str], preds: dict[str, np.ndarray], n_windows: int
) -> np.ndarray:
    """Per-window disagreement index in [0, 1].

    For each window: fraction of unique predicted classes among all models
    relative to maximum possible disagreement.
    Returns zeros if <=1 model loaded.
    """
    if len(model_names) <= 1 or n_windows == 0:
        return np.zeros(n_windows, dtype=np.float32)

    stacked = np.stack(
        [preds[name] for name in model_names], axis=1
    )  # (N, n_models)

    disagreement = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        row = stacked[i]
        n_unique = len(np.unique(row))
        # Fraction: 0 = all agree, 1 = all differ (max = min(n_models, 3) unique classes)
        max_possible = min(len(model_names), 3)
        disagreement[i] = (n_unique - 1) / max(max_possible - 1, 1)

    return disagreement
