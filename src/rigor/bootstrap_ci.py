"""bootstrap_ci.py — Block bootstrap confidence intervals for test metrics.

Block bootstrap accounts for autocorrelation in time-series predictions.
Default block size = window_size (60 bars) = one non-overlapping window.

Effective n ≈ n_bars // window_size (non-overlapping windows).
For 3514 test bars and window=60: effective_n = 58.
CIs will be wide — this is honest and expected per plan.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def effective_n(n_bars: int, window_size: int, stride: int = 1) -> int:
    """Compute effective independent sample count.

    Uses non-overlapping window count: n_bars // window_size.
    stride parameter is accepted for interface completeness but not used
    (overlapping windows are not independent).
    """
    return n_bars // window_size


def block_bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 60,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Block bootstrap 95% confidence intervals for macro, bull, and bear F1.

    Parameters
    ----------
    y_true : (N,) integer true labels
    y_pred : (N,) integer predicted labels
    block_size : int — length of each contiguous block (default 60 = one window)
    n_iterations : int — number of bootstrap resamplings
    seed : int — RNG seed for reproducibility

    Returns
    -------
    {
        "macro_f1":  {"point": float, "ci_lower": float, "ci_upper": float},
        "bull_f1":   {"point": float, "ci_lower": float, "ci_upper": float},
        "bear_f1":   {"point": float, "ci_lower": float, "ci_upper": float},
        "n_bootstrap": int,
        "block_size": int,
        "effective_n": int,
    }
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}."
        )

    n = len(y_true)
    rng = np.random.default_rng(seed)

    # Point estimates on full test set
    point_macro = _macro_f1(y_true, y_pred)
    point_per_class = _per_class_f1(y_true, y_pred)

    # Build list of block start indices
    n_blocks = max(1, n // block_size)
    block_starts = np.arange(0, n - block_size + 1)

    bootstrap_macro: list[float] = []
    bootstrap_bull: list[float] = []
    bootstrap_bear: list[float] = []

    for _ in range(n_iterations):
        # Sample n_blocks block starts with replacement
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)

        indices = np.concatenate([
            np.arange(start, min(start + block_size, n))
            for start in chosen
        ])

        bt_true = y_true[indices]
        bt_pred = y_pred[indices]

        bootstrap_macro.append(_macro_f1(bt_true, bt_pred))
        pcf = _per_class_f1(bt_true, bt_pred)
        bootstrap_bull.append(pcf[1])
        bootstrap_bear.append(pcf[2])

    macro_arr = np.array(bootstrap_macro)
    bull_arr = np.array(bootstrap_bull)
    bear_arr = np.array(bootstrap_bear)

    return {
        "macro_f1": {
            "point": point_macro,
            "ci_lower": float(np.percentile(macro_arr, 2.5)),
            "ci_upper": float(np.percentile(macro_arr, 97.5)),
        },
        "bull_f1": {
            "point": float(point_per_class[1]),
            "ci_lower": float(np.percentile(bull_arr, 2.5)),
            "ci_upper": float(np.percentile(bull_arr, 97.5)),
        },
        "bear_f1": {
            "point": float(point_per_class[2]),
            "ci_lower": float(np.percentile(bear_arr, 2.5)),
            "ci_upper": float(np.percentile(bear_arr, 97.5)),
        },
        "n_bootstrap": n_iterations,
        "block_size": block_size,
        "effective_n": effective_n(n, block_size),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    scores = f1_score(y_true, y_pred, average=None, zero_division=0.0, labels=[0, 1, 2])
    # Ensure 3 elements even if some classes absent in bootstrap sample
    result = [0.0, 0.0, 0.0]
    for i, v in enumerate(scores):
        if i < 3:
            result[i] = float(v)
    return result
