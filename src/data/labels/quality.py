"""Label-quality gates for the SMC pipeline.

A degenerate FVG label distribution silently breaks training: a per-class
positive rate that is too LOW (< 0.3%) is unlearnable noise, and one that is
too HIGH (> 15%) means the labelling lookbacks are too loose — the labels no
longer mark rare smart-money events. This module exposes the post-label gate
that the data pipeline enforces.

The functions operate on the **raw** ternary FVG label series
(``-1`` = bear, ``0`` = none, ``1`` = bull), which is the form produced by
``ValidFVGLabeller.label`` before encoding to ``{0, 1, 2}``.

Band (per class): [0.3%, 15%]. The H1 ValidFVG baseline is ≈1.5%/class.
"""

from __future__ import annotations

import pandas as pd

# Acceptable per-class positive-rate band (fractions in [0, 1]).
POSITIVE_RATE_MIN = 0.003  # 0.3% — below this the class is degenerate/unlearnable
POSITIVE_RATE_MAX = 0.15   # 15%  — above this the lookbacks are too loose


def compute_positive_rate(labels: "pd.Series") -> dict[str, float]:
    """Compute per-class and total positive FVG rate from raw ternary labels.

    Parameters
    ----------
    labels : pd.Series
        Raw FVG labels with values in ``{-1, 0, 1}``.

    Returns
    -------
    dict[str, float]
        Keys ``"bull"``, ``"bear"``, ``"total"`` — fractions in ``[0, 1]``.
    """
    n = len(labels)
    if n == 0:
        return {"bull": 0.0, "bear": 0.0, "total": 0.0}
    n_bull = int((labels == 1).sum())
    n_bear = int((labels == -1).sum())
    return {
        "bull": n_bull / n,
        "bear": n_bear / n,
        "total": (n_bull + n_bear) / n,
    }


def check_positive_rate_gate(
    rates: dict[str, float],
    min_rate: float = POSITIVE_RATE_MIN,
    max_rate: float = POSITIVE_RATE_MAX,
) -> tuple[bool, str]:
    """Evaluate whether per-class positive rates are within the acceptable band.

    Parameters
    ----------
    rates : dict[str, float]
        Output of :func:`compute_positive_rate` (needs ``"bull"`` / ``"bear"``).
    min_rate, max_rate : float
        Inclusive band bounds (fractions in ``[0, 1]``).

    Returns
    -------
    tuple[bool, str]
        ``(gate_passed, message)``. The gate passes only when both bull and
        bear rates lie within ``[min_rate, max_rate]``. A class that never
        appears (rate ``0.0``) fails the lower bound.
    """
    issues: list[str] = []
    for cls in ("bull", "bear"):
        r = rates[cls]
        if r < min_rate:
            issues.append(
                f"{cls} rate {r:.4%} < min {min_rate:.4%} (degenerate / unlearnable)"
            )
        elif r > max_rate:
            issues.append(
                f"{cls} rate {r:.4%} > max {max_rate:.4%} (lookbacks too loose)"
            )
    if issues:
        return False, "; ".join(issues)
    return True, "gate passed"


def enforce_positive_rate_gate(
    labels: "pd.Series",
    *,
    context: str = "labels",
    min_rate: float = POSITIVE_RATE_MIN,
    max_rate: float = POSITIVE_RATE_MAX,
) -> dict[str, float]:
    """Compute the positive rate and raise if it is out of band.

    Parameters
    ----------
    labels : pd.Series
        Raw FVG labels with values in ``{-1, 0, 1}``.
    context : str
        Label appended to the error message (e.g. the timeframe token) so a
        failure clearly identifies which dataset tripped the gate.
    min_rate, max_rate : float
        Band bounds forwarded to :func:`check_positive_rate_gate`.

    Returns
    -------
    dict[str, float]
        The computed rates (so callers can log them).

    Raises
    ------
    ValueError
        When either per-class positive rate is outside ``[min_rate, max_rate]``.
    """
    rates = compute_positive_rate(labels)
    passed, msg = check_positive_rate_gate(rates, min_rate=min_rate, max_rate=max_rate)
    if not passed:
        raise ValueError(
            f"Positive-rate gate FAILED for {context}: {msg}. "
            f"(bull={rates['bull']:.4%}, bear={rates['bear']:.4%}, "
            f"total={rates['total']:.4%}). "
            "Revisit the labelling lookbacks before training, or pass "
            "enforce_positive_rate_gate=False to bypass (synthetic/unit data)."
        )
    return rates
