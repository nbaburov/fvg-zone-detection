"""Tests for annotate.py — gold set sampling and kappa computation."""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from src.data.annotate import compute_kappa, sample_gold_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_labelled_h1(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic labelled H1 DataFrame with realistic label distribution.
    Produces ~10% positive labels (bull + bear).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 1.0, n).cumsum() + 350.0

    # Generate labels with ~10% positive rate
    raw_label = rng.choice([0, 1, -1], size=n, p=[0.88, 0.06, 0.06])

    # Create ATR-like volatility column for stratification
    price_range = rng.uniform(0.5, 5.0, n)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + price_range,
            "low": prices - price_range,
            "close": prices + rng.normal(0, 0.2, n),
            "volume": rng.integers(5000, 20000, n).astype(float),
            "session_type": pd.Categorical(["full"] * n, categories=["full", "half"]),
            "raw_label": raw_label,
            "label": pd.Series(raw_label).map({0: 0, 1: 1, -1: 2}).to_numpy(),
        },
        index=idx,
    )
    return df


def _make_gold_csv(path: str, n: int = 10, kappa_target: float = 0.75) -> None:
    """
    Write a synthetic gold_labels.csv with known approximate kappa.
    kappa_target=0.75 → produce 25% disagreement.
    """
    rows = []
    for i in range(n):
        prog = i % 3 - 1  # cycle: -1, 0, 1
        # Agree 75% of time
        if i < int(n * (1 - (1 - kappa_target) / 2)):
            human = prog
        else:
            human = 0  # disagree: use 0 when prog != 0
        rows.append(
            {
                "candle_index": i,
                "datetime": f"2020-01-0{(i % 9) + 1} 09:30:00",
                "programmatic_label": prog,
                "human_label": human,
                "annotator_note": "",
            }
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candle_index", "datetime", "programmatic_label", "human_label", "annotator_note"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Tests: sample_gold_set
# ---------------------------------------------------------------------------


def test_sample_gold_set_returns_75_rows():
    df = _make_labelled_h1(1000)
    gold = sample_gold_set(df, n_fvg_rich=25, n_low_vol=25, n_random=25, seed=42)
    assert len(gold) == 75, f"Expected 75 rows, got {len(gold)}"


def test_sample_gold_set_no_duplicates():
    df = _make_labelled_h1(1000)
    gold = sample_gold_set(df, n_fvg_rich=25, n_low_vol=25, n_random=25, seed=42)
    assert gold["candle_index"].nunique() == len(gold), "Duplicate candle_index in gold set"


def test_sample_gold_set_all_strata_present():
    df = _make_labelled_h1(1000)
    gold = sample_gold_set(df, n_fvg_rich=25, n_low_vol=25, n_random=25, seed=42)
    assert "stratum" in gold.columns, "Missing 'stratum' column in gold set"
    strata = set(gold["stratum"].unique())
    assert "fvg_rich" in strata, "Missing 'fvg_rich' stratum"
    assert "low_vol" in strata, "Missing 'low_vol' stratum"
    assert "random" in strata, "Missing 'random' stratum"


def test_sample_gold_set_has_required_columns():
    df = _make_labelled_h1(1000)
    gold = sample_gold_set(df, n_fvg_rich=25, n_low_vol=25, n_random=25, seed=42)
    required = {"candle_index", "datetime", "programmatic_label"}
    assert required.issubset(set(gold.columns)), f"Missing columns: {required - set(gold.columns)}"


# ---------------------------------------------------------------------------
# Tests: compute_kappa
# ---------------------------------------------------------------------------


def test_compute_kappa_matches_sklearn():
    """compute_kappa must match sklearn.metrics.cohen_kappa_score."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "gold_labels.csv")

        # Write known data
        prog_labels = [0, 1, -1, 0, 1, -1, 0, 0, 1, -1]
        human_labels = [0, 1, -1, 0, 1,  0, 0, 1, 1, -1]

        rows = []
        for i, (p, h) in enumerate(zip(prog_labels, human_labels)):
            rows.append(
                {
                    "candle_index": i,
                    "datetime": f"2020-01-02 09:30:00",
                    "programmatic_label": p,
                    "human_label": h,
                    "annotator_note": "",
                }
            )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["candle_index", "datetime", "programmatic_label", "human_label", "annotator_note"],
            )
            writer.writeheader()
            writer.writerows(rows)

        kappa = compute_kappa(csv_path)
        expected = cohen_kappa_score(prog_labels, human_labels)
        assert abs(kappa - expected) < 1e-6, f"kappa={kappa}, expected={expected}"


def test_compute_kappa_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        compute_kappa("/nonexistent/path/gold_labels.csv")
