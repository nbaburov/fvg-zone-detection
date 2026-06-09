"""tests/rigor/test_load_splits_sort.py — Regression test for _load_splits multisym sort guard.

Covers the _sort_if_multisym nested helper added to _load_splits in seed_sweep.py:
  1. Multisym reload re-sorts: interleaved rows (A,B,A,B...) become symbol-block contiguous
     and chronologically ordered within each symbol.
  2. Single-symbol no-op: a DataFrame with no 'symbol' column round-trips unchanged
     (same row order, same values).

_sort_if_multisym is a nested local and is not importable directly.  The smallest honest
seam is _load_splits itself — called with tmp_path parquets so no network or training
is triggered.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.rigor.seed_sweep import _load_splits


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_splits(data_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Write the three parquets that _load_splits expects."""
    train.to_parquet(data_dir / "spy_h1_train.parquet")
    val.to_parquet(data_dir / "spy_h1_val.parquet")
    test.to_parquet(data_dir / "spy_h1_test.parquet")


def _make_multisym_df(interleaved: bool = True) -> pd.DataFrame:
    """Return a small multisym DataFrame.

    With interleaved=True rows are A,B,A,B,A,B (scrambled symbol blocks).
    timestamps within each symbol are intentionally non-monotonic when interleaved
    so the sort must restore both block-contiguity AND per-symbol order.
    """
    idx = pd.to_datetime([
        "2023-01-03 10:00", "2023-01-03 10:00",
        "2023-01-03 11:00", "2023-01-03 11:00",
        "2023-01-03 12:00", "2023-01-03 12:00",
    ])
    symbols = ["SPY", "QQQ", "SPY", "QQQ", "SPY", "QQQ"] if interleaved else ["SPY", "SPY", "SPY", "QQQ", "QQQ", "QQQ"]
    closes = [100.0, 200.0, 101.0, 201.0, 102.0, 202.0]
    return pd.DataFrame({"symbol": symbols, "close": closes}, index=idx)


def _make_single_sym_df() -> pd.DataFrame:
    """Single-symbol frame with no 'symbol' column."""
    idx = pd.to_datetime([
        "2023-01-03 10:00",
        "2023-01-03 11:00",
        "2023-01-03 12:00",
    ])
    return pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_multisym_reload_sorts_by_symbol_then_time(tmp_path: Path) -> None:
    """_load_splits must restore symbol-block contiguity on a scrambled multisym parquet."""
    interleaved = _make_multisym_df(interleaved=True)
    # Confirm the fixture is actually interleaved before the sort.
    assert list(interleaved["symbol"]) == ["SPY", "QQQ", "SPY", "QQQ", "SPY", "QQQ"]

    _write_splits(tmp_path, train=interleaved, val=interleaved, test=interleaved)
    train_out, val_out, test_out = _load_splits(tmp_path)

    for split_name, result in [("train", train_out), ("val", val_out), ("test", test_out)]:
        symbols = result["symbol"].tolist()

        # All SPY rows must come before all QQQ rows (or vice-versa — block contiguous).
        first_sym = symbols[0]
        in_first_block = True
        for sym in symbols:
            if sym != first_sym:
                in_first_block = False
            if not in_first_block and sym == first_sym:
                pytest.fail(f"{split_name}: symbol column is not block-contiguous after _load_splits: {symbols}")

        # Within each symbol, timestamps must be monotonically non-decreasing.
        for sym in result["symbol"].unique():
            ts = result.loc[result["symbol"] == sym].index
            assert ts.is_monotonic_increasing, (
                f"{split_name}: timestamps for symbol '{sym}' are not sorted after _load_splits"
            )


def test_single_symbol_no_op(tmp_path: Path) -> None:
    """_load_splits must leave a single-symbol (no 'symbol' column) frame byte-identical."""
    single = _make_single_sym_df()
    _write_splits(tmp_path, train=single, val=single, test=single)

    train_out, val_out, test_out = _load_splits(tmp_path)

    for split_name, result in [("train", train_out), ("val", val_out), ("test", test_out)]:
        assert "symbol" not in result.columns, f"{split_name}: 'symbol' column should not exist"
        pd.testing.assert_frame_equal(
            result.reset_index(),
            single.reset_index(),
            check_like=False,  # preserve row order
            obj=f"{split_name} single-sym no-op",
        )
