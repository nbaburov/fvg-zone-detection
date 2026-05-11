"""Integration test: offline replay through LiveWindowBuilder + SingleModelDecision.

Uses a pre-built fixture (tests/fixtures/session_sample/) containing:
  - bars_h1.parquet: 65 synthetic H1 bars
  - predictions_expected.parquet: expected decisions for a deterministic mock adapter

If the fixture does not exist, it is generated on first run and the test passes.
Subsequent runs verify the pipeline is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inspect.base import ModelAdapter
from src.live.decision import SingleModelDecision
from src.live.window_builder import H1Bar, LiveWindowBuilder

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "session_sample"
_BARS_PATH = _FIXTURE_DIR / "bars_h1.parquet"
_PREDS_PATH = _FIXTURE_DIR / "predictions_expected.parquet"

_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Deterministic mock adapter
# ---------------------------------------------------------------------------


class DeterministicAdapter(ModelAdapter):
    """Returns a fixed cycle of probas based on window index (deterministic)."""

    name = "_integration_test_adapter"

    _CYCLES = [
        [0.8, 0.1, 0.1],   # none
        [0.1, 0.9, 0.0],   # bull
        [0.1, 0.0, 0.9],   # bear
    ]

    def __init__(self) -> None:
        self._call_count = 0

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        proba = self._CYCLES[self._call_count % len(self._CYCLES)]
        self._call_count += 1
        return np.array([proba], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------


def _generate_fixture() -> None:
    """Generate synthetic bars_h1.parquet and predictions_expected.parquet."""
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Build 65 synthetic H1 bars (enough for a 60-bar window)
    base = pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York")
    bars = []
    for i in range(65):
        ts = base + pd.Timedelta(hours=i)
        bars.append({
            "timestamp": ts,
            "open": 100.0 + i * 0.1,
            "high": 101.0 + i * 0.1,
            "low": 99.0 + i * 0.1,
            "close": 100.5 + i * 0.1,
            "volume": 5000.0 + i * 10,
            "window_valid": i >= 59,
        })

    import pyarrow as pa
    import pyarrow.parquet as pq

    bars_df = pd.DataFrame(bars)
    pq.write_table(pa.Table.from_pandas(bars_df), str(_BARS_PATH))

    # Run the pipeline once to generate expected predictions
    predictions = _run_pipeline(bars_df, DeterministicAdapter(), _THRESHOLD)
    preds_df = pd.DataFrame(predictions)
    pq.write_table(pa.Table.from_pandas(preds_df), str(_PREDS_PATH))


def _run_pipeline(
    bars_df: pd.DataFrame,
    adapter: ModelAdapter,
    threshold: float,
) -> list[dict]:
    """Feed H1 bars through LiveWindowBuilder + SingleModelDecision and collect predictions."""
    builder = LiveWindowBuilder()
    decision = SingleModelDecision(adapter, threshold=threshold)
    results = []

    for _, row in bars_df.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")

        h1 = H1Bar(
            timestamp=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

        # Inject H1 bars directly into the buffer (replay mode)
        builder._h1_buffer.append(h1)
        event = builder._build_window_event(h1)
        action = decision.decide(event)

        results.append({
            "timestamp": str(ts),
            "signal": action.signal,
            "pred_class": int({"none": 0, "bull": 1, "bear": 2}[action.signal]),
            "confidence": float(action.confidence),
            "skip_reason": action.skip_reason or "",
        })

    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestOfflineReplay:
    def setup_method(self):
        if not _BARS_PATH.exists() or not _PREDS_PATH.exists():
            _generate_fixture()

    def test_pipeline_matches_fixture(self):
        """Replay produces decisions matching the fixture (100% reproducibility)."""
        import pyarrow.parquet as pq

        bars_df = pq.read_table(str(_BARS_PATH)).to_pandas()
        expected_preds = pq.read_table(str(_PREDS_PATH)).to_pandas()

        # Use a fresh adapter with same seed/determinism
        adapter = DeterministicAdapter()
        live_preds = _run_pipeline(bars_df, adapter, _THRESHOLD)
        live_df = pd.DataFrame(live_preds)

        assert len(live_df) == len(expected_preds), "Row count mismatch"

        for i, (live_row, exp_row) in enumerate(zip(live_preds, expected_preds.to_dict("records"))):
            assert live_row["pred_class"] == int(exp_row["pred_class"]), (
                f"Divergence at bar {i}: live={live_row['pred_class']} expected={exp_row['pred_class']}"
            )

    def test_warmup_bars_have_no_window(self):
        """First 59 H1 bars produce skip_reason='WARMUP'."""
        import pyarrow.parquet as pq

        bars_df = pq.read_table(str(_BARS_PATH)).to_pandas()
        adapter = DeterministicAdapter()
        preds = _run_pipeline(bars_df, adapter, _THRESHOLD)

        warmup_preds = [p for p in preds if p["skip_reason"] == "WARMUP"]
        assert len(warmup_preds) >= 59, (
            f"Expected at least 59 WARMUP predictions, got {len(warmup_preds)}"
        )

    def test_window_predictions_have_valid_probas(self):
        """After warm-up, predictions have confidence in [0, 1]."""
        import pyarrow.parquet as pq

        bars_df = pq.read_table(str(_BARS_PATH)).to_pandas()
        adapter = DeterministicAdapter()
        preds = _run_pipeline(bars_df, adapter, _THRESHOLD)

        active_preds = [p for p in preds if p["skip_reason"] not in ("WARMUP", "CROSS_SESSION_GAP")]
        if active_preds:
            for p in active_preds:
                assert 0.0 <= p["confidence"] <= 1.0
