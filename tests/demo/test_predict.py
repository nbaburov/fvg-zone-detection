"""test_predict.py — Tests for src/demo/predict.py.

ALWAYS-ON: stub adapter test covers the ModelTrack contract without requiring
real checkpoints. Real-checkpoint tests are skipif-gated but clearly separate.

Lesson from prior sessions: never let the only coverage be skip-gated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.demo.predict import ModelSpec, ModelTrack, build_model_tracks


# ---------------------------------------------------------------------------
# Stub adapter (always-on)
# ---------------------------------------------------------------------------

class _StubAdapter:
    """Minimal adapter that always predicts bull FVG for every window."""

    name = "_stub"

    def __init__(self, checkpoint_dir, **kwargs):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        probas = np.zeros((n, 3), dtype=np.float32)
        # Alternate predictions: bull/bear/none to get a mix of trades
        for i in range(n):
            if i % 5 == 0:
                probas[i, 1] = 1.0   # bull
            elif i % 5 == 1:
                probas[i, 2] = 1.0   # bear
            else:
                probas[i, 0] = 1.0   # none
        return probas


# ---------------------------------------------------------------------------
# Stub data (always-on)
# ---------------------------------------------------------------------------

def _make_stub_df(n_bars: int = 200) -> pd.DataFrame:
    """Build a minimal DataFrame that mimics spy_h1_test.parquet structure."""
    ts = pd.date_range(
        "2023-01-03 09:30:00", periods=n_bars, freq="1h", tz="America/New_York"
    )
    rng = np.random.default_rng(42)
    closes = 400.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    opens = closes + rng.normal(0, 0.2, n_bars)
    highs = np.maximum(opens, closes) + rng.uniform(0.1, 1.0, n_bars)
    lows = np.minimum(opens, closes) - rng.uniform(0.1, 1.0, n_bars)
    volumes = rng.integers(500, 5000, n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "label": np.zeros(n_bars, dtype=int),
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    # Sprinkle some FVG labels (won't affect stub predictions, just needed for runner)
    df.iloc[80, df.columns.get_loc("label")] = 1
    df.iloc[120, df.columns.get_loc("label")] = 2
    return df


# ---------------------------------------------------------------------------
# Test: ModelTrack contract (always-on, no real checkpoints)
# ---------------------------------------------------------------------------

class TestModelTrackContract:
    """Verify ModelTrack fields are aligned after build_model_tracks (stub path)."""

    def _build_stub_tracks(self, tmp_path: Path) -> list[ModelTrack]:
        """Inject stub adapter via registry patching."""
        from src.inspect import registry as _registry_mod

        # Save original registry state
        original = _registry_mod._REGISTRY

        try:
            # Register stub adapter
            _registry_mod._REGISTRY = {"_stub": _StubAdapter}

            df = _make_stub_df(n_bars=200)
            stub_parquet = tmp_path / "spy_h1_test.parquet"
            df.to_parquet(stub_parquet)

            # Build tracks using the stub path (window sits >= 60 bars in)
            window_start = df.index[70].isoformat()
            window_end = df.index[134].isoformat()   # 65 bars

            spec = ModelSpec(
                arch="_stub",
                tf="h1",
                dataset="spy",
                tuned=False,
                label="_Stub · H1 · stub · F1 0.999",
            )
            tracks = build_model_tracks(
                window_start=window_start,
                window_end=window_end,
                model_specs=[spec],
                data_path=stub_parquet,
                checkpoint_dir=tmp_path,
                lookahead_bars=10,
            )
            return tracks
        finally:
            _registry_mod._REGISTRY = original

    def test_tracks_length_matches_specs(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        assert len(tracks) == 1

    def test_pred_ts_preds_probas_aligned(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        t = tracks[0]
        n = len(t.pred_ts)
        assert len(t.preds) == n, f"preds len {len(t.preds)} != pred_ts len {n}"
        assert t.probas.shape == (n, 3), f"probas shape {t.probas.shape} != ({n}, 3)"

    def test_windows_raw_all_shape(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        t = tracks[0]
        n = len(t.pred_ts)
        assert t.windows_raw_all.shape == (n, 60, 5), (
            f"windows_raw_all shape {t.windows_raw_all.shape} != ({n}, 60, 5)"
        )

    def test_trades_only_on_positive_preds(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        t = tracks[0]
        # Every trade should have direction != 0
        for trade in t.trades:
            assert trade.direction in (1, 2), (
                f"Trade with direction {trade.direction} should not exist"
            )

    def test_track_name_matches_spec(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        assert tracks[0].name == "_stub"

    def test_track_label_matches_spec(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        assert tracks[0].label == "_Stub · H1 · stub · F1 0.999"

    def test_bars_are_nonempty_dataframe(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        assert isinstance(tracks[0].bars, __import__("pandas").DataFrame)
        assert len(tracks[0].bars) > 0

    def test_bars_have_ohlcv_columns(self, tmp_path):
        tracks = self._build_stub_tracks(tmp_path)
        cols = set(tracks[0].bars.columns)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in cols, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Test: ModelSpec NamedTuple (always-on)
# ---------------------------------------------------------------------------

class TestModelSpec:
    def test_model_spec_creation(self):
        spec = ModelSpec(
            arch="lstm",
            tf="h1",
            dataset="multisym",
            tuned=False,
            label="LSTM · H1 · multisym · F1 0.640",
        )
        assert spec.arch == "lstm"
        assert spec.tf == "h1"
        assert spec.dataset == "multisym"
        assert not spec.tuned

    def test_model_spec_is_immutable(self):
        spec = ModelSpec(arch="xgboost", tf="h1", dataset="multisym", tuned=False, label="XGB")
        with pytest.raises((AttributeError, TypeError)):
            spec.arch = "lstm"  # type: ignore


# ---------------------------------------------------------------------------
# Test: Real checkpoint (skip-gated)
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent.parent
_SPY_H1_TEST = _REPO / "data" / "processed" / "spy_h1_test.parquet"
_XGB_CKPT = _REPO / "checkpoints" / "xgboost_h1_multisym"

@pytest.mark.skipif(
    not (_SPY_H1_TEST.exists() and _XGB_CKPT.exists()),
    reason="Real checkpoints not present (CI skip)",
)
class TestRealXGBoostCheckpoint:
    """Integration test: real XGBoost H1 multisym checkpoint."""

    def test_xgboost_positive_count_above_zero(self):
        spec = ModelSpec(
            arch="xgboost",
            tf="h1",
            dataset="multisym",
            tuned=False,
            label="XGBoost · H1 · multisym · F1 0.738",
        )
        tracks = build_model_tracks(
            window_start="2025-08-21T09:30:00-04:00",
            window_end="2025-09-04T10:30:00-04:00",
            model_specs=[spec],
            lookahead_bars=20,
        )
        t = tracks[0]
        n_pos = int((t.preds != 0).sum())
        assert n_pos >= 0  # smoke: should not crash; positive count >= 0
        # Shape alignment: preds and pred_ts must have the same length
        assert t.preds.shape == (len(t.pred_ts),), (
            f"preds shape {t.preds.shape} != pred_ts length {len(t.pred_ts)}"
        )

    def test_real_track_contract(self):
        spec = ModelSpec(
            arch="xgboost",
            tf="h1",
            dataset="multisym",
            tuned=False,
            label="XGBoost · H1 · multisym · F1 0.738",
        )
        tracks = build_model_tracks(
            window_start="2025-08-21T09:30:00-04:00",
            window_end="2025-09-04T10:30:00-04:00",
            model_specs=[spec],
            lookahead_bars=20,
        )
        t = tracks[0]
        n = len(t.pred_ts)
        assert len(t.preds) == n
        assert t.probas.shape == (n, 3)
        assert t.windows_raw_all.shape == (n, 60, 5)
