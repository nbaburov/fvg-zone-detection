"""test_sim_enhancements.py — Tests for three FVG sim enhancement axes.

  1. Confidence filter correctness (ExitConfig.confidence_threshold)
  2. Confidence definition: max(P_bull, P_bear) — NOT P_predicted or P_none
  3. fill_mode conservative vs optimistic
  4. All-seeds aggregation math (_aggregate_seed_summaries)
  5. Backward-compat: defaults yield identical results to pre-enhancement path
  6. Validation: out-of-range confidence_threshold / invalid fill_mode raises

All expected values are hand-computed from synthetic geometry.
No mocking of code under test.  Deterministic, CPU, no network.

Candle indexing (from exits.py):
  bar_56 = window[56]  candle-1 (pre-impulse)
  bar_57 = window[57]  candle-2 (impulse)
  bar_58 = window[58]  candle-3 (reaction / gap near-edge)
  Column order: [open=0, high=1, low=2, close=3, volume=4]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable when pytest is run from any cwd
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.strategy.exits import TICK, ExitConfig, compute_exit
from src.inspect.outcomes import (
    Trade,
    TradeOutcome,
    compute_trades_for_model,
    summarise_trades,
)


# ---------------------------------------------------------------------------
# Shared geometry helpers (mirrors test_fvg_exits.py pattern; no shared import
# to keep test files self-contained — the helpers are short)
# ---------------------------------------------------------------------------

def _make_window(
    bar56_high: float,
    bar56_low: float,
    bar57_high: float,
    bar57_low: float,
    bar58_high: float,
    bar58_low: float,
    base_price: float = 400.0,
    base_tr: float = 1.0,
) -> np.ndarray:
    """Build a (60, 5) un-normalised OHLCV window.

    Bars 0–55: uniform candles with TR = base_tr so ATR is predictable.
    Bars 56–58: controlled candles for geometry tests.
    """
    window = np.zeros((60, 5), dtype=np.float64)
    half = base_tr / 2.0
    for i in range(56):
        window[i] = [base_price, base_price + half, base_price - half, base_price, 1000.0]
    window[56] = [bar56_low, bar56_high, bar56_low, bar56_high, 1000.0]
    window[57] = [bar57_low, bar57_high, bar57_low, bar57_high, 1000.0]
    window[58] = [bar58_low, bar58_high, bar58_low, bar58_high, 1000.0]
    return window


def _make_future(*bars: tuple[float, float, float, float]) -> np.ndarray:
    """Build (N, 5) future OHLCV array from (open, high, low, close) tuples."""
    arr = np.zeros((len(bars), 5), dtype=np.float64)
    for i, (o, h, lo, c) in enumerate(bars):
        arr[i] = [o, h, lo, c, 1000.0]
    return arr


def _make_trade_outcome(
    outcome: str,
    direction: int = 1,
    entry: float = 400.0,
    sl: float = 399.0,
    r_multiple: float = 0.0,
    filled: bool = True,
) -> TradeOutcome:
    """Minimal TradeOutcome for summarise_trades tests."""
    return TradeOutcome(
        window_idx=0,
        entry_ts=pd.NaT,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=402.0,
        outcome=outcome,
        exit_idx_in_future=-1,
        exit_ts=None,
        exit_price=entry,
        r_multiple=r_multiple,
        filled=filled,
        swing_fallback=False,
    )


# ---------------------------------------------------------------------------
# Batch builder for compute_trades_for_model tests
# ---------------------------------------------------------------------------

def _build_conf_batch():
    """Five windows; preds mixed; probas vary in conviction.

    Window geometry: bull FVG, market entry (fixed_2r).
      bar_56.high = 400, future[0].open = 403
      SL = 400 - 0.01 = 399.99, risk = 3.01, TP = 403 + 2*3.01 = 409.02

    Predictions and probas:
      idx 0: pred=1 (bull), confidence = max(P1=0.85, P2=0.05) = 0.85  → HIGH
      idx 1: pred=1 (bull), confidence = max(P1=0.45, P2=0.10) = 0.45  → LOW
      idx 2: pred=2 (bear), confidence = max(P1=0.08, P2=0.72) = 0.72  → HIGH
      idx 3: pred=2 (bear), confidence = max(P1=0.05, P2=0.35) = 0.35  → LOW
      idx 4: pred=0 (none), confidence doesn't matter                   → IGNORED
    """
    # Single reusable window — bull geometry
    w = _make_window(
        bar56_high=400.0, bar56_low=395.0,
        bar57_high=408.0, bar57_low=396.0,
        bar58_high=403.0, bar58_low=401.0,
        base_price=400.0, base_tr=1.0,
    )
    # future: TP is hit at bar1 (high = 412 > TP = 409.02)
    entry = 403.0
    sl = 400.0 - TICK   # 399.99
    risk = entry - sl   # 3.01
    tp = entry + 2.0 * risk   # 409.02
    fut_tp = _make_future(
        (entry, 406, 402, 404),          # bar0: no hit
        (405, tp + 2.0, 404, 405),       # bar1: TP hit
        (405, 407, 404, 406),            # bar2: filler
    )
    # future for bear (SL immediately)
    w_bear = _make_window(
        bar56_high=406.0, bar56_low=400.0,
        bar57_high=403.0, bar57_low=392.0,
        bar58_high=404.0, bar58_low=399.0,
        base_price=400.0, base_tr=1.0,
    )
    sl_bear = 400.0 + TICK  # 400.01
    fut_sl_bear = _make_future(
        (398.0, sl_bear + 0.5, 397.0, 398.0),  # SL hit bar0
        (399.0, 400.0, 398.0, 399.0),
        (399.0, 400.0, 398.0, 399.0),
    )
    # None pred window
    w_none = _make_window(400, 395, 408, 396, 403, 401, base_price=400.0)
    fut_none = _make_future(
        (403, 406, 402, 404),
        (405, 410, 404, 405),
        (405, 407, 404, 406),
    )

    windows_raw = np.stack([w, w, w_bear, w_bear, w_none])
    future_ohlcv = np.stack([fut_tp, fut_tp, fut_sl_bear, fut_sl_bear, fut_none])
    preds = np.array([1, 1, 2, 2, 0], dtype=np.int64)
    probas = np.array([
        [0.10, 0.85, 0.05],   # idx0: bull high confidence 0.85
        [0.45, 0.45, 0.10],   # idx1: bull low confidence 0.45
        [0.20, 0.08, 0.72],   # idx2: bear high confidence 0.72
        [0.60, 0.05, 0.35],   # idx3: bear low confidence 0.35
        [0.90, 0.05, 0.05],   # idx4: none (irrelevant)
    ], dtype=np.float32)
    future_timestamps = [[], [], [], [], []]
    return windows_raw, future_ohlcv, future_timestamps, preds, probas


# ===========================================================================
# 1 — Confidence filter correctness
# ===========================================================================

class TestConfidenceFilterCorrectness:
    """REVERT-SENSITIVE: confidence filter must gate on max(P_bull, P_bear) >= threshold.

    If the filter is removed (confidence_threshold ignored), all 4 positive
    preds become trades regardless of confidence.  The low-confidence ones
    (idx 1 and 3) must be excluded when threshold > their max-directional proba.
    """

    def test_all_positives_trade_without_probas(self):
        """probas=None (default): all 4 positive predictions become trades."""
        ws, futs, ts, preds, _ = _build_conf_batch()
        cfg = ExitConfig("fixed_2r")
        trades = compute_trades_for_model(ws, futs, ts, preds, exit_config=cfg)
        assert len(trades) == 4

    def test_all_positives_trade_with_threshold_zero(self):
        """threshold=0.0 (default off): even low-confidence preds trade — same as no-probas path.

        REVERT-SENSITIVE: if default threshold were accidentally set to >0, this
        test would fail because the low-confidence predictions would be excluded.
        """
        ws, futs, ts, preds, probas = _build_conf_batch()
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.0)
        trades = compute_trades_for_model(ws, futs, ts, preds, probas=probas, exit_config=cfg)
        assert len(trades) == 4

    def test_threshold_50_excludes_low_confidence(self):
        """threshold=0.5: idx1 (conf=0.45) and idx3 (conf=0.35) are excluded.

        REVERT-SENSITIVE: a revert removing the filter would re-include the
        low-confidence predictions, producing 4 trades instead of 2.
        """
        ws, futs, ts, preds, probas = _build_conf_batch()
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.5)
        trades = compute_trades_for_model(ws, futs, ts, preds, probas=probas, exit_config=cfg)
        # Only idx0 (conf=0.85) and idx2 (conf=0.72) pass threshold=0.5
        assert len(trades) == 2
        assert trades[0].direction == 1   # bull (idx0)
        assert trades[1].direction == 2   # bear (idx2)

    def test_threshold_just_under_excludes_nothing(self):
        """threshold=0.44 — just below idx1's confidence=0.45 → idx1 passes."""
        ws, futs, ts, preds, probas = _build_conf_batch()
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.44)
        trades = compute_trades_for_model(ws, futs, ts, preds, probas=probas, exit_config=cfg)
        # idx0 (0.85), idx1 (0.45), idx2 (0.72) pass; idx3 (0.35) excluded
        assert len(trades) == 3

    def test_threshold_just_over_excludes_border_case(self):
        """threshold=0.46 — just above idx1's confidence=0.45 → idx1 excluded.

        REVERT-SENSITIVE: the boundary check must be strictly less-than: confidence
        >= threshold passes.  If the comparison is inverted or off-by-one, a window
        with confidence=0.45 would wrongly pass a threshold=0.46 filter.
        """
        ws, futs, ts, preds, probas = _build_conf_batch()
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.46)
        trades = compute_trades_for_model(ws, futs, ts, preds, probas=probas, exit_config=cfg)
        # idx0 (0.85) and idx2 (0.72) pass; idx1 (0.45) and idx3 (0.35) excluded
        assert len(trades) == 2

    def test_n_signals_vs_n_eligible_are_distinct(self):
        """Verify n_signals (total positive preds) differs from n_trades (post-filter).

        n_signals is computed by the caller as int((preds != 0).sum()).
        n_trades (summarise_trades output) counts only eligible trades.
        """
        ws, futs, ts, preds, probas = _build_conf_batch()
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.5)
        trades = compute_trades_for_model(ws, futs, ts, preds, probas=probas, exit_config=cfg)
        n_signals = int((preds != 0).sum())  # caller-side count = 4
        n_eligible = len(trades)             # post-filter = 2
        assert n_signals == 4
        assert n_eligible == 2
        # summarise_trades n_signals reflects the trades list length (n_eligible)
        summary = summarise_trades(trades, strategy_name="fixed_2r")
        assert summary["n_signals"] == n_eligible  # 2, not 4


# ===========================================================================
# 2 — Confidence definition: max(P_bull, P_bear) NOT P_predicted or P_none
# ===========================================================================

class TestConfidenceDefinition:
    """Confidence must equal max(P_bull=probas[:,1], P_bear=probas[:,2]).

    Construct a window where the three candidate definitions differ:
      - P_predicted_class = probas[i, preds[i]]  (prob of the predicted class)
      - P_none = probas[i, 0]
      - max(P_bull, P_bear)

    For a bull prediction (pred=1):
      probas = [0.70, 0.20, 0.10]
      P_none = 0.70, P_predicted (P_bull) = 0.20, max(P_bull,P_bear) = 0.20

    Wait — for any argmax-positive pred, P_predicted == P_predicted_class,
    and max(P_bull,P_bear) == P_predicted_class (since argmax selected it).

    For a BEAR prediction (pred=2) where P_bull > P_bear:
      This cannot happen since argmax selected P_bear.

    The plan (D2) states: "For argmax=bull: confidence = probas[i,1] (which
    equals max(P_bull,P_bear) since argmax=bull)."  So for any argmax-positive
    pred, P_predicted_class == max(P_bull,P_bear).

    The distinction is: confidence must NOT use P_none.  We test a window
    where P_none is large but max(P_bull,P_bear) is small — the window must
    be excluded by a threshold test, proving P_none did not gate it.
    """

    def _single_batch(self, pred: int, probas_row: list) -> tuple:
        """One window, one prediction."""
        w = _make_window(400, 395, 408, 396, 403, 401, base_price=400.0)
        entry = 403.0
        sl = 400.0 - TICK
        risk = entry - sl
        tp = entry + 2.0 * risk
        fut = _make_future(
            (entry, 406, 402, 404),
            (405, tp + 2.0, 404, 405),
        )
        windows_raw = np.array([w])
        future_ohlcv = np.array([fut])
        preds = np.array([pred], dtype=np.int64)
        probas = np.array([probas_row], dtype=np.float32)
        return windows_raw, future_ohlcv, [[]], preds, probas

    def test_high_p_none_does_not_gate_high_directional_confidence(self):
        """P_none=0.80 but max(P_bull,P_bear)=0.15; threshold=0.10 — trade passes.

        If confidence were incorrectly computed as 1 - P_predicted_class or
        P_none, the threshold logic would be inverted or wrong.  The test
        confirms the window DOES trade (max(P_bull,P_bear)=0.15 >= threshold=0.10).
        """
        # pred=1 (bull), P_none=0.80, P_bull=0.15, P_bear=0.05
        # max(P_bull, P_bear) = 0.15 >= 0.10 → should trade
        ws, futs, ts, preds, probas = self._single_batch(1, [0.80, 0.15, 0.05])
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.10)
        trades = compute_trades_for_model(ws, futs, ts, preds, probas=probas, exit_config=cfg)
        assert len(trades) == 1, "Window with max(P_bull,P_bear)=0.15 >= threshold=0.10 must trade"

    def test_high_p_none_with_low_directional_excluded(self):
        """P_none=0.80, max(P_bull,P_bear)=0.15; threshold=0.20 — window excluded.

        Proves confidence = max(P_bull, P_bear), not max(P_none, P_predicted) or
        any other formulation.  If confidence were wrongly set to P_none (0.80),
        the window would NOT be excluded by threshold=0.20.
        """
        ws, futs, ts, preds, probas = self._single_batch(1, [0.80, 0.15, 0.05])
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.20)
        trades = compute_trades_for_model(ws, futs, ts, preds, probas=probas, exit_config=cfg)
        assert len(trades) == 0, "max(P_bull,P_bear)=0.15 < threshold=0.20 must be excluded"

    def test_bear_pred_uses_p_bear_not_p_bull(self):
        """For pred=2 (bear), confidence = P_bear.  P_bull < P_bear (by argmax).

        Construct: P_bear=0.60, P_bull=0.30 → max = 0.60.
        threshold=0.55 → should trade.
        """
        # pred=2 (bear), P_none=0.10, P_bull=0.30, P_bear=0.60
        w_bear = _make_window(406, 400, 403, 392, 404, 399, base_price=400.0)
        sl_bear = 400.0 + TICK
        fut_bear = _make_future(
            (398.0, sl_bear + 0.5, 397.0, 398.0),
            (399.0, 400.0, 398.0, 399.0),
        )
        ws = np.array([w_bear])
        futs = np.array([fut_bear])
        preds = np.array([2], dtype=np.int64)
        probas = np.array([[0.10, 0.30, 0.60]], dtype=np.float32)
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.55)
        trades = compute_trades_for_model(ws, futs, [[]], preds, probas=probas, exit_config=cfg)
        # P_bear=0.60 >= 0.55 → trade
        assert len(trades) == 1


# ===========================================================================
# 3 — fill_mode conservative
# ===========================================================================

class TestFillModeConservative:
    """REVERT-SENSITIVE: conservative fill must check bar.close <= limit (bull).

    Geometry uses ict_iofed (V2) — limit-entry at bar_58.low.
    A bar whose low <= limit (would fill optimistic) but close > limit
    must produce no_fill under conservative.
    A bar whose close <= limit must fill under both modes.
    Optimistic default must be byte-identical to pre-fill_mode behavior.
    """

    # Bull FVG: bar_56.high=400, bar_58.low=396 → limit = 396 (ict_iofed near-edge)
    WINDOW = _make_window(
        bar56_high=400.0, bar56_low=394.0,
        bar57_high=408.0, bar57_low=395.0,
        bar58_high=402.0, bar58_low=396.0,
        base_price=400.0, base_tr=1.0,
    )
    LIMIT = 396.0   # bar_58.low

    def test_optimistic_fills_when_low_touches_limit(self):
        """Optimistic: bar.low <= limit → fill.

        Bar: open=397, high=398, low=395.5, close=397.2
        low (395.5) <= limit (396) → optimistic fills.

        REVERT-SENSITIVE: if fill_mode default were changed to 'conservative',
        this bar (close=397.2 > limit=396) would NOT fill.
        """
        future = _make_future(
            (397.0, 398.0, 395.5, 397.2),   # low<=limit, close>limit
            (398.0, 402.0, 397.0, 400.0),   # TP zone
            (400.0, 410.0, 399.0, 405.0),   # TP hit
        )
        cfg = ExitConfig("ict_iofed", fill_mode="optimistic")
        result = compute_exit(self.WINDOW, future, 1, cfg)
        assert result.filled is True, "Optimistic must fill when bar.low <= limit"
        assert result.outcome != "no_fill"

    def test_conservative_no_fill_when_low_touches_but_close_above(self):
        """Conservative: bar.low <= limit but close > limit → no_fill.

        Same bar as above; only fill_mode differs.
        REVERT-SENSITIVE: if the conservative condition were incorrectly
        implemented as low-based (optimistic), this would fill instead of no_fill.
        """
        future = _make_future(
            (397.0, 398.0, 395.5, 397.2),   # low<=limit, close>limit
            (398.0, 402.0, 397.0, 400.0),
            (400.0, 410.0, 399.0, 405.0),
        )
        cfg = ExitConfig("ict_iofed", fill_timeout_bars=1, fill_mode="conservative")
        result = compute_exit(self.WINDOW, future, 1, cfg)
        assert result.outcome == "no_fill", (
            "Conservative must NOT fill when bar.close (397.2) > limit (396.0)"
        )

    def test_conservative_fills_when_close_at_or_below_limit(self):
        """Conservative: bar.close <= limit → fill.

        Bar: open=397, high=398, low=395.5, close=395.8  (close <= limit=396)
        """
        future = _make_future(
            (397.0, 398.0, 395.5, 395.8),   # low<=limit AND close<=limit
            (396.0, 402.0, 395.0, 400.0),   # TP zone
            (400.0, 410.0, 399.0, 405.0),   # TP hit
        )
        cfg = ExitConfig("ict_iofed", fill_mode="conservative")
        result = compute_exit(self.WINDOW, future, 1, cfg)
        assert result.filled is True, "Conservative must fill when bar.close <= limit"
        assert result.outcome != "no_fill"

    def test_optimistic_default_unchanged_from_pre_enhancement(self):
        """ExitConfig() default fill_mode='optimistic' is byte-identical to pre-enhancement.

        REVERT-SENSITIVE: this test would break if the default were changed to
        'conservative', because conservative produces fewer fills.
        """
        future = _make_future(
            (397.0, 398.0, 395.5, 397.2),   # low<=limit, close>limit
            (398.0, 402.0, 397.0, 400.0),
            (400.0, 410.0, 399.0, 405.0),
        )
        cfg_default = ExitConfig("ict_iofed")
        cfg_explicit = ExitConfig("ict_iofed", fill_mode="optimistic")
        r1 = compute_exit(self.WINDOW, future, 1, cfg_default)
        r2 = compute_exit(self.WINDOW, future, 1, cfg_explicit)
        assert r1.outcome == r2.outcome
        assert r1.entry == pytest.approx(r2.entry)
        assert r1.r_multiple == pytest.approx(r2.r_multiple)

    def test_bear_conservative_no_fill_when_high_touches_but_close_below(self):
        """Bear conservative: close >= limit (bear limit = bar_58.high = 404).

        Bar: close=403.5 < bear_limit=404 → conservative no_fill.
        Bar: high=404.5 >= 404 → optimistic fill.
        """
        window_bear = _make_window(
            bar56_high=400.0, bar56_low=394.0,
            bar57_high=392.0, bar57_low=388.0,
            bar58_high=404.0, bar58_low=399.0,
            base_price=400.0, base_tr=1.0,
        )
        bear_limit = 404.0  # bar_58.high for bear ict_iofed
        future = _make_future(
            (405.0, 404.5, 403.0, 403.5),  # high>=limit, close<limit (403.5 < 404)
            (404.0, 405.0, 403.0, 404.5),
        )
        cfg_cons = ExitConfig("ict_iofed", fill_timeout_bars=1, fill_mode="conservative")
        result = compute_exit(window_bear, future, 2, cfg_cons)
        assert result.outcome == "no_fill", (
            "Bear conservative must NOT fill when close (403.5) < limit (404.0)"
        )

    def test_bear_conservative_fills_when_close_at_or_above_limit(self):
        """Bear conservative: bar.close >= limit → fill."""
        window_bear = _make_window(
            bar56_high=400.0, bar56_low=394.0,
            bar57_high=392.0, bar57_low=388.0,
            bar58_high=404.0, bar58_low=399.0,
            base_price=400.0, base_tr=1.0,
        )
        future = _make_future(
            (405.0, 405.5, 403.0, 404.2),  # high>=limit AND close>=limit(=404)
            (404.0, 405.0, 403.0, 404.5),
            (404.0, 395.0, 390.0, 392.0),  # SL hit (wide bear SL above entry)
        )
        cfg_cons = ExitConfig("ict_iofed", fill_mode="conservative")
        result = compute_exit(window_bear, future, 2, cfg_cons)
        assert result.filled is True, "Bear conservative must fill when close >= limit"


# ===========================================================================
# 4 — All-seeds aggregation math
# ===========================================================================

class TestAggregateSeedSummaries:
    """Verify _aggregate_seed_summaries computes mean±std correctly.

    Import the function directly from scripts/inspect_models.py.
    """

    @staticmethod
    def _import_aggregate():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "inspect_models",
            str(_ROOT / "scripts" / "inspect_models.py"),
        )
        mod = importlib.util.load_from_spec = None  # unused; use importlib direct
        # Standard sys.path import since _ROOT is already on path
        if "inspect_models" not in sys.modules:
            import importlib
            spec2 = importlib.util.spec_from_file_location(
                "inspect_models",
                str(_ROOT / "scripts" / "inspect_models.py"),
            )
            mod = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(mod)
            sys.modules["inspect_models"] = mod
        return sys.modules["inspect_models"]

    def _get_fn(self):
        mod = self._import_aggregate()
        return mod._aggregate_seed_summaries, mod._resolve_multisym_checkpoints

    def test_mean_std_computed_correctly(self):
        """Known inputs → exact mean and std (ddof=1).

        Values: after_cost_total_r = [1.0, 3.0, 5.0, 7.0, 9.0]
        mean = 5.0, std (ddof=1) = sqrt(10) ≈ 3.1623
        """
        aggregate_fn, _ = self._get_fn()
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        seeds = [0, 17, 42, 123, 2024]
        seed_summaries = [
            (seed, {"after_cost_total_r": v, "win_rate": 0.5, "n_trades": 10})
            for seed, v in zip(seeds, values)
        ]
        result = aggregate_fn(seed_summaries)
        assert result["mean_after_cost_total_r"] == pytest.approx(5.0)
        assert result["std_after_cost_total_r"] == pytest.approx(
            float(np.std([1.0, 3.0, 5.0, 7.0, 9.0], ddof=1))
        )

    def test_win_rate_aggregated(self):
        """mean_win_rate = mean of input win_rates."""
        aggregate_fn, _ = self._get_fn()
        win_rates = [0.4, 0.5, 0.6, 0.5, 0.45]
        seed_summaries = [
            (seed, {"after_cost_total_r": 0.0, "win_rate": wr, "n_trades": 5})
            for seed, wr in zip([0, 17, 42, 123, 2024], win_rates)
        ]
        result = aggregate_fn(seed_summaries)
        assert result["mean_win_rate"] == pytest.approx(np.mean(win_rates))

    def test_n_trades_aggregated(self):
        """mean_n_trades = mean of input n_trades."""
        aggregate_fn, _ = self._get_fn()
        ns = [10, 12, 8, 15, 11]
        seed_summaries = [
            (seed, {"after_cost_total_r": 0.0, "win_rate": 0.5, "n_trades": n})
            for seed, n in zip([0, 17, 42, 123, 2024], ns)
        ]
        result = aggregate_fn(seed_summaries)
        assert result["mean_n_trades"] == pytest.approx(np.mean(ns))

    def test_per_seed_rows_present(self):
        """per_seed key holds one dict per seed with correct seed and value."""
        aggregate_fn, _ = self._get_fn()
        seed_summaries = [
            (42, {"after_cost_total_r": 2.5, "win_rate": 0.55, "n_trades": 8}),
        ]
        result = aggregate_fn(seed_summaries)
        assert len(result["per_seed"]) == 1
        assert result["per_seed"][0]["seed"] == 42
        assert result["per_seed"][0]["after_cost_total_r"] == pytest.approx(2.5)

    def test_single_seed_std_is_zero(self):
        """With one seed, std = 0.0 (not NaN; ddof=1 handled as special case)."""
        aggregate_fn, _ = self._get_fn()
        seed_summaries = [
            (42, {"after_cost_total_r": 3.7, "win_rate": 0.5, "n_trades": 5}),
        ]
        result = aggregate_fn(seed_summaries)
        assert result["std_after_cost_total_r"] == pytest.approx(0.0)

    def test_resolve_multisym_checkpoints_empty_dir_returns_empty(self):
        """Missing arch directory → arch is omitted (not errored); graceful skip.

        Uses a tmp directory with no checkpoint files.
        """
        import tempfile, os
        _, resolve_fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_fn(Path(tmpdir))
        # No files in tmpdir → no archs found
        assert result == {}

    def test_resolve_multisym_checkpoints_finds_existing_files(self):
        """Creates fake .pt files on disk → resolver returns correct (seed, path) pairs."""
        import tempfile
        _, resolve_fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake lstm multisym checkpoints for seeds 0 and 42
            subdir = Path(tmpdir) / "lstm_h1_multisym"
            subdir.mkdir(parents=True)
            (subdir / "lstm_seed0.pt").touch()
            (subdir / "lstm_seed42.pt").touch()

            result = resolve_fn(Path(tmpdir))

        assert "lstm" in result
        seeds_found = [s for s, _ in result["lstm"]]
        assert 0 in seeds_found
        assert 42 in seeds_found
        assert 17 not in seeds_found   # not created

    def test_resolve_multisym_checkpoints_xlstm_empty_gracefully_omitted(self):
        """xlstm_multisym dir exists but is empty → xlstm omitted from result.

        This is the D4a flag scenario: xlstm has no checkpoints.
        """
        import tempfile
        _, resolve_fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty xlstm dir
            (Path(tmpdir) / "xlstm_h1_multisym").mkdir(parents=True)
            # Create one valid lstm checkpoint so result is non-empty
            lstm_dir = Path(tmpdir) / "lstm_h1_multisym"
            lstm_dir.mkdir(parents=True)
            (lstm_dir / "lstm_seed42.pt").touch()

            result = resolve_fn(Path(tmpdir))

        assert "xlstm" not in result, "Empty xlstm dir must be gracefully omitted"
        assert "lstm" in result


# ===========================================================================
# 5 — Backward compatibility: defaults byte-identical to pre-enhancement path
# ===========================================================================

class TestBackwardCompat:
    """REVERT-SENSITIVE: ExitConfig() defaults must preserve all pre-enhancement behavior.

    If confidence_threshold or fill_mode defaults were accidentally changed,
    these tests would catch it by comparing against explicitly pre-enhancement
    equivalent configs.
    """

    def test_exit_config_defaults(self):
        """ExitConfig() has confidence_threshold=0.0 and fill_mode='optimistic'."""
        cfg = ExitConfig()
        assert cfg.confidence_threshold == 0.0
        assert cfg.fill_mode == "optimistic"

    def test_compute_trades_probas_none_identical_to_no_probas_path(self):
        """probas=None → identical trade list as if probas param did not exist.

        REVERT-SENSITIVE: adding a probas param that inadvertently changes the
        default branch (e.g. skips all trades) would be caught here.
        """
        ws, futs, ts, preds, probas = _build_conf_batch()
        cfg = ExitConfig("fixed_2r")
        trades_no_probas = compute_trades_for_model(ws, futs, ts, preds, exit_config=cfg)
        trades_with_none = compute_trades_for_model(ws, futs, ts, preds, probas=None, exit_config=cfg)
        assert len(trades_no_probas) == len(trades_with_none)
        for t1, t2 in zip(trades_no_probas, trades_with_none):
            assert t1.direction == t2.direction
            assert t1.outcome == t2.outcome
            assert t1.r_multiple == pytest.approx(t2.r_multiple)

    def test_fill_mode_optimistic_default_identical_to_explicit_optimistic(self):
        """ExitConfig('ict_iofed') == ExitConfig('ict_iofed', fill_mode='optimistic')."""
        window = _make_window(400, 394, 408, 395, 402, 396, base_price=400.0)
        future = _make_future(
            (397.0, 398.0, 395.5, 396.8),   # low<=limit, close>limit
            (397.0, 408.0, 396.0, 405.0),
            (405.0, 410.0, 404.0, 408.0),
        )
        cfg_default = ExitConfig("ict_iofed")
        cfg_explicit = ExitConfig("ict_iofed", fill_mode="optimistic")
        r1 = compute_exit(window, future, 1, cfg_default)
        r2 = compute_exit(window, future, 1, cfg_explicit)
        assert r1.filled == r2.filled
        assert r1.outcome == r2.outcome
        assert r1.entry == pytest.approx(r2.entry)

    def test_summarise_trades_unchanged_when_realism_off(self):
        """summarise_trades without report_realism: new fields NOT present."""
        trades = [_make_trade_outcome("tp", r_multiple=2.0)]
        summary = summarise_trades(trades)
        assert "after_cost_total_r" not in summary
        assert "median_r" not in summary
        assert "n_outlier_r" not in summary

    def test_full_fixed2r_pipeline_with_default_config(self):
        """Full compute_trades + summarise with defaults produces same result as before.

        Uses 3 bull TP trades — the pipeline must not lose trades or alter outcomes
        when new default fields are present.
        """
        w = _make_window(400, 395, 408, 396, 403, 401, base_price=400.0)
        entry = 403.0
        sl = 400.0 - TICK
        risk = entry - sl
        tp = entry + 2.0 * risk
        fut = _make_future(
            (entry, 406, 402, 404),
            (405, tp + 2.0, 404, 405),
        )
        windows_raw = np.stack([w, w, w])
        future_ohlcv = np.stack([fut, fut, fut])
        preds = np.array([1, 1, 1], dtype=np.int64)
        trades = compute_trades_for_model(
            windows_raw, future_ohlcv, [[], [], []], preds
        )
        summary = summarise_trades(trades, strategy_name="fixed_2r")
        assert summary["n_signals"] == 3
        assert summary["n_trades"] == 3
        assert summary["win_rate"] == pytest.approx(1.0)
        assert summary["total_r"] == pytest.approx(6.0)


# ===========================================================================
# 6 — Validation: out-of-range / invalid values raise
# ===========================================================================

class TestExitConfigValidation:
    """ExitConfig.__post_init__ must raise for invalid confidence_threshold and fill_mode."""

    def test_negative_confidence_threshold_raises(self):
        """confidence_threshold < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            ExitConfig("fixed_2r", confidence_threshold=-0.01)

    def test_confidence_threshold_above_one_is_valid(self):
        """confidence_threshold > 1.0 is allowed (it will just produce 0 trades)."""
        # No raise expected — threshold=1.1 is valid; it just never fires.
        cfg = ExitConfig("fixed_2r", confidence_threshold=1.1)
        assert cfg.confidence_threshold == pytest.approx(1.1)

    def test_invalid_fill_mode_raises(self):
        """fill_mode not in {'optimistic', 'conservative'} must raise ValueError."""
        with pytest.raises(ValueError, match="fill_mode"):
            ExitConfig("fixed_2r", fill_mode="aggressive")

    def test_fill_mode_case_sensitive(self):
        """fill_mode='Optimistic' (wrong case) must raise — not silently accepted."""
        with pytest.raises(ValueError, match="fill_mode"):
            ExitConfig("fixed_2r", fill_mode="Optimistic")

    def test_valid_confidence_threshold_zero(self):
        """confidence_threshold=0.0 is valid (default off)."""
        cfg = ExitConfig("fixed_2r", confidence_threshold=0.0)
        assert cfg.confidence_threshold == pytest.approx(0.0)

    def test_valid_fill_mode_conservative(self):
        """fill_mode='conservative' is valid."""
        cfg = ExitConfig("fixed_2r", fill_mode="conservative")
        assert cfg.fill_mode == "conservative"
