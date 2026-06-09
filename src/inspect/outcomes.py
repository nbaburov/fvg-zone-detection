"""outcomes.py — Compute realised trade outcomes (TP/SL/undecided) per model.

Outcome semantics
-----------------
For each window where a model predicts bullish (1) or bearish (2) FVG:
- The 60-bar input window's last bar is the N+2 reaction candle (label bar).
- Entry = open of the FIRST future bar (bar index 60 from the input window's start).
- Bull FVG gap zone = [high[N-1], low[N+1]] = [raw[56].high, raw[58].low].
- Bear FVG gap zone = [high[N+1], low[N-1]] = [raw[58].high, raw[56].low].
- SL = opposite edge of gap (the edge price came from):
  * Bull: SL = raw[56].high - tick (below gap lower edge)
  * Bear: SL = raw[56].low + tick (above gap upper edge)
- R = abs(entry - SL). TP = entry + tp_rr * R (bull) / entry - tp_rr * R (bear).
- Walk future bars; first to touch (TP|SL) wins. Tie within a bar = SL (conservative).
- If neither hit in `lookahead_bars`, outcome = undecided.

Back-compat note
----------------
``Trade`` is an alias for ``TradeOutcome`` (defined in ``src.strategy.exits``).
Existing importers (``viz.py``, ``report.py``, tests) continue to work unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.exits import (
    ExitConfig,
    TradeOutcome,
    compute_exit,
)

# Back-compat alias — do NOT remove; viz.py and report.py import this name.
Trade = TradeOutcome

__all__ = ["Trade", "TradeOutcome", "ExitConfig", "compute_trades_for_model", "summarise_trades"]


def compute_trades_for_model(
    windows_raw: np.ndarray,
    future_ohlcv: np.ndarray,
    future_timestamps: list[list],
    preds: np.ndarray,
    probas: np.ndarray | None = None,
    tp_rr: float = 2.0,
    exit_config: ExitConfig | None = None,
) -> list[Trade]:
    """Produce Trade records for every positive prediction by this model.

    Parameters
    ----------
    windows_raw : np.ndarray, shape (N, 60, 5)
    future_ohlcv : np.ndarray, shape (N, L, ≥4)
    future_timestamps : list[list]
        Per-window list of bar timestamps.
    preds : np.ndarray, shape (N,)
        Integer class predictions (0=none, 1=bull, 2=bear).
    probas : np.ndarray | None, shape (N, 3)
        Class probabilities.  When provided and
        ``exit_config.confidence_threshold > 0``, predictions where
        ``max(probas[i,1], probas[i,2]) < confidence_threshold`` are skipped
        (not simulated as trades).  Default ``None`` = no filter.

        **Calibration note:** confidence is raw softmax max(P_bull, P_bear),
        NOT a calibrated probability.  Valid as a relative conviction filter
        only.
    tp_rr : float
        Reward multiple for V1 / fallback.  Forwarded to ExitConfig when
        ``exit_config`` is None.
    exit_config : ExitConfig | None
        Strategy configuration.  Defaults to ``ExitConfig("fixed_2r", tp_rr)``.

    Returns
    -------
    list[Trade]
        One entry per eligible positive prediction (post confidence filter).
    """
    if exit_config is None:
        exit_config = ExitConfig(strategy="fixed_2r", tp_rr=tp_rr)

    _apply_conf_filter = (
        probas is not None and exit_config.confidence_threshold > 0.0
    )

    trades: list[Trade] = []
    if future_ohlcv.shape[1] == 0:
        return trades

    n = len(preds)
    for i in range(n):
        d = int(preds[i])
        if d == 0:
            continue

        # Confidence filter (Addition 1).  Skips low-conviction predictions.
        # n_signals (all argmax-positive) is tracked by the caller via
        # int((preds != 0).sum()); n_eligible = len(returned trades) for
        # market-entry, or can be inferred from n_signals vs no_fill for limit.
        if _apply_conf_filter:
            confidence = max(float(probas[i, 1]), float(probas[i, 2]))  # type: ignore[index]
            if confidence < exit_config.confidence_threshold:
                continue

        future = future_ohlcv[i]
        entry_ts = (
            future_timestamps[i][0]
            if future_timestamps and future_timestamps[i]
            else pd.NaT
        )

        result = compute_exit(
            window_raw=windows_raw[i],
            future_ohlcv=future,
            direction=d,
            config=exit_config,
            window_idx=i,
            entry_ts=entry_ts,
            future_timestamps=future_timestamps[i] if future_timestamps else None,
        )
        trades.append(result)

    return trades


_MARKET_ENTRY_STRATEGIES = frozenset({"fixed_2r"})


def summarise_trades(
    trades: list[Trade],
    strategy_name: str | None = None,
    realism_config: ExitConfig | None = None,
) -> dict:
    """Aggregate counts + win-rate + total R.

    Parameters
    ----------
    trades : list[Trade]
    strategy_name : str | None
        When provided, market-entry variants (``fixed_2r``) always receive
        ``fill_rate = float("nan")`` (displayed as "N/A") — a numeric fill_rate
        below 1.0 on those strategies only reflects ``no_future`` windows, which
        is misleading.  Limit-entry variants (V2/V3/V4) retain the computed
        ratio.
    realism_config : ExitConfig | None
        When provided and ``realism_config.report_realism`` is ``True``, five
        additional keys are appended to the return dict:

        - ``"median_r"`` — median r_multiple across filled trades
        - ``"after_cost_total_r"`` — total R with per-trade cost drag subtracted
        - ``"after_cost_avg_r"`` — after_cost_total_r / n_trades
        - ``"n_outlier_r"`` — count of filled trades where ``|r_multiple| > 10``
        - ``"winsorized_total_r"`` — sum of ``clip(r_multiple, -10, 10)``

        Cost drag per trade (in R units, 1-share normalisation):
        ``(slippage_ticks × 0.01 × 2 + commission_per_share × 2) / risk_dollars``
        where ``risk_dollars = abs(entry - sl)``.  Applied to all filled trades
        (including undecided); not applied to ``no_fill`` / ``no_future``.

    Notes
    -----
    ``"no_fill"`` outcomes are counted in the fill-rate denominator but
    excluded from win-rate, total-R, and avg-R calculations.  Existing keys
    are never modified — Guard 3 only adds new keys when ``report_realism``
    is active.
    """
    counts: dict[str, int] = {
        "tp": 0, "sl": 0, "undecided": 0, "no_future": 0, "no_fill": 0
    }
    total_r = 0.0
    n_filled = 0
    n_swing_fallback = 0

    # Realism accumulators (Guard 2 + Guard 3)
    _realism_on = (
        realism_config is not None and realism_config.report_realism
    )
    _tick_size = 0.01
    if _realism_on:
        _cost_per_rt_dollars = (
            realism_config.slippage_ticks * _tick_size * 2  # type: ignore[union-attr]
            + realism_config.commission_per_share * 2       # type: ignore[union-attr]
        )
    else:
        _cost_per_rt_dollars = 0.0

    after_cost_total_r = 0.0
    r_multiples_filled: list[float] = []

    for t in trades:
        counts[t.outcome] = counts.get(t.outcome, 0) + 1
        if t.filled:
            n_filled += 1
            total_r += t.r_multiple
            if _realism_on:
                r_multiples_filled.append(t.r_multiple)
                risk_dollars = abs(t.entry - t.sl) if not (
                    np.isnan(t.entry) or np.isnan(t.sl)
                ) else 0.0
                cost_drag_r = (
                    _cost_per_rt_dollars / risk_dollars
                    if risk_dollars > 0 else 0.0
                )
                after_cost_total_r += t.r_multiple - cost_drag_r
        if getattr(t, "swing_fallback", False):
            n_swing_fallback += 1

    n_signals = len(trades)
    n_trades = n_filled  # trades that actually filled
    decided = counts["tp"] + counts["sl"]
    win_rate = counts["tp"] / decided if decided > 0 else 0.0

    # fill_rate: NaN for market-entry strategies (fixed_2r always fills —
    # any ratio < 1.0 only reflects no_future windows, which is misleading).
    # For limit-entry variants: n_filled / n_signals.
    if strategy_name in _MARKET_ENTRY_STRATEGIES:
        fill_rate = float("nan")
    else:
        has_no_fill = counts["no_fill"] > 0
        if has_no_fill or n_signals > 0:
            fill_rate = n_filled / n_signals if n_signals > 0 else float("nan")
        else:
            fill_rate = 1.0

    # swing_fallback_rate among filled trades that use swing TP (V2/V3/V4)
    swing_fallback_rate = (
        n_swing_fallback / n_trades if n_trades > 0 else 0.0
    )

    result = {
        "n_signals": n_signals,
        "n_trades": n_trades,
        "n_tp": counts["tp"],
        "n_sl": counts["sl"],
        "n_undecided": counts["undecided"],
        "n_no_future": counts["no_future"],
        "n_no_fill": counts["no_fill"],
        "fill_rate": fill_rate,
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": total_r / n_trades if n_trades > 0 else 0.0,
        "swing_fallback_rate": swing_fallback_rate,
    }

    # Guard 3 — robust realism stats (only when report_realism=True)
    if _realism_on:
        arr = np.array(r_multiples_filled, dtype=float) if r_multiples_filled else np.array([], dtype=float)
        result["median_r"] = float(np.median(arr)) if arr.size > 0 else 0.0
        result["after_cost_total_r"] = after_cost_total_r
        result["after_cost_avg_r"] = after_cost_total_r / n_trades if n_trades > 0 else 0.0
        result["n_outlier_r"] = int(np.sum(np.abs(arr) > 10)) if arr.size > 0 else 0
        result["winsorized_total_r"] = float(np.sum(np.clip(arr, -10.0, 10.0))) if arr.size > 0 else 0.0

    return result
