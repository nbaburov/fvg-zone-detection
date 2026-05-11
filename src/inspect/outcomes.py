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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TICK = 0.01  # SPY tick size


@dataclass
class Trade:
    window_idx: int
    entry_ts: pd.Timestamp           # timestamp of entry bar (first future bar)
    direction: int                   # 1 = bull, 2 = bear
    entry: float
    sl: float
    tp: float
    outcome: str                     # "tp" | "sl" | "undecided" | "no_future"
    exit_idx_in_future: int          # bar index within future_ohlcv where exit occurred; -1 if none
    exit_ts: pd.Timestamp | None
    exit_price: float                # entry if no_future / undecided
    r_multiple: float                # realised reward multiple (sign + magnitude)


def compute_trades_for_model(
    windows_raw: np.ndarray,
    future_ohlcv: np.ndarray,
    future_timestamps: list[list],
    preds: np.ndarray,
    tp_rr: float = 2.0,
) -> list[Trade]:
    """Produce Trade records for every positive prediction by this model."""
    trades: list[Trade] = []
    if future_ohlcv.shape[1] == 0:
        return trades

    n = len(preds)
    for i in range(n):
        d = int(preds[i])
        if d == 0:
            continue

        bar_56 = windows_raw[i, 56]
        bar_58 = windows_raw[i, 58]
        future = future_ohlcv[i]
        if np.isnan(future[0, 0]):
            trades.append(_no_future_trade(i, d, bar_56, bar_58, tp_rr))
            continue

        entry = float(future[0, 0])  # open of first future bar
        entry_ts = future_timestamps[i][0] if future_timestamps and future_timestamps[i] else pd.NaT

        if d == 1:
            sl = float(bar_56[1]) - TICK    # bar 56 high - tick
            risk = entry - sl
            tp = entry + tp_rr * risk
        else:
            sl = float(bar_56[2]) + TICK    # bar 56 low + tick
            risk = sl - entry
            tp = entry - tp_rr * risk

        if risk <= 0:
            # Degenerate gap geometry — skip with no_future-style record
            trades.append(Trade(
                window_idx=i, entry_ts=entry_ts, direction=d,
                entry=entry, sl=sl, tp=tp, outcome="undecided",
                exit_idx_in_future=-1, exit_ts=None,
                exit_price=entry, r_multiple=0.0,
            ))
            continue

        outcome, exit_idx, exit_price = _walk_future(future, d, sl, tp)
        if exit_idx >= 0 and future_timestamps and future_timestamps[i]:
            exit_ts = future_timestamps[i][exit_idx] if exit_idx < len(future_timestamps[i]) else None
        else:
            exit_ts = None

        if outcome == "tp":
            r_multiple = tp_rr
        elif outcome == "sl":
            r_multiple = -1.0
        else:
            # undecided — mark to market at last available future bar's close
            last_valid = _last_valid_close(future)
            r_multiple = (last_valid - entry) / risk if d == 1 else (entry - last_valid) / risk
            exit_price = last_valid

        trades.append(Trade(
            window_idx=i, entry_ts=entry_ts, direction=d,
            entry=entry, sl=sl, tp=tp, outcome=outcome,
            exit_idx_in_future=exit_idx, exit_ts=exit_ts,
            exit_price=exit_price, r_multiple=r_multiple,
        ))

    return trades


def _walk_future(future: np.ndarray, direction: int, sl: float, tp: float) -> tuple[str, int, float]:
    """Return (outcome, bar_idx, exit_price). outcome in {tp, sl, undecided}."""
    for j in range(future.shape[0]):
        if np.isnan(future[j, 0]):
            break
        hi = float(future[j, 1])
        lo = float(future[j, 2])
        if direction == 1:
            sl_hit = lo <= sl
            tp_hit = hi >= tp
            if sl_hit and tp_hit:
                return "sl", j, sl     # conservative: SL first
            if sl_hit:
                return "sl", j, sl
            if tp_hit:
                return "tp", j, tp
        else:
            sl_hit = hi >= sl
            tp_hit = lo <= tp
            if sl_hit and tp_hit:
                return "sl", j, sl
            if sl_hit:
                return "sl", j, sl
            if tp_hit:
                return "tp", j, tp
    return "undecided", -1, 0.0


def _last_valid_close(future: np.ndarray) -> float:
    for j in range(future.shape[0] - 1, -1, -1):
        if not np.isnan(future[j, 3]):
            return float(future[j, 3])
    return float("nan")


def _no_future_trade(i: int, d: int, bar_56: np.ndarray, bar_58: np.ndarray, tp_rr: float) -> Trade:
    return Trade(
        window_idx=i, entry_ts=pd.NaT, direction=d,
        entry=float("nan"), sl=float("nan"), tp=float("nan"),
        outcome="no_future", exit_idx_in_future=-1, exit_ts=None,
        exit_price=float("nan"), r_multiple=0.0,
    )


def summarise_trades(trades: list[Trade]) -> dict:
    """Aggregate counts + win-rate + total R."""
    n = len(trades)
    counts = {"tp": 0, "sl": 0, "undecided": 0, "no_future": 0}
    total_r = 0.0
    for t in trades:
        counts[t.outcome] = counts.get(t.outcome, 0) + 1
        total_r += t.r_multiple
    decided = counts["tp"] + counts["sl"]
    win_rate = counts["tp"] / decided if decided > 0 else 0.0
    return {
        "n_trades": n,
        "n_tp": counts["tp"],
        "n_sl": counts["sl"],
        "n_undecided": counts["undecided"],
        "n_no_future": counts["no_future"],
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": total_r / n if n > 0 else 0.0,
    }
