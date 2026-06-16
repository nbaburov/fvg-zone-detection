"""exits.py — FVG exit-strategy simulation module.

Four exit variants are supported:

  V1  fixed_2r    Market entry at N+2 open; SL at gap edge; TP = entry ± 2R.
                  Reproduces legacy outcomes.py logic byte-for-byte.
  V2  ict_iofed   Limit entry at gap near-edge (bar_58 low/high); SL at
                  candle-1 far edge; TP = causal rolling swing high/low.
  V3  ce_50pct    Limit entry at FVG 50% midpoint (CE); SL at candle-1 far
                  edge; TP = causal rolling swing.
  V4  tradinglab  Limit entry at gap near-edge (same as V2); SL at candle-2
                  impulse (tighter); TP = causal rolling swing.

Limit-entry variants (V2/V3/V4) apply an optimistic fill model: scan
``future_ohlcv[0:fill_timeout_bars]`` bar by bar; a bull fills if
``future[j].low <= limit``; bear if ``future[j].high >= limit``.  Fill price
= exact limit.  If no fill within the timeout window the outcome is
``"no_fill"`` (excluded from win-rate / total-R; counted in fill-rate
denominator).

Candle indexing (confirmed against outcomes.py + plan D2):
  bar_56 = window[56]  — candle-1 (pre-impulse)
  bar_57 = window[57]  — candle-2 (impulse)
  bar_58 = window[58]  — candle-3 (reaction / gap near-edge)

Column order in window_raw: [open=0, high=1, low=2, close=3, volume=4]

TICK constant: 0.01 (SPY tick size).

Phase 4 interface note
----------------------
When ``future_ohlcv`` has 0 rows (or is None) ``compute_exit`` still
returns a valid ``TradeOutcome`` with ``entry``, ``sl``, and ``tp``
populated (outcome = ``"no_future"``).  A live ``PaperExecutor`` can
call ``compute_exit(future_ohlcv=empty, config=ExitConfig(strategy))``
and read ``.sl`` / ``.tp`` for bracket-order placement without
triggering any fill simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TICK: float = 0.01  # SPY tick size

_STRATEGIES = frozenset({"fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExitConfig:
    """Configuration for a single exit-strategy variant.

    Parameters
    ----------
    strategy : str
        One of ``"fixed_2r"``, ``"ict_iofed"``, ``"ce_50pct"``,
        ``"tradinglab"``.
    tp_rr : float
        Reward multiple used for TP in V1 and as fallback for swing-TP
        variants when the swing level does not exceed entry.  Default 2.0.
    swing_lookback : int
        Number of bars for causal swing-high/low lookback (V2/V3/V4).
        The lookback window ends at candle-3 (bar_58) inclusive.
        Default 20 (bars 39–58).
    fill_timeout_bars : int
        Maximum number of future bars to scan for a limit fill (V2/V3/V4).
        Default 10 (≈ 2 trading days on H1).
    min_stop_atr_k : float | None
        When set, enforces a minimum stop distance of
        ``min_stop_atr_k × ATR(window[0:60])``.  ATR is the mean true range
        across the 60-bar input window (causal — all bars precede the entry
        bar).  When the raw stop is tighter than the floor the SL is widened
        in the adverse direction before the TP/SL walk, so the outcome mix
        (win-rate + per-win R) changes correctly.
        Default ``None`` = guard disabled (backward-compatible).
    slippage_ticks : int
        Ticks of slippage on each leg (entry + exit).  Applied as a cost
        drag in R units inside ``summarise_trades``.
        Default ``0`` = no slippage (backward-compatible).
    commission_per_share : float
        Per-share round-trip commission (entry + exit combined).
        Default ``0.0`` = no commission (backward-compatible).
    report_realism : bool
        When ``True``, ``summarise_trades`` appends realism columns:
        ``median_r``, ``after_cost_total_r``, ``after_cost_avg_r``,
        ``n_outlier_r``, ``winsorized_total_r``.
        Default ``False`` = backward-compatible summarise output.
    confidence_threshold : float
        When > 0.0, ``compute_trades_for_model`` skips any argmax-positive
        prediction where ``max(probas[i,1], probas[i,2]) < confidence_threshold``.
        Default ``0.0`` = no filter (backward-compatible).

        **Calibration caveat:** this is raw softmax max(P_bull, P_bear), NOT a
        calibrated probability.  Use as a relative conviction filter only.
        With ~97 % none-class imbalance, directional probas may rarely exceed
        ~0.7; high thresholds may yield few/zero eligible trades.
    fill_mode : str
        ``"optimistic"`` (default): limit fills when ``bar.low <= limit``
        (bull) or ``bar.high >= limit`` (bear).  Byte-identical to previous
        behaviour.
        ``"conservative"``: limit fills only when ``bar.close <= limit``
        (bull) or ``bar.close >= limit`` (bear) — the bar must *close*
        through the level.  Reduces fill rate; overstates live tradability
        less than optimistic.
    """

    strategy: str = "fixed_2r"
    tp_rr: float = 2.0
    swing_lookback: int = 20
    fill_timeout_bars: int = 10
    min_stop_atr_k: Optional[float] = None
    slippage_ticks: int = 0
    commission_per_share: float = 0.0
    report_realism: bool = False
    confidence_threshold: float = 0.0
    fill_mode: str = "optimistic"

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Must be one of: {sorted(_STRATEGIES)}"
            )
        if self.min_stop_atr_k is not None and self.min_stop_atr_k <= 0:
            raise ValueError(
                f"min_stop_atr_k must be > 0 when set, got {self.min_stop_atr_k}"
            )
        if self.slippage_ticks < 0:
            raise ValueError(
                f"slippage_ticks must be >= 0, got {self.slippage_ticks}"
            )
        if self.commission_per_share < 0:
            raise ValueError(
                f"commission_per_share must be >= 0, got {self.commission_per_share}"
            )
        if self.confidence_threshold < 0.0:
            raise ValueError(
                f"confidence_threshold must be >= 0.0, got {self.confidence_threshold}"
            )
        if self.fill_mode not in {"optimistic", "conservative"}:
            raise ValueError(
                f"fill_mode must be 'optimistic' or 'conservative', got '{self.fill_mode}'"
            )


# ---------------------------------------------------------------------------
# Trade outcome
# ---------------------------------------------------------------------------

@dataclass
class TradeOutcome:
    """Realised (or attempted) trade outcome for a single FVG window.

    Parameters
    ----------
    window_idx : int
        Index of the window within the batch.
    entry_ts : pd.Timestamp
        Timestamp of the entry bar (first future bar for market; fill bar for
        limit).  ``pd.NaT`` when unknown.
    direction : int
        1 = bullish FVG, 2 = bearish FVG.
    entry : float
        Actual entry price (NaN when no_future).
    sl : float
        Stop-loss price.
    tp : float
        Take-profit price.
    outcome : str
        One of ``"tp"``, ``"sl"``, ``"undecided"``, ``"no_fill"``,
        ``"no_future"``.
    exit_idx_in_future : int
        Bar index within ``future_ohlcv`` where the exit (TP/SL) occurred.
        ``-1`` if none.
    exit_ts : pd.Timestamp | None
        Timestamp of the exit bar.  ``None`` if no exit.
    exit_price : float
        Price at which the trade was closed.  Entry price when undecided /
        no_future.
    r_multiple : float
        Realised reward multiple (signed: positive = profit).
    filled : bool
        ``True`` for market entries (always) and limit entries that filled.
        ``False`` for ``"no_fill"`` outcomes.
    swing_fallback : bool
        ``True`` when swing-TP was not beyond entry and the fallback 2R TP
        was used instead.  Always ``False`` for ``fixed_2r``.
    """

    window_idx: int
    entry_ts: pd.Timestamp
    direction: int
    entry: float
    sl: float
    tp: float
    outcome: str
    exit_idx_in_future: int
    exit_ts: Optional[pd.Timestamp]
    exit_price: float
    r_multiple: float
    filled: bool = True
    swing_fallback: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_exit(
    window_raw: np.ndarray,
    future_ohlcv: np.ndarray,
    direction: int,
    config: ExitConfig,
    window_idx: int = 0,
    entry_ts: pd.Timestamp = pd.NaT,
    future_timestamps: Optional[List[pd.Timestamp]] = None,
) -> TradeOutcome:
    """Compute a single realised trade outcome for one FVG window.

    Parameters
    ----------
    window_raw : np.ndarray, shape (60, 5)
        Un-normalised OHLCV window.  Column order: open, high, low, close,
        volume.
    future_ohlcv : np.ndarray, shape (L, 4) or (L, 5)
        OHLC(V) future bars starting at the bar AFTER bar_58 (i.e. N+2 open
        is ``future_ohlcv[0, 0]`` for market entry).  Pass an empty array
        (shape ``(0, …)``) or ``None`` for live SL/TP extraction without
        simulation.
    direction : int
        1 = bullish, 2 = bearish.
    config : ExitConfig
        Strategy configuration.
    window_idx : int
        Batch index for identification.
    entry_ts : pd.Timestamp
        Timestamp of the first future bar (market entry) or the fill bar
        (limit entry).  May be ``pd.NaT``.
    future_timestamps : list[pd.Timestamp] | None
        Per-bar timestamps for ``future_ohlcv``.  Used to populate
        ``exit_ts``.

    Returns
    -------
    TradeOutcome
        Fully populated outcome.  ``entry``, ``sl``, ``tp`` are always set
        (NaN only when window geometry is degenerate).  ``outcome`` is
        ``"no_future"`` when ``future_ohlcv`` has 0 rows.
    """
    if future_ohlcv is None:
        future_ohlcv = np.empty((0, 4), dtype=np.float32)

    strategy = config.strategy

    if strategy == "fixed_2r":
        return _compute_fixed_2r(
            window_raw, future_ohlcv, direction, config,
            window_idx, entry_ts, future_timestamps,
        )
    else:
        return _compute_limit_variant(
            window_raw, future_ohlcv, direction, config,
            window_idx, entry_ts, future_timestamps,
        )


def market_tp_from_entry(
    entry: float,
    sl: float,
    tp_rr: float,
    direction: int,
) -> float:
    """Take-profit price for a market (fixed_2r) entry, given the real fill.

    Single source of truth for the *fill-time* TP geometry of a market entry:
    the planner cannot know the fill price ahead of time (entry = next-bar
    open), so the live ``PaperExecutor`` computes the TP once the entry is
    known.  This keeps that arithmetic in ``exits.py`` (§0 keystone) instead of
    inline in ``execution.py``.

    Mirrors ``_compute_fixed_2r``: ``risk = |entry - sl|`` and
    ``tp = entry ± tp_rr * risk`` (``+`` for bull, ``-`` for bear).

    Parameters
    ----------
    entry : float
        Actual entry/fill price.
    sl : float
        Stop-loss price (from the planner / ``compute_exit``).
    tp_rr : float
        Reward-to-risk multiple.
    direction : int
        1 = bull, 2 = bear.

    Returns
    -------
    float
        Take-profit price.
    """
    risk = abs(entry - sl)
    if direction == 1:
        return entry + tp_rr * risk
    return entry - tp_rr * risk


# ---------------------------------------------------------------------------
# V1 — fixed_2r (market entry, gap-edge SL, 2R TP)
# ---------------------------------------------------------------------------

def _compute_fixed_2r(
    window_raw: np.ndarray,
    future_ohlcv: np.ndarray,
    direction: int,
    config: ExitConfig,
    window_idx: int,
    entry_ts: pd.Timestamp,
    future_timestamps: Optional[List[pd.Timestamp]],
) -> TradeOutcome:
    """V1: exact reproduction of pre-refactor outcomes.py logic."""
    bar_56 = window_raw[56]

    # SL at gap-edge (candle-1 high/low ± TICK)
    if direction == 1:
        sl = float(bar_56[1]) - TICK   # bar_56.high - tick
    else:
        sl = float(bar_56[2]) + TICK   # bar_56.low + tick

    # No future data available
    if future_ohlcv.shape[0] == 0 or np.isnan(future_ohlcv[0, 0]):
        return TradeOutcome(
            window_idx=window_idx, entry_ts=pd.NaT, direction=direction,
            entry=float("nan"), sl=sl, tp=float("nan"),
            outcome="no_future", exit_idx_in_future=-1,
            exit_ts=None, exit_price=float("nan"), r_multiple=0.0,
            filled=False, swing_fallback=False,
        )

    entry = float(future_ohlcv[0, 0])  # N+2 bar open

    # Guard 1 — ATR min-stop floor (applied BEFORE TP/SL walk; floor must be
    # applied after entry is known so we can use entry as the reference for V1).
    if config.min_stop_atr_k is not None:
        sl = _apply_atr_floor(entry, sl, direction, config, window_raw)

    if direction == 1:
        risk = entry - sl
        tp = entry + config.tp_rr * risk
    else:
        risk = sl - entry
        tp = entry - config.tp_rr * risk

    if risk <= 0:
        return TradeOutcome(
            window_idx=window_idx, entry_ts=entry_ts, direction=direction,
            entry=entry, sl=sl, tp=entry,
            outcome="undecided", exit_idx_in_future=-1,
            exit_ts=None, exit_price=entry, r_multiple=0.0,
            filled=True, swing_fallback=False,
        )

    outcome, exit_idx, exit_price = _walk_future(future_ohlcv, direction, sl, tp)
    exit_ts = _get_exit_ts(future_timestamps, exit_idx)
    r_multiple = _calc_r(outcome, direction, exit_price, entry, sl, tp,
                         config.tp_rr, risk, future_ohlcv)

    return TradeOutcome(
        window_idx=window_idx, entry_ts=entry_ts, direction=direction,
        entry=entry, sl=sl, tp=tp, outcome=outcome,
        exit_idx_in_future=exit_idx, exit_ts=exit_ts,
        exit_price=exit_price if exit_idx >= 0 else entry,
        r_multiple=r_multiple, filled=True, swing_fallback=False,
    )


# ---------------------------------------------------------------------------
# V2 / V3 / V4 — limit-entry variants
# ---------------------------------------------------------------------------

def _compute_limit_variant(
    window_raw: np.ndarray,
    future_ohlcv: np.ndarray,
    direction: int,
    config: ExitConfig,
    window_idx: int,
    entry_ts: pd.Timestamp,
    future_timestamps: Optional[List[pd.Timestamp]],
) -> TradeOutcome:
    """V2 (ict_iofed), V3 (ce_50pct), V4 (tradinglab): limit entry + swing TP."""
    bar_56 = window_raw[56]
    bar_57 = window_raw[57]
    bar_58 = window_raw[58]

    # --- Compute gap coords ---
    if direction == 1:
        gap_low = float(bar_58[2])   # bar_58.low
        gap_high = float(bar_56[1])  # bar_56.high
    else:
        gap_low = float(bar_56[2])   # bar_56.low
        gap_high = float(bar_58[1])  # bar_58.high

    # --- Entry limit price ---
    strategy = config.strategy
    if strategy in ("ict_iofed", "tradinglab"):
        # Near-edge of the gap
        if direction == 1:
            limit = float(bar_58[2])   # bar_58.low
        else:
            limit = float(bar_58[1])   # bar_58.high
    else:  # ce_50pct
        ce = (gap_high + gap_low) / 2.0
        limit = ce

    # --- SL anchor ---
    if strategy == "tradinglab":
        # Candle-2 impulse, tight
        if direction == 1:
            sl = float(bar_57[2]) - TICK   # bar_57.low - tick
        else:
            sl = float(bar_57[1]) + TICK   # bar_57.high + tick
    else:
        # Candle-1 far edge, wide (V2 and V3)
        if direction == 1:
            sl = float(bar_56[2]) - TICK   # bar_56.low - tick
        else:
            sl = float(bar_56[1]) + TICK   # bar_56.high + tick

    # Guard 1 — ATR min-stop floor for limit-entry variants.
    # Reference = limit price (prospective entry).  Floor applied before the
    # TP/SL walk so the outcome mix resolves on the floored SL.
    if config.min_stop_atr_k is not None:
        sl = _apply_atr_floor(limit, sl, direction, config, window_raw)

    # --- Swing TP (causal, D3) ---
    swing_start = max(0, 59 - config.swing_lookback)  # 59 - 20 = 39
    if direction == 1:
        swing_tp = float(window_raw[swing_start:59, 1].max())  # bars 39:59 highs
    else:
        swing_tp = float(window_raw[swing_start:59, 2].min())  # bars 39:59 lows

    # --- No future data: return SL/TP/limit for live use ---
    if future_ohlcv.shape[0] == 0:
        # Determine risk from limit and SL to set tp
        if direction == 1:
            risk_preview = limit - sl
            tp_preview = swing_tp if swing_tp > limit else limit + config.tp_rr * max(risk_preview, 0.0)
        else:
            risk_preview = sl - limit
            tp_preview = swing_tp if swing_tp < limit else limit - config.tp_rr * max(risk_preview, 0.0)
        return TradeOutcome(
            window_idx=window_idx, entry_ts=pd.NaT, direction=direction,
            entry=limit, sl=sl, tp=tp_preview,
            outcome="no_future", exit_idx_in_future=-1,
            exit_ts=None, exit_price=float("nan"), r_multiple=0.0,
            filled=False, swing_fallback=False,
        )

    # --- Fill simulation ---
    # optimistic (default): bar.low <= limit (bull) / bar.high >= limit (bear)
    # conservative: bar.close <= limit (bull) / bar.close >= limit (bear)
    timeout = min(config.fill_timeout_bars, future_ohlcv.shape[0])
    fill_bar = -1
    _conservative = config.fill_mode == "conservative"
    for j in range(timeout):
        if np.isnan(future_ohlcv[j, 0]):
            break
        if direction == 1:
            if _conservative:
                filled = float(future_ohlcv[j, 3]) <= limit  # close
            else:
                filled = float(future_ohlcv[j, 2]) <= limit  # low touches limit
            if filled:
                fill_bar = j
                break
        else:
            if _conservative:
                filled = float(future_ohlcv[j, 3]) >= limit  # close
            else:
                filled = float(future_ohlcv[j, 1]) >= limit  # high touches limit
            if filled:
                fill_bar = j
                break

    if fill_bar < 0:
        # No fill — not a trade
        return TradeOutcome(
            window_idx=window_idx, entry_ts=entry_ts, direction=direction,
            entry=limit, sl=sl, tp=float("nan"),
            outcome="no_fill", exit_idx_in_future=-1,
            exit_ts=None, exit_price=float("nan"), r_multiple=0.0,
            filled=False, swing_fallback=False,
        )

    # Filled at fill_bar, at exact limit price
    fill_ts = _get_exit_ts(future_timestamps, fill_bar)
    entry_price = limit
    entry_ts_actual = fill_ts if fill_ts is not None else entry_ts

    # SL/TP walk starts from future[fill_bar + 1]
    walk_start = fill_bar + 1
    if walk_start >= future_ohlcv.shape[0] or np.isnan(future_ohlcv[walk_start, 0]):
        return TradeOutcome(
            window_idx=window_idx, entry_ts=entry_ts_actual, direction=direction,
            entry=entry_price, sl=sl, tp=float("nan"),
            outcome="no_future", exit_idx_in_future=-1,
            exit_ts=None, exit_price=entry_price, r_multiple=0.0,
            filled=True, swing_fallback=False,
        )

    # Compute risk and final TP
    if direction == 1:
        risk = entry_price - sl
    else:
        risk = sl - entry_price

    if risk <= 0:
        return TradeOutcome(
            window_idx=window_idx, entry_ts=entry_ts_actual, direction=direction,
            entry=entry_price, sl=sl, tp=entry_price,
            outcome="undecided", exit_idx_in_future=-1,
            exit_ts=None, exit_price=entry_price, r_multiple=0.0,
            filled=True, swing_fallback=False,
        )

    # Swing TP validity check + fallback
    if direction == 1:
        swing_valid = swing_tp > entry_price
    else:
        swing_valid = swing_tp < entry_price

    if swing_valid:
        tp = swing_tp
        swing_fallback = False
    else:
        tp = entry_price + config.tp_rr * risk if direction == 1 else entry_price - config.tp_rr * risk
        swing_fallback = True

    future_walk = future_ohlcv[walk_start:]
    walk_ts = future_timestamps[walk_start:] if future_timestamps is not None else None

    outcome, exit_idx_rel, exit_price = _walk_future(future_walk, direction, sl, tp)
    exit_idx_abs = walk_start + exit_idx_rel if exit_idx_rel >= 0 else -1
    exit_ts = _get_exit_ts(walk_ts, exit_idx_rel)

    r_multiple = _calc_r_limit(outcome, direction, exit_price, entry_price, sl, tp,
                                config.tp_rr, risk, future_walk)

    return TradeOutcome(
        window_idx=window_idx, entry_ts=entry_ts_actual, direction=direction,
        entry=entry_price, sl=sl, tp=tp, outcome=outcome,
        exit_idx_in_future=exit_idx_abs, exit_ts=exit_ts,
        exit_price=exit_price if exit_idx_abs >= 0 else entry_price,
        r_multiple=r_multiple, filled=True, swing_fallback=swing_fallback,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_atr(window_raw: np.ndarray) -> float:
    """Return mean true range over the 60-bar input window.

    Causal — all 60 bars precede the entry bar (N+2).

    Parameters
    ----------
    window_raw : np.ndarray, shape (60, 5)
        Columns: [open, high, low, close, volume].  Must be un-normalised
        (dollar prices); raises ``ValueError`` if ``window_raw.max() <= 1.0``
        (normalised input would produce a near-zero floor that silently
        defeats the guard).

    Returns
    -------
    float
        Mean TR across bars 1–59 (59 values).  Returns ``0.0`` for degenerate
        windows where all TRs are zero — the caller skips the floor in that
        case.
    """
    if window_raw.max() <= 1.0:
        raise ValueError(
            "_compute_atr received a normalised (max <= 1.0) window; "
            "pass the un-normalised window_raw array."
        )
    highs = window_raw[1:, 1].astype(float)   # bars 1–59
    lows = window_raw[1:, 2].astype(float)
    prev_closes = window_raw[:-1, 3].astype(float)  # bars 0–58

    true_high = np.maximum(highs, prev_closes)
    true_low = np.minimum(lows, prev_closes)
    tr = true_high - true_low  # shape (59,)

    valid = tr[np.isfinite(tr)]
    if valid.size == 0 or valid.sum() == 0.0:
        return 0.0
    return float(valid.mean())


def _apply_atr_floor(
    entry_ref: float,
    sl: float,
    direction: int,
    config: ExitConfig,
    window_raw: np.ndarray,
) -> float:
    """Widen ``sl`` to the ATR-floor if it is tighter than the floor.

    Parameters
    ----------
    entry_ref : float
        Entry price (market entry) or limit price (limit-entry variants) —
        used as the reference point for measuring raw stop distance.
    sl : float
        Raw stop-loss price before the floor check.
    direction : int
        1 = bullish, 2 = bearish.
    config : ExitConfig
        Must have ``min_stop_atr_k`` set (caller is responsible).
    window_raw : np.ndarray, shape (60, 5)
        Un-normalised OHLCV window (passed to ``_compute_atr``).

    Returns
    -------
    float
        Floored SL price.  Unchanged if the raw distance already meets the
        floor, or if ATR is zero (degenerate window).
    """
    atr = _compute_atr(window_raw)
    if atr == 0.0:
        if config.min_stop_atr_k is not None:
            logger.warning(
                "_apply_atr_floor: ATR=0 on this window — min_stop_atr_k floor "
                "(k=%s) bypassed; SL returned unchanged.",
                config.min_stop_atr_k,
            )
        return sl
    floor_dist = config.min_stop_atr_k * atr  # type: ignore[operator]
    raw_dist = abs(entry_ref - sl)
    if raw_dist < floor_dist:
        if direction == 1:
            sl = entry_ref - floor_dist
        else:
            sl = entry_ref + floor_dist
    return sl


def _walk_future(
    future: np.ndarray,
    direction: int,
    sl: float,
    tp: float,
) -> tuple[str, int, float]:
    """Walk future bars to find TP/SL.  Tie within a bar → SL (conservative)."""
    for j in range(future.shape[0]):
        if np.isnan(future[j, 0]):
            break
        hi = float(future[j, 1])
        lo = float(future[j, 2])
        if direction == 1:
            sl_hit = lo <= sl
            tp_hit = hi >= tp
        else:
            sl_hit = hi >= sl
            tp_hit = lo <= tp
        if sl_hit and tp_hit:
            return "sl", j, sl   # conservative: SL first on tie
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


def _get_exit_ts(
    timestamps: Optional[List],
    idx: int,
) -> Optional[pd.Timestamp]:
    if timestamps is None or idx < 0 or idx >= len(timestamps):
        return None
    return timestamps[idx]


def _calc_r(
    outcome: str,
    direction: int,
    exit_price: float,
    entry: float,
    sl: float,
    tp: float,
    tp_rr: float,
    risk: float,
    future: np.ndarray,
) -> float:
    """Compute r_multiple for market-entry variants (V1)."""
    if outcome == "tp":
        return tp_rr
    if outcome == "sl":
        return -1.0
    # undecided — mark to market
    last = _last_valid_close(future)
    if np.isnan(last) or risk <= 0:
        return 0.0
    return (last - entry) / risk if direction == 1 else (entry - last) / risk


def _calc_r_limit(
    outcome: str,
    direction: int,
    exit_price: float,
    entry: float,
    sl: float,
    tp: float,
    tp_rr: float,
    risk: float,
    future_walk: np.ndarray,
) -> float:
    """Compute r_multiple for limit-entry variants (V2/V3/V4).

    For swing-TP trades the r_multiple on TP is NOT necessarily tp_rr —
    it is the actual distance from entry to swing_tp in R units.
    """
    if outcome == "tp":
        if risk <= 0:
            return 0.0
        return abs(tp - entry) / risk
    if outcome == "sl":
        return -1.0
    # undecided — mark to market
    last = _last_valid_close(future_walk)
    if np.isnan(last) or risk <= 0:
        return 0.0
    return (last - entry) / risk if direction == 1 else (entry - last) / risk
