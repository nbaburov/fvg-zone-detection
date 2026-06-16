"""sim_executor.py — SimFillExecutor: 1-min intrabar preview resolution.

Implements the ``Executor`` ABC for simulated (paper) fills against the live
1-min bar stream.  One instance per ``(ticker, model, strategy)`` cell
(``cell_key``).

Resolution grain
----------------
This is the **1-min preview grain** (WS-D).  R-multiples here are computed
over the realized 1-min OHLCV slice via ``compute_exit`` — the single engine
(§0 keystone).  The authoritative number is the H1-replay in WS-G; do NOT
conflate the two.

Fill rules
----------
- ``fixed_2r`` (market): fills at the **open** of the next 1-min bar after
  ``on_intent`` is called.
- Limit intents: fill per ``realism.fill_mode``:
    - ``"optimistic"``: bull fills when ``bar.low <= limit``; bear when
      ``bar.high >= limit``.  Fill price = limit.
    - ``"conservative"``: bull fills when ``bar.close <= limit``; bear when
      ``bar.close >= limit``.  Fill price = limit.
- No fill-timeout cap in the live executor (positions stay open until
  TP/SL or ``on_session_close``).

SL/TP intrabar resolution
--------------------------
On each 1-min ``on_bar`` call, after (optionally) filling pending intents,
every open position accumulates the bar.  Resolution is checked by calling
``compute_exit`` over the full accumulated 1-min slice.  When
``outcome.outcome`` is ``"tp"`` or ``"sl"`` the position is resolved via
``FleetState.resolve``.

``on_session_close``
--------------------
All remaining open positions are resolved with ``outcome="session_close"``
and ``r_multiple`` from ``compute_exit`` (outcome will be ``"undecided"``
since bars ran out — we preserve that r_multiple and override the label).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.live.executor_base import Executor
from src.live.fleet_state import (
    FleetState,
    OpenSimPosition,
    SimTrade,
    make_intent_id,
)
from src.live.order_plan import OrderIntent
from src.live.stream import MinuteBar
from src.strategy.exits import ExitConfig, compute_exit

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: pending-fill record
# ---------------------------------------------------------------------------


@dataclass
class _PendingFill:
    """An intent waiting for its first qualifying bar to fill."""

    intent: OrderIntent
    intent_id: str
    window_raw: np.ndarray  # 60×5 — needed for compute_exit after fill


# ---------------------------------------------------------------------------
# SimFillExecutor
# ---------------------------------------------------------------------------


class SimFillExecutor(Executor):
    """Simulated fill executor.  Resolves positions against the 1-min stream.

    Parameters
    ----------
    realism : ExitConfig
        Exit-strategy config carrying ``strategy``, ``fill_mode``, ATR floor,
        slippage, commission, confidence threshold, etc.
    state : FleetState
        Durable state store.  Open positions from previous sessions are
        pre-loaded when the ``FleetState`` is constructed.
    cell_key : str
        ``"ticker:model:strategy"`` label.  Logged alongside every event.
    logger : logging.Logger, optional
        Caller-supplied logger; defaults to the module logger.

    Notes
    -----
    ``on_intent`` requires the ``OrderIntent`` to carry the original
    ``window_raw`` (60×5) array (attached by ``StrategyOrderPlanner``) so that
    ``compute_exit`` can be called once the position is open.  When the window
    is absent the intent is **skipped** rather than resolved against a
    fabricated window (M3) — that would yield a wrong R.

    The ``window_raw`` attached at fill time is the 60-bar H1 window that
    produced the signal.  ``compute_exit`` is then called with the
    accumulated 1-min bars as ``future_ohlcv`` (shape (N, 4), OHLC only).
    This replicates what the H1 replay (WS-G) will do on the same data.
    """

    def __init__(
        self,
        realism: ExitConfig,
        state: FleetState,
        cell_key: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cfg = realism
        self._state = state
        self._cell_key = cell_key
        self._log = logger or _log

        # Pending fills: intent_id -> _PendingFill
        self._pending: dict[str, _PendingFill] = {}

        # Restore pending fills from open positions (cross-session resume).
        # Open positions in FleetState already have accumulated_bars; they do
        # NOT go into _pending (they are already filled).  _pending only holds
        # positions waiting for their first fill bar.
        # NOTE: on reload, previously-filled open positions stay in FleetState
        # and are advanced on each on_bar call below.

    # ------------------------------------------------------------------
    # Executor ABC implementation
    # ------------------------------------------------------------------

    def on_intent(self, intent: OrderIntent) -> None:
        """Accept an ``OrderIntent`` (uniform ``Executor`` contract).

        The 60×5 raw window required for ``compute_exit`` R-calculation is read
        from ``intent.window_raw`` (populated by ``StrategyOrderPlanner``). If
        absent, the intent is **skipped** (logged ``SKIP``) — we never resolve
        an R against a fabricated window, which would produce a wrong number
        (M3).  Callers must plan via ``StrategyOrderPlanner`` so the window is
        always attached.
        """
        if intent.skip_reason is not None:
            self._log.debug(
                "SKIP %s intent_id=%s reason=%s",
                self._cell_key,
                _intent_id(intent),
                intent.skip_reason,
            )
            return
        if intent.window_raw is None:
            self._log.info(
                "SKIP %s intent_id=%s reason=NO_WINDOW (cannot resolve R "
                "without the real decision window)",
                self._cell_key,
                _intent_id(intent),
            )
            return
        self._accept_intent(intent, intent.window_raw)

    def on_bar(self, bar: MinuteBar) -> None:
        """Deliver a 1-minute bar.

        1. Fill any pending market/limit intents against this bar.
        2. Accumulate bar into every open position.
        3. Check each open position for intrabar SL/TP resolution.
        """
        if bar.symbol != self._ticker_from_cell_key():
            return  # not our symbol

        # Step 1: fill pending intents
        filled_ids: list[str] = []
        drop_ids: list[str] = []
        for iid, pf in self._pending.items():
            pos = self._try_fill(pf, bar)
            if pos is _SENTINEL:
                # NaN limit — drop without opening a position
                drop_ids.append(iid)
            elif pos is not None:
                self._state.add_open(pos)
                filled_ids.append(iid)
                self._log.info(
                    "FILL %s intent_id=%s fill=%.4f ts=%s",
                    self._cell_key,
                    iid,
                    pos.fill_price,
                    bar.timestamp,
                )
        for iid in filled_ids + drop_ids:
            del self._pending[iid]

        # Step 2+3: advance open positions
        bar_ohlcv = [bar.open, bar.high, bar.low, bar.close, bar.volume]
        bar_ts_iso = bar.timestamp.isoformat()

        resolved_ids: list[str] = []
        for pos in self._state.open_positions(self._cell_key):
            pos.accumulated_bars.append(
                {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "ts": bar_ts_iso,
                }
            )
            trade = self._check_resolution(pos, bar.timestamp)
            if trade is not None:
                self._state.resolve(trade)
                resolved_ids.append(pos.intent_id)
                self._log.info(
                    "RESOLVE %s intent_id=%s outcome=%s r=%.3f ts=%s",
                    self._cell_key,
                    pos.intent_id,
                    trade.outcome,
                    trade.r_multiple,
                    bar.timestamp,
                )
            else:
                # Persist updated accumulated_bars
                self._state.add_open(pos)

    def on_session_close(self) -> None:
        """Flatten all open and pending positions at session close."""
        # Cancel pending fills
        for iid, pf in self._pending.items():
            self._log.info(
                "SESSION_CLOSE pending cancelled %s intent_id=%s",
                self._cell_key,
                iid,
            )
        self._pending.clear()

        # Resolve open positions as session_close (this cell only)
        for pos in self._state.open_positions(self._cell_key):
            trade = self._force_resolve(pos, outcome_label="session_close")
            self._state.resolve(trade)
            self._log.info(
                "SESSION_CLOSE resolved %s intent_id=%s r=%.3f",
                self._cell_key,
                pos.intent_id,
                trade.r_multiple,
            )

        self._state.persist()

    def open_count(self) -> int:
        """Return number of open positions + pending fills for THIS cell."""
        return len(self._state.open_positions(self._cell_key)) + len(self._pending)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ticker_from_cell_key(self) -> str:
        return self._cell_key.split(":")[0]

    def _accept_intent(self, intent: OrderIntent, window_raw: np.ndarray) -> None:
        iid = _intent_id(intent)
        pf = _PendingFill(intent=intent, intent_id=iid, window_raw=window_raw)
        self._pending[iid] = pf
        self._log.debug(
            "PENDING %s intent_id=%s entry_type=%s",
            self._cell_key,
            iid,
            intent.entry_type,
        )

    def _try_fill(
        self, pf: _PendingFill, bar: MinuteBar
    ) -> Optional[OpenSimPosition]:
        """Attempt to fill a pending intent against a 1-min bar.

        Returns ``OpenSimPosition`` on fill, ``None`` if not yet filled.
        """
        intent = pf.intent

        if intent.entry_type == "market":
            # Market: fill at the open of this bar (next bar after on_intent)
            fill_price = bar.open
        else:
            # Limit: fill per fill_mode
            limit = intent.entry  # limit price from order_plan geometry
            if math.isnan(limit):
                # Degenerate — drop
                self._log.warning(
                    "PENDING_DROP %s intent_id=%s NaN limit",
                    self._cell_key,
                    pf.intent_id,
                )
                return _SENTINEL  # signal: remove but don't open position

            direction = intent.direction
            conservative = self._cfg.fill_mode == "conservative"

            if direction == 1:  # bull — we want to buy at or below limit
                if conservative:
                    filled = bar.close <= limit
                else:
                    filled = bar.low <= limit
            else:  # bear — we want to sell at or above limit
                if conservative:
                    filled = bar.close >= limit
                else:
                    filled = bar.high >= limit

            if not filled:
                return None

            fill_price = limit  # fill at the limit price

        return OpenSimPosition(
            intent_id=pf.intent_id,
            ticker=intent.ticker,
            model=intent.model,
            strategy=intent.strategy,
            direction=intent.direction,
            fill_price=fill_price,
            sl=intent.sl,
            tp=intent.tp,
            fill_ts=bar.timestamp,
            accumulated_bars=[],
            cell_key=self._cell_key,
            window_raw_flat=pf.window_raw.flatten().tolist(),
        )

    def _check_resolution(
        self, pos: OpenSimPosition, bar_ts: pd.Timestamp
    ) -> Optional[SimTrade]:
        """Run ``compute_exit`` over the accumulated slice; return SimTrade if resolved."""
        if not pos.accumulated_bars:
            return None

        future = _bars_to_ohlc(pos.accumulated_bars)

        outcome = compute_exit(
            window_raw=_make_window_raw(pos),
            future_ohlcv=future,
            direction=pos.direction,
            config=self._cfg,
        )

        if outcome.outcome in ("tp", "sl"):
            return SimTrade(
                intent_id=pos.intent_id,
                ticker=pos.ticker,
                model=pos.model,
                strategy=pos.strategy,
                direction=pos.direction,
                fill_price=pos.fill_price,
                sl=pos.sl,
                tp=pos.tp,
                fill_ts=pos.fill_ts,
                close_price=outcome.exit_price,
                close_ts=bar_ts,
                outcome=outcome.outcome,
                r_multiple=outcome.r_multiple,
                cell_key=pos.cell_key,
            )
        return None

    def _force_resolve(
        self, pos: OpenSimPosition, outcome_label: str
    ) -> SimTrade:
        """Force-resolve a position (session close or similar)."""
        r_multiple = 0.0
        exit_price = pos.fill_price

        if pos.accumulated_bars:
            future = _bars_to_ohlc(pos.accumulated_bars)
            try:
                outcome = compute_exit(
                    window_raw=_make_window_raw(pos),
                    future_ohlcv=future,
                    direction=pos.direction,
                    config=self._cfg,
                )
                r_multiple = outcome.r_multiple
                exit_price = outcome.exit_price
            except Exception as exc:
                self._log.warning(
                    "compute_exit failed during force-resolve %s: %s",
                    pos.intent_id,
                    exc,
                )

        return SimTrade(
            intent_id=pos.intent_id,
            ticker=pos.ticker,
            model=pos.model,
            strategy=pos.strategy,
            direction=pos.direction,
            fill_price=pos.fill_price,
            sl=pos.sl,
            tp=pos.tp,
            fill_ts=pos.fill_ts,
            close_price=exit_price,
            close_ts=pd.NaT,
            outcome=outcome_label,
            r_multiple=r_multiple,
            cell_key=pos.cell_key,
        )


# ---------------------------------------------------------------------------
# Sentinel: pending fill to be dropped (NaN limit)
# ---------------------------------------------------------------------------

# A private sentinel OpenSimPosition used as a truthy-but-drop signal from
# _try_fill when the limit is NaN.  The caller checks ``is _SENTINEL``.
_SENTINEL = OpenSimPosition(
    intent_id="__sentinel__",
    ticker="",
    model="",
    strategy="",
    direction=0,
    fill_price=float("nan"),
    sl=float("nan"),
    tp=float("nan"),
    fill_ts=pd.NaT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent_id(intent: OrderIntent) -> str:
    return make_intent_id(
        intent.ticker,
        intent.model,
        intent.strategy,
        intent.h1_timestamp,
    )


def _bars_to_ohlc(bars: list[dict]) -> np.ndarray:
    """Convert accumulated bar dicts to (N, 4) OHLC float32 array."""
    arr = np.array(
        [[b["open"], b["high"], b["low"], b["close"]] for b in bars],
        dtype=np.float32,
    )
    return arr


def _make_window_raw(pos: OpenSimPosition) -> np.ndarray:
    """Return the 60×5 window_raw for this position.

    Uses the stored ``window_raw_flat`` (serialised from the original H1
    window at fill time) when available.  Falls back to a flat approximation
    (all bars = fill_price) when the field is absent — this occurs only for
    positions loaded from state files written before this field was added, or
    for positions created via ``on_intent`` without a window.  The fallback
    produces incorrect ATR-floor geometry; callers should always pass an intent
    carrying ``window_raw`` (the planner populates it) so ``on_intent`` resolves
    R against the real window.
    """
    if pos.window_raw_flat:
        arr = np.array(pos.window_raw_flat, dtype=np.float64)
        if arr.size == 300:
            return arr.reshape(60, 5)

    # Fallback: flat approximation
    price = pos.fill_price if not math.isnan(pos.fill_price) else 400.0
    window = np.full((60, 5), price, dtype=np.float64)
    window[:, 4] = 1.0
    return window
