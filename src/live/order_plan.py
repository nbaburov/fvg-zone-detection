"""order_plan.py — OrderIntent dataclass + StrategyOrderPlanner.

§0 keystone: ALL SL/TP geometry flows through
``src.strategy.exits.compute_exit``.  No exit math is duplicated here.

Usage
-----
::

    from src.strategy.exits import ExitConfig
    from src.live.order_plan import StrategyOrderPlanner

    config = ExitConfig(strategy="ict_iofed")
    planner = StrategyOrderPlanner(config)
    intent = planner.plan(
        window_raw,
        direction=1,
        h1_timestamp=ts,
        ticker="SPY",
        model="cnn_lstm",
        signal="bull",
        confidence=0.72,
    )

The returned ``OrderIntent`` is ready to be forwarded to an
``Executor.on_intent()`` call.  When window geometry is degenerate (compute_exit
returns NaN entry), ``intent.skip_reason="DEGENERATE_GEOMETRY"``; the
executor must ignore such intents (and should log them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.strategy.exits import ExitConfig, TradeOutcome, compute_exit

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrderIntent:
    """A fully-resolved order plan for one (ticker, model, strategy) cell.

    All price fields come directly from ``compute_exit``; no arithmetic
    is repeated here.

    Parameters
    ----------
    ticker : str
        Market symbol (e.g. "SPY", "QQQ").
    model : str
        Model identifier (e.g. "lstm", "cnn_lstm", "xgboost").
    strategy : str
        Exit-strategy name, one of ``{"fixed_2r","ict_iofed","ce_50pct",
        "tradinglab"}``.
    h1_timestamp : pd.Timestamp
        Timestamp of the H1 bar that triggered the signal.
    direction : int
        1 = bullish FVG, 2 = bearish FVG.
    signal : str
        Raw signal string from ``TradeAction``, one of
        ``{"bull", "bear", "none"}``.
    confidence : float
        Max softmax probability among bull/bear classes.
    entry : float
        Proposed entry price; ``nan`` when geometry is degenerate.
    sl : float
        Stop-loss price.
    tp : float
        Take-profit price; ``nan`` when geometry is degenerate.
    entry_type : Literal["market", "limit"]
        ``"market"`` for ``fixed_2r``; ``"limit"`` for all other strategies.
    skip_reason : Optional[str]
        ``None`` for valid intents.  Set to ``"DEGENERATE_GEOMETRY"`` when
        ``compute_exit`` returns a NaN entry — executors must drop such
        intents.
    """

    ticker: str
    model: str
    strategy: str
    h1_timestamp: pd.Timestamp
    direction: int
    signal: str
    confidence: float
    entry: float
    sl: float
    tp: float
    entry_type: Literal["market", "limit"]
    skip_reason: Optional[str]
    # The 60×5 raw OHLCV window that produced this intent. Carried so a
    # SimFillExecutor can call compute_exit for R at resolution time while the
    # Executor.on_intent(intent) contract stays uniform (LSP). Excluded from
    # equality/repr; never serialised to the intent jsonl.
    window_raw: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class StrategyOrderPlanner:
    """Maps a live window to an ``OrderIntent`` via ``compute_exit``.

    Single responsibility: translate a model signal + raw OHLCV window into
    a fully-formed order plan.  Depends *only* on
    ``src.strategy.exits.compute_exit`` for all price geometry.

    Parameters
    ----------
    config : ExitConfig
        Exit-strategy configuration (strategy name + realism params).
    """

    def __init__(self, config: ExitConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        window_raw: np.ndarray,
        direction: int,
        h1_timestamp: pd.Timestamp,
        *,
        ticker: str,
        model: str,
        signal: str,
        confidence: float,
    ) -> OrderIntent:
        """Compute an ``OrderIntent`` for the given window and signal.

        Calls ``compute_exit`` with an empty future array (shape ``(0, 4)``)
        so that only entry/SL/TP geometry is extracted — no fill simulation
        is performed.  ``outcome`` will be ``"no_future"`` and is discarded.

        Parameters
        ----------
        window_raw : np.ndarray, shape (60, 5)
            Un-normalised OHLCV window (columns: open, high, low, close,
            volume).
        direction : int
            1 = bullish, 2 = bearish.
        h1_timestamp : pd.Timestamp
            Timestamp of the triggering H1 bar close.
        ticker : str
            Market symbol.
        model : str
            Model identifier.
        signal : str
            ``"bull"``, ``"bear"``, or ``"none"`` (from ``TradeAction``).
        confidence : float
            Model confidence score.

        Returns
        -------
        OrderIntent
            Fully populated intent.  ``skip_reason="DEGENERATE_GEOMETRY"``
            when the window produces a NaN entry from ``compute_exit``.
        """
        empty_future = np.empty((0, 4), dtype=np.float32)

        outcome: TradeOutcome = compute_exit(
            window_raw,
            empty_future,
            direction,
            self._config,
        )

        entry_type: Literal["market", "limit"] = (
            "market" if self._config.strategy == "fixed_2r" else "limit"
        )

        # Detect degenerate geometry.
        #
        # For ``fixed_2r`` the entry is always NaN with an empty future (the
        # entry is the N+2 bar open, which is only known at fill time).  A NaN
        # entry is therefore *expected* for fixed_2r and does NOT indicate a
        # broken window — the executor will obtain the real entry from the
        # market fill.  Geometry is degenerate only when the SL is also NaN,
        # which happens when bar_56 coordinates are themselves NaN.
        #
        # For limit-entry variants the limit price is derived directly from
        # bar_58 coordinates: a NaN entry means the window bars contain NaN
        # and geometry cannot be trusted.
        skip_reason: Optional[str] = None
        sl_nan = math.isnan(outcome.sl)
        entry_nan = math.isnan(outcome.entry)
        is_fixed_2r = self._config.strategy == "fixed_2r"

        if sl_nan or (entry_nan and not is_fixed_2r):
            skip_reason = "DEGENERATE_GEOMETRY"

        return OrderIntent(
            ticker=ticker,
            model=model,
            strategy=self._config.strategy,
            h1_timestamp=h1_timestamp,
            direction=direction,
            signal=signal,
            confidence=confidence,
            entry=outcome.entry,
            sl=outcome.sl,
            tp=outcome.tp,
            entry_type=entry_type,
            skip_reason=skip_reason,
            window_raw=window_raw,
        )
