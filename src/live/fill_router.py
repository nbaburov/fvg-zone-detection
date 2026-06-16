"""fill_router.py — FillRouter: symbol-keyed fill dispatch + fleet exposure cap.

``FillRouter`` holds a ``{ticker -> PaperExecutor}`` registry.  Its two
responsibilities are:

1. **Fill dispatch** — route an incoming ``FillEvent`` to the single executor
   that owns the corresponding symbol.  The ``unique-ticker guard`` ensures
   exactly one executor owns each ticker; a ``ValueError`` is raised at
   registration time if a duplicate is attempted.

2. **Fleet capital guard** — ``can_enter(ticker, risk_amount, equity)`` returns
   ``True`` only when adding ``risk_amount`` would *not* push the total open
   real risk above ``max_real_exposure * equity``.  Sim-only cells (no
   ``PaperExecutor``) are exempt — they are never counted.

Usage
-----
::

    router = FillRouter(max_real_exposure=0.04)
    router.register("SPY", spy_executor)
    router.register("QQQ", qqq_executor)

    # Route a fill arriving from the Alpaca trading stream
    router.on_fill(fill_event)

    # Guard before submitting a new real entry
    if router.can_enter("SPY", risk_amount=50.0, equity=10_000.0):
        executor.on_intent(intent)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.live.execution import PaperExecutor
    from src.live.logger import FillEvent

logger = logging.getLogger(__name__)


class FillRouter:
    """Routes ``FillEvent`` objects to the owning ``PaperExecutor`` by symbol.

    Parameters
    ----------
    max_real_exposure : float
        Fleet-level cap on summed open real risk as a fraction of equity.
        ``can_enter`` blocks a new entry when the cap would be breached.
        Default ``0.04`` (4 %).

    Notes
    -----
    *Unique-ticker invariant* — each symbol may be registered **at most once**.
    Attempting to register a second executor for the same ticker raises
    ``ValueError`` immediately, before any connection is opened.

    *Sim cells are exempt* — ``FillRouter`` is only used for real
    (``PaperExecutor``) cells.  Sim cells (``SimFillExecutor``) handle their
    own internal position tracking and never call ``on_fill`` on this router.
    """

    def __init__(self, max_real_exposure: float = 0.04) -> None:
        if not (0.0 < max_real_exposure <= 1.0):
            raise ValueError(
                f"max_real_exposure must be in (0, 1]; got {max_real_exposure}"
            )
        self._max_real_exposure = max_real_exposure
        # ticker -> PaperExecutor (unique-ticker invariant enforced in register())
        self._executors: dict[str, "PaperExecutor"] = {}
        # ticker -> per-trade risk amount for currently open positions
        # (tracked externally via can_enter / on_fill)
        self._open_risk: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, ticker: str, executor: "PaperExecutor") -> None:
        """Register a ``PaperExecutor`` as the sole owner of ``ticker``.

        Parameters
        ----------
        ticker : str
            Market symbol (e.g. ``"SPY"``).
        executor : PaperExecutor
            Executor that manages real bracket orders for this symbol.

        Raises
        ------
        ValueError
            If ``ticker`` has already been registered.
        """
        ticker = ticker.upper()
        if ticker in self._executors:
            raise ValueError(
                f"FillRouter: ticker '{ticker}' already registered. "
                "Each ticker must have exactly one owning PaperExecutor."
            )
        self._executors[ticker] = executor
        logger.debug("FillRouter: registered executor for %s", ticker)

    # ------------------------------------------------------------------
    # Fill dispatch
    # ------------------------------------------------------------------

    def on_fill(self, fill: "FillEvent") -> None:
        """Dispatch ``fill`` to the executor that owns ``fill.order_id``'s symbol.

        The router resolves the symbol by inspecting the executor whose
        ``_symbol`` attribute matches the fill.  Since fills arrive from
        Alpaca's trading stream and do *not* carry a symbol field directly,
        ``FillRouter`` iterates the registry to find the owner.

        In practice the match is O(n) over the number of registered tickers
        (typically ≤ 10), which is negligible.

        If no executor claims the fill (e.g. a fill for an order not in this
        session), a warning is logged and the fill is dropped.

        Parameters
        ----------
        fill : FillEvent
            The fill notification from Alpaca's trading stream.
        """
        # Prefer an executor whose symbol matches the fill's symbol attribute
        # when available (FillEvent may not have a symbol field — handle both).
        fill_symbol: str | None = getattr(fill, "symbol", None)

        if fill_symbol is not None:
            fill_symbol = fill_symbol.upper()
            executor = self._executors.get(fill_symbol)
            if executor is not None:
                logger.debug(
                    "FillRouter: routing fill order_id=%s symbol=%s",
                    fill.order_id,
                    fill_symbol,
                )
                executor.on_fill(fill)
                # Release open-risk bucket on a closing fill
                if fill.fill_type == "fill":
                    self._open_risk.pop(fill_symbol, None)
                return

        # Fallback: no symbol on the FillEvent — log and drop.
        logger.warning(
            "FillRouter: could not route fill order_id=%s (symbol=%s not registered). "
            "Dropping.",
            fill.order_id,
            fill_symbol,
        )

    # ------------------------------------------------------------------
    # Fleet exposure cap
    # ------------------------------------------------------------------

    def can_enter(
        self,
        ticker: str,
        risk_amount: float,
        equity: float,
    ) -> bool:
        """Return ``True`` if adding ``risk_amount`` stays within the cap.

        Checks whether the *current* total open real risk plus ``risk_amount``
        would exceed ``max_real_exposure * equity``.  If ``risk_amount`` alone
        already exceeds the cap, returns ``False``.

        Parameters
        ----------
        ticker : str
            Symbol requesting entry (used for logging only).
        risk_amount : float
            Dollar risk for the proposed new position
            (typically ``sl_distance * qty``).
        equity : float
            Current account portfolio value in dollars.

        Returns
        -------
        bool
            ``True`` → entry is permitted; ``False`` → entry is blocked.
        """
        if equity <= 0:
            logger.warning("FillRouter.can_enter: equity <= 0, blocking entry for %s", ticker)
            return False

        current_open_risk = sum(self._open_risk.values())
        cap = self._max_real_exposure * equity

        if current_open_risk + risk_amount > cap:
            logger.info(
                "FillRouter.can_enter: BLOCKED %s — open_risk=%.2f + proposed=%.2f > cap=%.2f (%.1f%% of %.2f)",
                ticker,
                current_open_risk,
                risk_amount,
                cap,
                self._max_real_exposure * 100,
                equity,
            )
            return False

        return True

    def record_entry(self, ticker: str, risk_amount: float) -> None:
        """Record that a new real entry has been committed for ``ticker``.

        Must be called immediately after an order is successfully submitted so
        that subsequent ``can_enter`` calls see the updated open-risk total.

        Parameters
        ----------
        ticker : str
            Symbol that just entered.
        risk_amount : float
            Dollar risk committed (``sl_distance * qty``).
        """
        ticker = ticker.upper()
        self._open_risk[ticker] = self._open_risk.get(ticker, 0.0) + risk_amount
        logger.debug(
            "FillRouter.record_entry: %s += %.2f → total_open_risk=%.2f",
            ticker,
            risk_amount,
            sum(self._open_risk.values()),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def registered_tickers(self) -> list[str]:
        """Sorted list of registered tickers."""
        return sorted(self._executors.keys())

    def total_open_risk(self) -> float:
        """Sum of all open real risk amounts tracked by this router."""
        return sum(self._open_risk.values())
