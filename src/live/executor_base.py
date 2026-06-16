"""executor_base.py — Executor ABC (ISP-thin interface for WS-C).

Defines the contract that all executor implementations (PaperExecutor,
SimFillExecutor) must satisfy.  ``on_fill`` is intentionally NOT on this
ABC — it lives only on ``PaperExecutor`` (the sim has no exchange fills);
keeping it off the base honours the Interface Segregation Principle and
lets downstream WS-D/WS-E import without pulling in trading-client deps.

FillRouter (WS-E) accesses ``on_fill`` by typing against ``PaperExecutor``
directly (or via a ``FillTarget`` protocol — see fill_router.py), never via
this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.live.order_plan import OrderIntent
from src.live.stream import MinuteBar


class Executor(ABC):
    """Common interface for all executor implementations.

    Method contracts
    ----------------
    on_intent(intent)
        Receive a fully-resolved ``OrderIntent``.  Implementations MUST
        silently drop intents where ``intent.skip_reason`` is not ``None``
        (log them, but never submit an order).

    on_bar(bar)
        Deliver the latest 1-minute bar.  Used by ``SimFillExecutor`` for
        intrabar SL/TP resolution; ``PaperExecutor`` uses it for
        limit-timeout tracking.

    on_session_close()
        End-of-RTH cleanup.  Implementations must cancel open orders and
        flatten positions (real or simulated).

    open_count()
        Return the number of currently open positions / pending intents
        managed by this executor.  Used by ``FillRouter`` for the
        fleet-level exposure cap.
    """

    @abstractmethod
    def on_intent(self, intent: OrderIntent) -> None:
        """Process a new ``OrderIntent``."""

    @abstractmethod
    def on_bar(self, bar: MinuteBar) -> None:
        """Deliver a 1-minute bar to the executor."""

    @abstractmethod
    def on_session_close(self) -> None:
        """End-of-session cleanup: cancel orders, flatten positions."""

    @abstractmethod
    def open_count(self) -> int:
        """Return the number of currently open positions."""
