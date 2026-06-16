"""Tests for FillRouter — symbol routing + fleet exposure cap.

Uses a tiny ``FakeExecutor`` stub instead of a real ``PaperExecutor`` so no
Alpaca TradingClient is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pytest

from src.live.fill_router import FillRouter
from src.live.logger import FillEvent


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeExecutor:
    """Minimal PaperExecutor stub exposing symbol + on_fill."""

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self.fills_received: list[FillEvent] = []

    def on_fill(self, fill: FillEvent) -> None:
        self.fills_received.append(fill)

    def open_count(self) -> int:
        return 0


def _make_fill(symbol: str, fill_type: str = "fill", order_id: str = "ord-001") -> FillEvent:
    fill = FillEvent(
        order_id=order_id,
        filled_at=pd.Timestamp("2024-01-02 10:31:00", tz="America/New_York"),
        fill_price=452.30,
        fill_qty=5,
        fill_type=fill_type,
        side="buy",
    )
    # Attach symbol (FillEvent doesn't have it by default; FillRouter uses getattr)
    fill.symbol = symbol  # type: ignore[attr-defined]
    return fill


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_single_ticker(self):
        router = FillRouter()
        ex = FakeExecutor("SPY")
        router.register("SPY", ex)
        assert "SPY" in router.registered_tickers

    def test_duplicate_ticker_raises(self):
        router = FillRouter()
        router.register("SPY", FakeExecutor("SPY"))
        with pytest.raises(ValueError, match="already registered"):
            router.register("SPY", FakeExecutor("SPY"))

    def test_case_normalised(self):
        router = FillRouter()
        router.register("spy", FakeExecutor("spy"))
        assert "SPY" in router.registered_tickers

    def test_multiple_tickers_distinct(self):
        router = FillRouter()
        router.register("SPY", FakeExecutor("SPY"))
        router.register("QQQ", FakeExecutor("QQQ"))
        assert set(router.registered_tickers) == {"SPY", "QQQ"}

    def test_invalid_max_real_exposure_raises(self):
        with pytest.raises(ValueError):
            FillRouter(max_real_exposure=0.0)
        with pytest.raises(ValueError):
            FillRouter(max_real_exposure=1.5)


# ---------------------------------------------------------------------------
# Fill routing
# ---------------------------------------------------------------------------

class TestFillRouting:
    def test_fill_routed_to_correct_executor(self):
        router = FillRouter()
        spy_ex = FakeExecutor("SPY")
        qqq_ex = FakeExecutor("QQQ")
        router.register("SPY", spy_ex)
        router.register("QQQ", qqq_ex)

        fill = _make_fill("SPY")
        router.on_fill(fill)

        assert len(spy_ex.fills_received) == 1
        assert len(qqq_ex.fills_received) == 0

    def test_fill_for_different_symbol_goes_to_correct_executor(self):
        router = FillRouter()
        spy_ex = FakeExecutor("SPY")
        qqq_ex = FakeExecutor("QQQ")
        router.register("SPY", spy_ex)
        router.register("QQQ", qqq_ex)

        router.on_fill(_make_fill("QQQ"))

        assert len(qqq_ex.fills_received) == 1
        assert len(spy_ex.fills_received) == 0

    def test_unknown_symbol_does_not_raise(self):
        """A fill for an unregistered symbol is warned and dropped, not raised."""
        router = FillRouter()
        router.register("SPY", FakeExecutor("SPY"))
        # IWM is not registered — should silently drop
        router.on_fill(_make_fill("IWM"))  # no error

    def test_fill_without_symbol_attr_does_not_raise(self):
        """FillEvent with no symbol attribute is dropped gracefully."""
        router = FillRouter()
        router.register("SPY", FakeExecutor("SPY"))
        fill = FillEvent(
            order_id="no-sym",
            filled_at=pd.Timestamp.now(tz="America/New_York"),
            fill_price=100.0,
            fill_qty=1,
            fill_type="fill",
            side="buy",
        )
        # No .symbol attribute → should warn + drop
        router.on_fill(fill)

    def test_closing_fill_releases_open_risk(self):
        router = FillRouter()
        router.register("SPY", FakeExecutor("SPY"))
        router.record_entry("SPY", 100.0)
        assert router.total_open_risk() == pytest.approx(100.0)

        fill = _make_fill("SPY", fill_type="fill")
        router.on_fill(fill)
        assert router.total_open_risk() == pytest.approx(0.0)

    def test_partial_fill_does_not_release_risk(self):
        router = FillRouter()
        router.register("SPY", FakeExecutor("SPY"))
        router.record_entry("SPY", 100.0)

        fill = _make_fill("SPY", fill_type="partial_fill")
        router.on_fill(fill)
        # partial fill → risk stays
        assert router.total_open_risk() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Fleet exposure cap
# ---------------------------------------------------------------------------

class TestExposureCap:
    def test_can_enter_when_below_cap(self):
        router = FillRouter(max_real_exposure=0.04)
        # 0 open risk + 300 proposed vs cap = 0.04 * 10_000 = 400 → allowed
        assert router.can_enter("SPY", risk_amount=300.0, equity=10_000.0) is True

    def test_blocked_when_proposed_exceeds_cap(self):
        router = FillRouter(max_real_exposure=0.04)
        # 0 + 500 vs cap 400 → blocked
        assert router.can_enter("SPY", risk_amount=500.0, equity=10_000.0) is False

    def test_blocked_when_open_plus_proposed_exceeds_cap(self):
        router = FillRouter(max_real_exposure=0.04)
        router.register("SPY", FakeExecutor("SPY"))
        router.record_entry("SPY", 350.0)  # 350 already open
        # 350 + 100 = 450 > 400 → blocked
        assert router.can_enter("QQQ", risk_amount=100.0, equity=10_000.0) is False

    def test_allowed_when_total_exactly_equals_cap(self):
        router = FillRouter(max_real_exposure=0.04)
        router.register("SPY", FakeExecutor("SPY"))
        router.record_entry("SPY", 300.0)
        # 300 + 100 = 400 == cap → allowed (not strictly greater)
        assert router.can_enter("QQQ", risk_amount=100.0, equity=10_000.0) is True

    def test_zero_equity_blocks_entry(self):
        router = FillRouter(max_real_exposure=0.04)
        assert router.can_enter("SPY", risk_amount=1.0, equity=0.0) is False

    def test_record_entry_accumulates(self):
        router = FillRouter(max_real_exposure=0.10)
        router.register("SPY", FakeExecutor("SPY"))
        router.register("QQQ", FakeExecutor("QQQ"))
        router.record_entry("SPY", 200.0)
        router.record_entry("QQQ", 150.0)
        assert router.total_open_risk() == pytest.approx(350.0)

    def test_total_open_risk_starts_zero(self):
        router = FillRouter()
        assert router.total_open_risk() == pytest.approx(0.0)
