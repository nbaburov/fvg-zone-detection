"""Unit tests for SlippageModel."""

from __future__ import annotations

import pytest

from src.live.slippage import SlippageModel


class TestSlippageModel:
    def test_adjusted_entry_buy_slips_up(self):
        model = SlippageModel(ticks=1, tick_size=0.01)
        assert model.adjusted_entry(100.0, "buy") == pytest.approx(100.01)

    def test_adjusted_entry_sell_slips_down(self):
        model = SlippageModel(ticks=1, tick_size=0.01)
        assert model.adjusted_entry(100.0, "sell") == pytest.approx(99.99)

    def test_adjusted_entry_two_ticks(self):
        model = SlippageModel(ticks=2, tick_size=0.01)
        assert model.adjusted_entry(100.0, "buy") == pytest.approx(100.02)

    def test_commission_round_trip(self):
        model = SlippageModel(commission_per_share=0.005)
        # 10 shares, round-trip = 10 * 0.005 * 2 = 0.10
        assert model.commission(10) == pytest.approx(0.10)

    def test_adjusted_pnl_long_profitable(self):
        """Long trade: buy at 100, exit at 102, 10 shares, 1-tick slippage each leg."""
        model = SlippageModel(ticks=1, tick_size=0.01, commission_per_share=0.005)
        # adj_entry = 100.01 (buy slips up)
        # adj_exit = 101.99 (sell slips down)
        # raw_pnl = (101.99 - 100.01) * 10 = 19.80
        # commission = 10 * 0.005 * 2 = 0.10
        # net = 19.70
        pnl = model.adjusted_pnl(100.0, 102.0, 10, "buy")
        assert pnl == pytest.approx(19.70)

    def test_adjusted_pnl_short_profitable(self):
        """Short trade: sell at 100, buy back at 98, 10 shares."""
        model = SlippageModel(ticks=1, tick_size=0.01, commission_per_share=0.005)
        # adj_entry (sell) = 99.99 (sell slips down)
        # adj_exit (buy back) = 98.01 (buy slips up)
        # raw_pnl = (99.99 - 98.01) * 10 = 19.80
        # commission = 10 * 0.005 * 2 = 0.10
        # net = 19.70
        pnl = model.adjusted_pnl(100.0, 98.0, 10, "sell")
        assert pnl == pytest.approx(19.70)

    def test_adjusted_pnl_long_losing(self):
        """Long trade that loses money."""
        model = SlippageModel(ticks=1, tick_size=0.01, commission_per_share=0.005)
        pnl = model.adjusted_pnl(100.0, 99.0, 10, "buy")
        assert pnl < 0
