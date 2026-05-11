"""slippage.py — 1-tick slippage overlay for P&L accounting.

Adjusts reported P&L to account for realistic entry/exit cost.
Does NOT modify actual Alpaca order prices.
"""

from __future__ import annotations


class SlippageModel:
    """Simple 1-tick slippage + fixed commission model.

    Parameters
    ----------
    ticks : int
        Number of ticks of slippage on each entry or exit leg. Default 1.
    tick_size : float
        Dollar value of one tick. Default $0.01 (SPY minimum increment).
    commission_per_share : float
        Per-share commission. Default $0.005.
    """

    def __init__(
        self,
        ticks: int = 1,
        tick_size: float = 0.01,
        commission_per_share: float = 0.005,
    ) -> None:
        self._ticks = ticks
        self._tick_size = tick_size
        self._commission_per_share = commission_per_share

    def adjusted_entry(self, fill_price: float, side: str) -> float:
        """Return fill_price adjusted for entry slippage.

        Buys slip up (worse price), sells slip down.
        """
        slippage = self._ticks * self._tick_size
        if side == "buy":
            return fill_price + slippage
        return fill_price - slippage

    def commission(self, qty: int) -> float:
        """Return round-trip commission for qty shares."""
        return qty * self._commission_per_share * 2  # entry + exit

    def adjusted_pnl(
        self,
        entry_fill: float,
        exit_fill: float,
        qty: int,
        side: str,
    ) -> float:
        """Compute P&L with slippage on both legs + round-trip commission.

        Parameters
        ----------
        entry_fill : float
            Actual fill price on entry.
        exit_fill : float
            Actual fill price on exit (TP or SL trigger).
        qty : int
            Number of shares.
        side : str
            "buy" for long, "sell" for short.
        """
        adj_entry = self.adjusted_entry(entry_fill, side)
        # Exit slippage is adverse: buys exit at lower fill (slip down), sells exit at higher (slip up)
        exit_side = "sell" if side == "buy" else "buy"
        adj_exit = self.adjusted_entry(exit_fill, exit_side)

        if side == "buy":
            raw_pnl = (adj_exit - adj_entry) * qty
        else:
            raw_pnl = (adj_entry - adj_exit) * qty

        return raw_pnl - self.commission(qty)
