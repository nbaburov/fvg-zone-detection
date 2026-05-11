"""ValidFVGLabeller — TradingLab 6-criteria valid FVG, vectorised, no row-loops."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from src.data.labels import register
from src.data.labels.base import BaseLabeller
from src.data.labels.bos import compute_bos
from src.data.labels.sr import compute_pivot_levels


@register("fvg_valid")
class ValidFVGLabeller(BaseLabeller):
    """
    Valid FVG labeller implementing TradingLab's 6-criteria rule.

    Criteria applied (Criterion #1 — unmitigated — is inference-only, not here):
      #1  Geometric FVG exists (3-candle pattern with gap + body).
      #2  Reaction candle (N+2) closes inside the gap zone (strict).
      #3  An S/R pivot level coincides with the FVG zone (ATR-expanded).
      #4  Label all passing candidates (priority feature, no elimination).
      #5  Bull FVG bottom ≤ 50-bar swing midpoint; bear FVG top ≥ midpoint.
      #6  A BOS occurred within the past bos_lookback bars before the FVG.

    Label index: N+2 (reaction candle — first bar where all criteria are knowable).
    Output: ternary {0=none, 1=bullish, -1=bearish}. No NaN. First 2 and last 2 rows = 0.

    Vectorised pandas/numpy throughout — no per-bar Python loops.
    """

    # ClassVar — inherited by all instances, matches BaseLabeller contract
    label_index_offset: ClassVar[int] = 2
    num_classes: ClassVar[int] = 3
    class_names: ClassVar[list[str]] = ["none", "bullish", "bearish"]
    encoded_map: ClassVar[dict[int, int]] = {0: 0, 1: 1, -1: 2}

    def __init__(
        self,
        swing_lookback: int = 50,
        bos_lookback: int = 50,
        sr_lookback: int = 5,
        atr_period: int = 14,
        confluence_atr_mult: float = 0.5,
        require_crit3: bool = False,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.bos_lookback = bos_lookback
        self.sr_lookback = sr_lookback
        self.atr_period = atr_period
        self.confluence_atr_mult = confluence_atr_mult
        self.require_crit3 = require_crit3

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Input:  DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
        Output: Series[int] same index as df. Values: {-1, 0, 1}. No NaN.
        """
        labels, _ = self._compute(df, return_ablation=False)
        return labels

    def label_with_ablation(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Same as label() but also returns the ablation DataFrame.

        ablation_df columns (all bool):
            geometric_bull, geometric_bear,
            crit2_strict_bull, crit2_strict_bear,
            crit2_loose_bull, crit2_loose_bear,
            crit3_sr_bull, crit3_sr_bear,
            crit5_gann_bull, crit5_gann_bear,
            crit6_bos_bull, crit6_bos_bear,
            valid_bull, valid_bear.
        All columns aligned to df.index. One row per candle.
        """
        return self._compute(df, return_ablation=True)

    # ------------------------------------------------------------------
    # Internal vectorised pipeline
    # ------------------------------------------------------------------

    def _compute(
        self, df: pd.DataFrame, return_ablation: bool
    ) -> tuple[pd.Series, pd.DataFrame]:
        n = len(df)
        empty_label = pd.Series(np.zeros(n, dtype=int), index=df.index)

        if n < 5:
            # Need at minimum bars for i-1, i, i+1, i+2 with i≥1 → n≥4, use 5 for safety
            empty_ablation = self._empty_ablation(df.index) if return_ablation else pd.DataFrame()
            return empty_label, empty_ablation

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        open_ = df["open"].to_numpy()
        close = df["close"].to_numpy()

        # i = middle candle of 3-candle pattern; ranges 1..n-3 (need i+2 < n)
        i = np.arange(1, n - 2)  # length n-3

        # ------------------------------------------------------------------
        # Criterion #1 — geometric FVG
        # ------------------------------------------------------------------
        # Bullish FVG: gap between bar i-1 high and bar i+1 low, body up
        bull_gap = high[i - 1] < low[i + 1]
        bull_body = close[i] > open_[i]
        geom_bull_i = bull_gap & bull_body

        # Bearish FVG: gap between bar i-1 low and bar i+1 high, body down
        bear_gap = low[i - 1] > high[i + 1]
        bear_body = close[i] < open_[i]
        geom_bear_i = bear_gap & bear_body

        # FVG zone edges (at pattern middle bar i)
        # Bull: bottom = high[i-1], top = low[i+1]
        # Bear: top = low[i-1], bottom = high[i+1]
        bull_fvg_bottom_i = high[i - 1]   # lower edge of bullish gap
        bull_fvg_top_i = low[i + 1]       # upper edge of bullish gap
        bear_fvg_bottom_i = high[i + 1]   # lower edge of bearish gap
        bear_fvg_top_i = low[i - 1]       # upper edge of bearish gap

        # ------------------------------------------------------------------
        # ATR proxy (simplified, no pandas_ta): |close diff|.rolling.mean
        # Use shift(1) so bar i uses only prior bars.
        # ------------------------------------------------------------------
        atr_series = df["close"].diff().abs().rolling(self.atr_period, min_periods=1).mean().shift(1)
        atr = atr_series.to_numpy()  # atr[i] is causal for bar i

        # ------------------------------------------------------------------
        # S/R pivot levels (causal) — for criterion #3
        # ------------------------------------------------------------------
        pivot_high, pivot_low = compute_pivot_levels(df, lookback=self.sr_lookback)
        ph = pivot_high.to_numpy()  # ph[i] = max of past sr_lookback highs before i
        pl = pivot_low.to_numpy()

        # ------------------------------------------------------------------
        # Swing high/low for Gann (criterion #5) — causal
        # ------------------------------------------------------------------
        swing_high_series = df["high"].rolling(self.swing_lookback, min_periods=1).max().shift(1)
        swing_low_series = df["low"].rolling(self.swing_lookback, min_periods=1).min().shift(1)
        sh = swing_high_series.to_numpy()
        sl = swing_low_series.to_numpy()

        # ------------------------------------------------------------------
        # BOS (criterion #6)
        # ------------------------------------------------------------------
        bull_bos_recent, bear_bos_recent = compute_bos(
            df, swing_lookback=self.swing_lookback, bos_lookback=self.bos_lookback
        )
        bull_bos_arr = bull_bos_recent.to_numpy()
        bear_bos_arr = bear_bos_recent.to_numpy()

        # ------------------------------------------------------------------
        # Per-middle-candle criterion masks (indexed on i vector)
        # ------------------------------------------------------------------

        # Criterion #2 — reaction candle (i+2) closes inside gap (strict)
        react_close_i = close[i + 2]

        crit2_strict_bull_i = (react_close_i >= bull_fvg_bottom_i) & (react_close_i <= bull_fvg_top_i)
        crit2_strict_bear_i = (react_close_i >= bear_fvg_bottom_i) & (react_close_i <= bear_fvg_top_i)

        # Loose ablation variant (not used in label)
        crit2_loose_bull_i = react_close_i >= bull_fvg_bottom_i
        crit2_loose_bear_i = react_close_i <= bear_fvg_top_i

        # Criterion #3 — S/R confluence at bar i (FVG formation bar)
        tol_i = self.confluence_atr_mult * atr[i]

        # Bull: S/R level (ph or pl at bar i) falls within expanded bull zone
        bull_zone_lo = bull_fvg_bottom_i - tol_i
        bull_zone_hi = bull_fvg_top_i + tol_i
        crit3_sr_bull_i = (
            ((ph[i] >= bull_zone_lo) & (ph[i] <= bull_zone_hi)) |
            ((pl[i] >= bull_zone_lo) & (pl[i] <= bull_zone_hi))
        )
        # Bear: S/R level falls within expanded bear zone
        bear_zone_lo = bear_fvg_bottom_i - tol_i
        bear_zone_hi = bear_fvg_top_i + tol_i
        crit3_sr_bear_i = (
            ((ph[i] >= bear_zone_lo) & (ph[i] <= bear_zone_hi)) |
            ((pl[i] >= bear_zone_lo) & (pl[i] <= bear_zone_hi))
        )
        # NaN pivot values → criterion fails
        crit3_sr_bull_i = np.where(np.isnan(ph[i]) & np.isnan(pl[i]), False, crit3_sr_bull_i)
        crit3_sr_bear_i = np.where(np.isnan(ph[i]) & np.isnan(pl[i]), False, crit3_sr_bear_i)

        # Criterion #5 — Gann box position at bar i
        swing_range_i = sh[i] - sl[i]
        gann_valid_range = swing_range_i >= 0.01  # guard: range collapse
        swing_mid_i = sl[i] + 0.5 * swing_range_i

        # Bull: FVG bottom must be at or below swing midpoint
        crit5_gann_bull_i = (bull_fvg_bottom_i <= swing_mid_i) & gann_valid_range
        # Bear: FVG top must be at or above swing midpoint
        crit5_gann_bear_i = (bear_fvg_top_i >= swing_mid_i) & gann_valid_range

        # Criterion #6 — recent BOS at bar i (BOS must precede the FVG middle candle)
        crit6_bos_bull_i = bull_bos_arr[i]
        crit6_bos_bear_i = bear_bos_arr[i]

        # ------------------------------------------------------------------
        # Combine: all criteria must pass
        # ------------------------------------------------------------------
        sr_bull = crit3_sr_bull_i if self.require_crit3 else np.ones_like(crit3_sr_bull_i, dtype=bool)
        sr_bear = crit3_sr_bear_i if self.require_crit3 else np.ones_like(crit3_sr_bear_i, dtype=bool)
        valid_bull_i = (
            geom_bull_i &
            crit2_loose_bull_i &
            sr_bull &
            crit5_gann_bull_i &
            crit6_bos_bull_i
        )
        valid_bear_i = (
            geom_bear_i &
            crit2_loose_bear_i &
            sr_bear &
            crit5_gann_bear_i &
            crit6_bos_bear_i
        )

        # ------------------------------------------------------------------
        # Assemble label at N+2 (label position = i + 2)
        # ------------------------------------------------------------------
        raw = np.zeros(n, dtype=np.int8)
        lpos = i + 2  # label index for each middle-candle i

        # Bear first, then bull overwrites on conflict (bull wins)
        raw[lpos[valid_bear_i]] = -1
        raw[lpos[valid_bull_i]] = 1

        # Boundary: first 2 and last 2 must be 0
        raw[:2] = 0
        raw[-2:] = 0

        label_series = pd.Series(raw.astype(int), index=df.index)

        # ------------------------------------------------------------------
        # Ablation DataFrame (aligned to df.index — one row per candle)
        # ------------------------------------------------------------------
        if return_ablation:
            ablation = self._build_ablation(
                df_index=df.index,
                i=i,
                n=n,
                geom_bull_i=geom_bull_i,
                geom_bear_i=geom_bear_i,
                crit2_strict_bull_i=crit2_strict_bull_i,
                crit2_strict_bear_i=crit2_strict_bear_i,
                crit2_loose_bull_i=crit2_loose_bull_i,
                crit2_loose_bear_i=crit2_loose_bear_i,
                crit3_sr_bull_i=crit3_sr_bull_i,
                crit3_sr_bear_i=crit3_sr_bear_i,
                crit5_gann_bull_i=crit5_gann_bull_i,
                crit5_gann_bear_i=crit5_gann_bear_i,
                crit6_bos_bull_i=crit6_bos_bull_i,
                crit6_bos_bear_i=crit6_bos_bear_i,
                valid_bull_i=valid_bull_i,
                valid_bear_i=valid_bear_i,
            )
        else:
            ablation = pd.DataFrame()

        return label_series, ablation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_ablation(
        self,
        df_index,
        i: np.ndarray,
        n: int,
        geom_bull_i,
        geom_bear_i,
        crit2_strict_bull_i,
        crit2_strict_bear_i,
        crit2_loose_bull_i,
        crit2_loose_bear_i,
        crit3_sr_bull_i,
        crit3_sr_bear_i,
        crit5_gann_bull_i,
        crit5_gann_bear_i,
        crit6_bos_bull_i,
        crit6_bos_bear_i,
        valid_bull_i,
        valid_bear_i,
    ) -> pd.DataFrame:
        """Map per-middle-candle boolean arrays (indexed on i) to full-length candle-aligned columns."""

        def _expand(arr_i: np.ndarray) -> np.ndarray:
            """Spread i-indexed boolean values into a length-n array at positions i."""
            out = np.zeros(n, dtype=bool)
            out[i] = arr_i
            return out

        return pd.DataFrame(
            {
                "geometric_bull": _expand(geom_bull_i),
                "geometric_bear": _expand(geom_bear_i),
                "crit2_strict_bull": _expand(crit2_strict_bull_i),
                "crit2_strict_bear": _expand(crit2_strict_bear_i),
                "crit2_loose_bull": _expand(crit2_loose_bull_i),
                "crit2_loose_bear": _expand(crit2_loose_bear_i),
                "crit3_sr_bull": _expand(crit3_sr_bull_i),
                "crit3_sr_bear": _expand(crit3_sr_bear_i),
                "crit5_gann_bull": _expand(crit5_gann_bull_i),
                "crit5_gann_bear": _expand(crit5_gann_bear_i),
                "crit6_bos_bull": _expand(crit6_bos_bull_i),
                "crit6_bos_bear": _expand(crit6_bos_bear_i),
                "valid_bull": _expand(valid_bull_i),
                "valid_bear": _expand(valid_bear_i),
            },
            index=df_index,
        )

    @staticmethod
    def _empty_ablation(index) -> pd.DataFrame:
        cols = [
            "geometric_bull", "geometric_bear",
            "crit2_strict_bull", "crit2_strict_bear",
            "crit2_loose_bull", "crit2_loose_bear",
            "crit3_sr_bull", "crit3_sr_bear",
            "crit5_gann_bull", "crit5_gann_bear",
            "crit6_bos_bull", "crit6_bos_bear",
            "valid_bull", "valid_bear",
        ]
        return pd.DataFrame({c: np.zeros(len(index), dtype=bool) for c in cols}, index=index)
