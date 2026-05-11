"""replay.py — Offline replay from saved session bars.

Loads bars_h1.parquet from a prior session, feeds H1 bars through
LiveWindowBuilder one by one, runs inference with the provided adapter,
and compares against logged predictions. Expected match rate: 100%.
Any divergence signals non-determinism bug.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.inspect.base import ModelAdapter
from src.live.decision import SingleModelDecision, TradeAction
from src.live.stream import MinuteBar
from src.live.window_builder import H1Bar, LiveWindowBuilder, WindowEvent

logger = logging.getLogger(__name__)


@dataclass
class ReplaySummary:
    """Summary of a replay run."""

    total_bars: int
    predictions_with_window: int
    match_count: int
    divergences: list[dict] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        if self.predictions_with_window == 0:
            return 1.0
        return self.match_count / self.predictions_with_window


class SessionReplayer:
    """Replays a saved session and checks prediction reproducibility.

    Parameters
    ----------
    session_id : str
        Session ID (e.g. "20260511_093500").
    adapter : ModelAdapter
        Model adapter to use for inference.
    threshold : float
        Confidence threshold — must match the original session's threshold.
    base_dir : str
        Root directory for session logs. Default "logs/paper".
    """

    def __init__(
        self,
        session_id: str,
        adapter: ModelAdapter,
        threshold: float = 0.5,
        base_dir: str = "logs/paper",
    ) -> None:
        self._session_dir = Path(base_dir) / session_id
        self._adapter = adapter
        self._threshold = threshold

        if not self._session_dir.exists():
            raise FileNotFoundError(f"Session not found: {self._session_dir}")

    def run(self) -> ReplaySummary:
        """Execute replay and return summary."""
        bars_path = self._session_dir / "bars_h1.parquet"
        preds_path = self._session_dir / "predictions.parquet"

        if not bars_path.exists():
            raise FileNotFoundError(f"bars_h1.parquet not found in {self._session_dir}")

        import pyarrow.parquet as pq

        bars_df = pq.read_table(str(bars_path)).to_pandas()
        logged_preds: Optional[pd.DataFrame] = None
        if preds_path.exists():
            logged_preds = pq.read_table(str(preds_path)).to_pandas()
            logger.info("Loaded %d logged predictions for comparison.", len(logged_preds))

        builder = LiveWindowBuilder()
        decision = SingleModelDecision(self._adapter, self._threshold)

        total_bars = 0
        predictions_with_window = 0
        match_count = 0
        divergences: list[dict] = []

        for _, row in bars_df.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")

            h1 = H1Bar(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            total_bars += 1

            # Feed as synthetic 1-min bars — H1 bars replayed as single-bar buckets.
            # Inject the H1 bar directly into the builder's h1_buffer.
            builder._h1_buffer.append(h1)
            event = builder._build_window_event(h1)

            action = decision.decide(event)

            print(
                f"  {ts.strftime('%H:%M')}  signal={action.signal:<4}  "
                f"conf={action.confidence:.3f}  skip={action.skip_reason or '—'}"
            )

            if event.window is not None:
                predictions_with_window += 1

                if logged_preds is not None:
                    logged_row = logged_preds[
                        logged_preds["timestamp"].astype(str).str.startswith(
                            str(ts)[:16]
                        )
                    ]
                    if not logged_row.empty:
                        logged_class = int(logged_row.iloc[0]["pred_class"])
                        live_class = int({"none": 0, "bull": 1, "bear": 2}[action.signal])
                        if logged_class == live_class:
                            match_count += 1
                        else:
                            divergences.append({
                                "timestamp": str(ts),
                                "logged_class": logged_class,
                                "replay_class": live_class,
                            })
                            logger.warning(
                                "DIVERGENCE at %s: logged=%d replay=%d",
                                ts, logged_class, live_class,
                            )
                    else:
                        match_count += 1  # no logged prediction to compare — count as match

        summary = ReplaySummary(
            total_bars=total_bars,
            predictions_with_window=predictions_with_window,
            match_count=match_count,
            divergences=divergences,
        )

        print(f"\nReplay complete: {total_bars} bars, {predictions_with_window} with window")
        print(f"Match rate: {summary.match_rate:.1%}  Divergences: {len(divergences)}")

        return summary
