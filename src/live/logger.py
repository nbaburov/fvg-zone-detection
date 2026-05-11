"""logger.py — Session logger: parquet + sqlite + jsonl.

Writes per-session data to logs/paper/<session-id>/. Three stores:
- Parquet (bars_1m, bars_h1, predictions) — buffered, flushed on H1 close or close().
- SQLite (orders, fills) — written immediately per event.
- JSONL (equity, events) — written immediately per event.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.live.slippage import SlippageModel
from src.live.stream import MinuteBar
from src.live.window_builder import H1Bar, WindowEvent
from src.live.decision import TradeAction

logger = logging.getLogger(__name__)


@dataclass
class FillEvent:
    """Fill or partial_fill notification from TradingStream."""

    order_id: str
    filled_at: pd.Timestamp
    fill_price: float
    fill_qty: int
    fill_type: str  # "fill" | "partial_fill"
    side: str       # "buy" | "sell"


class SessionLogger:
    """Logs all session events to logs/paper/<session-id>/."""

    def __init__(self, session_id: str, base_dir: str = "logs/paper") -> None:
        self._session_id = session_id
        self._dir = Path(base_dir) / session_id
        self._dir.mkdir(parents=True, exist_ok=True)

        # In-memory parquet buffers
        self._bars_1m: list[dict] = []
        self._bars_h1: list[dict] = []
        self._predictions: list[dict] = []

        # SQLite connection
        self._conn = sqlite3.connect(str(self._dir / "orders.sqlite"))
        self._fills_conn = sqlite3.connect(str(self._dir / "fills.sqlite"))
        self._init_sqlite()

        # JSONL handles
        self._equity_fh = open(self._dir / "equity.jsonl", "a", encoding="utf-8")
        self._events_fh = open(self._dir / "events.jsonl", "a", encoding="utf-8")

        logger.info("SessionLogger initialised: %s", self._dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_bar_1m(self, bar: MinuteBar) -> None:
        self._bars_1m.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "is_update": bar.is_update,
        })

    def log_bar_h1(self, bar: H1Bar, window_valid: bool) -> None:
        self._bars_h1.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "window_valid": window_valid,
        })
        # Flush parquet on each H1 close (low frequency, manageable size)
        self._flush_parquet()

    def log_prediction(self, event: WindowEvent, action: TradeAction) -> None:
        pred_class = int({"none": 0, "bull": 1, "bear": 2}[action.signal])
        self._predictions.append({
            "timestamp": event.h1_timestamp,
            "proba_none": float(action.proba[0]),
            "proba_bull": float(action.proba[1]),
            "proba_bear": float(action.proba[2]),
            "pred_class": pred_class,
            "signal": action.signal,
            "confidence": action.confidence,
            "skip_reason": action.skip_reason or "",
        })

    def log_order(
        self,
        order_id: str,
        action: TradeAction,
        qty: int,
        sl: float,
        tp: float,
        symbol: str = "SPY",
    ) -> None:
        side = "buy" if action.signal == "bull" else "sell"
        submitted_at = pd.Timestamp.now(tz="America/New_York").isoformat()
        self._conn.execute(
            """INSERT OR IGNORE INTO orders
               (order_id, submitted_at, symbol, side, qty, sl_price, tp_price, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, submitted_at, symbol, side, qty, sl, tp, "new"),
        )
        self._conn.commit()

    def log_fill(self, fill: FillEvent, slippage_model: SlippageModel) -> None:
        adj_pnl: Optional[float] = None
        # P&L is only calculable on a closing fill — track in executor, just record here
        commission = slippage_model.commission(fill.fill_qty)
        self._fills_conn.execute(
            """INSERT INTO fills
               (order_id, filled_at, fill_price, fill_qty, fill_type, adj_pnl, commission)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                fill.order_id,
                fill.filled_at.isoformat(),
                fill.fill_price,
                fill.fill_qty,
                fill.fill_type,
                adj_pnl,
                commission,
            ),
        )
        self._fills_conn.commit()

    def log_equity(self, timestamp: pd.Timestamp, account_snapshot: dict) -> None:
        record = {"timestamp": timestamp.isoformat(), **account_snapshot}
        self._equity_fh.write(json.dumps(record) + "\n")
        self._equity_fh.flush()

    def log_event(self, event_type: str, payload: dict) -> None:
        record = {
            "timestamp": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self._events_fh.write(json.dumps(record) + "\n")
        self._events_fh.flush()

    def close(self) -> None:
        """Flush all buffers and close file handles."""
        self._flush_parquet(force=True)
        self._conn.close()
        self._fills_conn.close()
        self._equity_fh.close()
        self._events_fh.close()
        logger.info("SessionLogger closed: %s", self._dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                submitted_at TEXT,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                sl_price REAL,
                tp_price REAL,
                status TEXT
            )
        """)
        self._conn.commit()

        self._fills_conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                filled_at TEXT,
                fill_price REAL,
                fill_qty INTEGER,
                fill_type TEXT,
                adj_pnl REAL,
                commission REAL
            )
        """)
        self._fills_conn.commit()

    def _flush_parquet(self, force: bool = False) -> None:
        """Write buffered rows to parquet files."""
        if self._bars_1m:
            _append_parquet(self._dir / "bars_1m.parquet", self._bars_1m)
            self._bars_1m = []

        if self._bars_h1:
            _append_parquet(self._dir / "bars_h1.parquet", self._bars_h1)
            self._bars_h1 = []

        if self._predictions:
            _append_parquet(self._dir / "predictions.parquet", self._predictions)
            self._predictions = []


def _append_parquet(path: Path, rows: list[dict]) -> None:
    """Append rows to a parquet file (or create it if it doesn't exist)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pq.read_table(str(path)).to_pandas()
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    pq.write_table(pa.Table.from_pandas(combined), str(path))
