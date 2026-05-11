"""Unit tests for SessionLogger.

Tests:
- Write a few rows to each store; read back and verify column names + types.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.live.decision import TradeAction
from src.live.logger import FillEvent, SessionLogger
from src.live.slippage import SlippageModel
from src.live.stream import MinuteBar
from src.live.window_builder import H1Bar, WindowEvent


def make_h1_bar(ts: str = "2024-01-02 10:30:00") -> H1Bar:
    return H1Bar(
        timestamp=pd.Timestamp(ts, tz="America/New_York"),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=5000.0,
    )


def make_window_event(ts: str = "2024-01-02 10:30:00") -> WindowEvent:
    h1 = make_h1_bar(ts)
    return WindowEvent(
        h1_timestamp=h1.timestamp,
        h1_bar=h1,
        window=np.zeros((60, 5), dtype=np.float32),
        raw_window=np.zeros((60, 5), dtype=np.float64),
        skip_reason=None,
    )


def make_trade_action(signal: str = "bull", ts: str = "2024-01-02 10:30:00") -> TradeAction:
    return TradeAction(
        signal=signal,
        confidence=0.8,
        proba=np.array([0.1, 0.8, 0.1], dtype=np.float32),
        h1_timestamp=pd.Timestamp(ts, tz="America/New_York"),
        gap_low=99.0,
        gap_high=101.0,
        skip_reason=None,
    )


@pytest.fixture
def tmp_logger(tmp_path):
    """Create a SessionLogger with a temporary base_dir."""
    log = SessionLogger(session_id="test_session", base_dir=str(tmp_path))
    yield log, tmp_path
    log.close()


class TestLoggerSchema:
    def test_bars_1m_parquet_schema(self, tmp_logger):
        log, base = tmp_logger

        ts = pd.Timestamp("2024-01-02 10:01:00", tz="America/New_York")
        bar = MinuteBar(
            symbol="SPY",
            timestamp=ts,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1234.0,
        )
        log.log_bar_1m(bar)
        log.close()

        # Reopen log object — parquet is flushed in log_bar_h1 or close()
        session_dir = base / "test_session"
        parquet_path = session_dir / "bars_1m.parquet"
        assert parquet_path.exists(), "bars_1m.parquet was not created"

        import pyarrow.parquet as pq
        df = pq.read_table(str(parquet_path)).to_pandas()
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume", "is_update"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 1

    def test_bars_h1_parquet_schema(self, tmp_logger):
        log, base = tmp_logger

        h1 = make_h1_bar()
        log.log_bar_1m(  # need at least one 1m bar to flush
            MinuteBar(
                symbol="SPY",
                timestamp=pd.Timestamp("2024-01-02 10:00:00", tz="America/New_York"),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0,
            )
        )
        log.log_bar_h1(h1, window_valid=True)
        log.close()

        session_dir = base / "test_session"
        parquet_path = session_dir / "bars_h1.parquet"
        assert parquet_path.exists()

        import pyarrow.parquet as pq
        df = pq.read_table(str(parquet_path)).to_pandas()
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume", "window_valid"}
        assert expected_cols.issubset(set(df.columns))
        assert df.iloc[0]["window_valid"] == True

    def test_predictions_parquet_schema(self, tmp_logger):
        log, base = tmp_logger

        event = make_window_event()
        action = make_trade_action("bull")
        log.log_prediction(event, action)

        h1 = make_h1_bar()
        log.log_bar_h1(h1, window_valid=True)
        log.close()

        session_dir = base / "test_session"
        parquet_path = session_dir / "predictions.parquet"
        assert parquet_path.exists()

        import pyarrow.parquet as pq
        df = pq.read_table(str(parquet_path)).to_pandas()
        expected_cols = {
            "timestamp", "proba_none", "proba_bull", "proba_bear",
            "pred_class", "signal", "confidence", "skip_reason",
        }
        assert expected_cols.issubset(set(df.columns))
        assert df.iloc[0]["signal"] == "bull"
        assert int(df.iloc[0]["pred_class"]) == 1

    def test_orders_sqlite_schema(self, tmp_logger):
        log, base = tmp_logger

        action = make_trade_action("bear")
        log.log_order("order_123", action, qty=10, sl=101.01, tp=97.98)
        log.close()

        session_dir = base / "test_session"
        conn = sqlite3.connect(str(session_dir / "orders.sqlite"))
        rows = conn.execute("SELECT * FROM orders WHERE order_id='order_123'").fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        # Columns: order_id, submitted_at, symbol, side, qty, sl_price, tp_price, status
        assert row[0] == "order_123"
        assert row[3] == "sell"
        assert row[4] == 10
        assert row[5] == pytest.approx(101.01)

    def test_fills_sqlite_schema(self, tmp_logger):
        log, base = tmp_logger

        fill = FillEvent(
            order_id="order_123",
            filled_at=pd.Timestamp("2024-01-02 10:31:05", tz="America/New_York"),
            fill_price=100.05,
            fill_qty=10,
            fill_type="fill",
            side="buy",
        )
        slippage = SlippageModel()
        log.log_fill(fill, slippage)
        log.close()

        session_dir = base / "test_session"
        conn = sqlite3.connect(str(session_dir / "fills.sqlite"))
        rows = conn.execute("SELECT * FROM fills").fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        assert row[1] == "order_123"
        assert float(row[3]) == pytest.approx(100.05)

    def test_equity_jsonl(self, tmp_logger):
        import json

        log, base = tmp_logger

        ts = pd.Timestamp("2024-01-02 10:30:00", tz="America/New_York")
        log.log_equity(ts, {"cash": 50000.0, "portfolio_value": 100000.0, "drawdown_pct": 0.0})
        log.close()

        session_dir = base / "test_session"
        lines = (session_dir / "equity.jsonl").read_text().strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "portfolio_value" in record
        assert record["portfolio_value"] == pytest.approx(100000.0)
