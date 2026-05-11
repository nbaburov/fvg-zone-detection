#!/usr/bin/env python
"""paper_trade.py — Live paper-trading harness for SMC FVG detection.

Single-model, single-process. Connects to Alpaca WebSocket, assembles
RTH-anchored H1 windows bit-identical to training, runs inference at
each H1 close, and submits bracket orders to a paper account.

Usage
-----
# Live trading
python scripts/paper_trade.py \\
    --model lstm:checkpoints/lstm/lstm_seed42.pt \\
    --session 20260511_093500 \\
    --symbol SPY \\
    --threshold 0.5 \\
    --risk-pct 1.0 \\
    --max-daily-dd 2.0 \\
    --tp-rr 2.0 \\
    --slippage-ticks 1

# Replay a prior session (offline, no network)
python scripts/paper_trade.py \\
    --replay 20260511_093500 \\
    --model lstm:checkpoints/lstm/lstm_seed42.pt \\
    --threshold 0.5

Environment variables required (live mode)
-------------------------------------------
    ALPACA_API_KEY      Your Alpaca API key
    ALPACA_SECRET_KEY   Your Alpaca secret key
    ALPACA_PAPER        Set to "true" for paper trading (default "true")

Environment variables (read from .env if python-dotenv installed)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_trade")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper_trade.py",
        description="SMC FVG live paper-trading harness (single-model per process).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        required=True,
        metavar="NAME:CHECKPOINT_PATH",
        help=(
            "Model adapter and checkpoint path, separated by colon. "
            "Example: lstm:checkpoints/lstm/lstm_seed42.pt"
        ),
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="SESSION_ID",
        help=(
            "Session ID for log directory (format YYYYMMDD_HHMMSS). "
            "Auto-generated from current time if not provided."
        ),
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Ticker symbol to trade. Default: SPY.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for signal emission. Default: 0.5.",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=1.0,
        metavar="PCT",
        help="Equity percent to risk per trade (e.g. 1.0 = 1%%). Default: 1.0.",
    )
    parser.add_argument(
        "--max-daily-dd",
        type=float,
        default=2.0,
        metavar="PCT",
        help="Daily drawdown %% that triggers kill switch. Default: 2.0.",
    )
    parser.add_argument(
        "--tp-rr",
        type=float,
        default=2.0,
        metavar="R",
        help="Take-profit R-multiple. Default: 2.0.",
    )
    parser.add_argument(
        "--slippage-ticks",
        type=int,
        default=1,
        dest="slippage_ticks",
        help="Ticks of slippage to apply in P&L accounting. Default: 1.",
    )
    parser.add_argument(
        "--replay",
        default=None,
        metavar="SESSION_ID",
        help=(
            "Offline replay mode. Load bars_h1.parquet from session and re-run "
            "inference. No streaming, no orders. Requires --model."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Stream and build windows but skip order submission. "
            "Useful for verifying stream + window pipeline without trading."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------


def load_adapter_from_spec(spec: str):
    """Parse 'name:path' and instantiate the matching ModelAdapter."""
    from src.inspect.registry import _get_registry

    if ":" not in spec:
        raise ValueError(
            f"--model must be NAME:CHECKPOINT_PATH (got '{spec}'). "
            "Example: lstm:checkpoints/lstm/lstm_seed42.pt"
        )

    name, checkpoint_path = spec.split(":", 1)
    checkpoint_path = Path(checkpoint_path)

    registry = _get_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown adapter '{name}'. Available: {available}")

    adapter_cls = registry[name]
    # Adapters accept checkpoint_dir as first arg — pass the parent dir
    # and handle single-file checkpoints via the checkpoint_file kwarg
    checkpoint_dir = checkpoint_path.parent.parent  # e.g. checkpoints/
    checkpoint_file = checkpoint_path.name

    logger.info("Loading adapter '%s' from %s", name, checkpoint_path)
    return adapter_cls(checkpoint_dir, checkpoint_file=checkpoint_file)


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------


def run_replay(args: argparse.Namespace) -> None:
    from src.live.replay import SessionReplayer

    logger.info("REPLAY MODE: session=%s", args.replay)
    adapter = load_adapter_from_spec(args.model)

    replayer = SessionReplayer(
        session_id=args.replay,
        adapter=adapter,
        threshold=args.threshold,
    )
    summary = replayer.run()

    if summary.divergences:
        logger.warning(
            "%d divergence(s) detected — check for non-determinism.", len(summary.divergences)
        )
        sys.exit(1)
    else:
        logger.info("Replay complete. Match rate: %.1f%%", summary.match_rate * 100)


# ---------------------------------------------------------------------------
# Live trading mode
# ---------------------------------------------------------------------------


async def live_main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

    if not args.dry_run and (not api_key or not secret_key):
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment or .env file."
        )

    session_id = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(
        "Starting live session: id=%s symbol=%s model=%s threshold=%.2f dry_run=%s",
        session_id, args.symbol, args.model, args.threshold, args.dry_run,
    )

    # Load adapter
    adapter = load_adapter_from_spec(args.model)

    # Initialise components
    from src.live.decision import SingleModelDecision
    from src.live.logger import SessionLogger
    from src.live.slippage import SlippageModel
    from src.live.stream import AlpacaBarStream
    from src.live.window_builder import LiveWindowBuilder

    log = SessionLogger(session_id)
    builder = LiveWindowBuilder()
    slippage = SlippageModel(ticks=args.slippage_ticks)
    decision = SingleModelDecision(adapter, threshold=args.threshold)

    # Trading client (only in live mode)
    trading_client = None
    executor = None

    if not args.dry_run:
        from alpaca.trading.client import TradingClient

        trading_client = TradingClient(api_key, secret_key, paper=paper)

        from src.live.execution import PaperExecutor

        executor = PaperExecutor(
            trading_client=trading_client,
            risk_pct=args.risk_pct / 100.0,
            tp_rr=args.tp_rr,
            max_daily_dd=args.max_daily_dd / 100.0,
            symbol=args.symbol,
        )

        # Cold-start backfill: fetch today's 1-min bars and warm up the window builder
        _backfill_from_rest(api_key, secret_key, args.symbol, builder)

    def on_bar(bar):
        log.log_bar_1m(bar)
        event = builder.on_bar(bar)
        if event is None:
            return

        log.log_bar_h1(event.h1_bar, window_valid=event.window is not None)
        action = decision.decide(event)
        log.log_prediction(event, action)

        if action.signal != "none" and executor is not None:
            executor.on_trade_action(action, log)

        if trading_client is not None:
            try:
                account = trading_client.get_account()
                log.log_equity(
                    event.h1_timestamp,
                    {
                        "cash": float(account.cash),
                        "portfolio_value": float(account.portfolio_value),
                        "unrealised_pnl": float(getattr(account, "unrealized_pl", 0) or 0),
                        "realised_pnl": float(getattr(account, "realized_pl_ytd", 0) or 0),
                        "drawdown_pct": 0.0,  # computed in executor
                    },
                )
            except Exception as exc:
                logger.warning("Could not fetch equity snapshot: %s", exc)

    def on_reconnect():
        logger.info("Reconnected — triggering REST gap-fill.")
        _backfill_from_rest(api_key, secret_key, args.symbol, builder)

    stream = AlpacaBarStream(
        api_key=api_key,
        secret_key=secret_key,
        symbol=args.symbol,
        on_bar=on_bar,
        on_reconnect=on_reconnect,
        paper=paper,
    )

    # Subscribe to fill events from TradingStream
    fill_task = None
    if not args.dry_run and trading_client is not None:
        fill_task = asyncio.create_task(_run_fill_stream(api_key, secret_key, paper, executor, log, slippage))

    try:
        await stream.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down session.")
    except RuntimeError as exc:
        if "STREAM_FATAL" in str(exc):
            logger.error("STREAM_FATAL received. Halting.")
            log.log_event("STREAM_FATAL", {"message": str(exc)})
        else:
            raise
    finally:
        if executor is not None:
            executor.on_session_close()
        if fill_task is not None:
            fill_task.cancel()
        log.close()
        logger.info("Session %s closed.", session_id)


async def _run_fill_stream(api_key, secret_key, paper, executor, log, slippage):
    """Subscribe to TradingStream for fill events."""
    try:
        from alpaca.trading.stream import TradingStream
        from src.live.logger import FillEvent
        import pandas as pd

        trading_stream = TradingStream(api_key, secret_key, paper=paper)

        async def fill_handler(data):
            if data.event in ("fill", "partial_fill"):
                fill = FillEvent(
                    order_id=str(data.order.id),
                    filled_at=pd.Timestamp.now(tz="America/New_York"),
                    fill_price=float(data.order.filled_avg_price or 0),
                    fill_qty=int(data.order.filled_qty or 0),
                    fill_type=data.event,
                    side=str(data.order.side.value),
                )
                if executor is not None:
                    executor.on_fill_event(fill)
                log.log_fill(fill, slippage)

        trading_stream.subscribe_trade_updates(fill_handler)
        await trading_stream._run_forever()
    except Exception as exc:
        logger.warning("Fill stream error: %s", exc)


def _backfill_from_rest(api_key: str, secret_key: str, symbol: str, builder: "LiveWindowBuilder") -> None:
    """Fetch today's 1-min bars via REST and feed through gap_fill()."""
    try:
        import pandas as pd
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from src.live.stream import MinuteBar

        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

        now = pd.Timestamp.now(tz="America/New_York")
        # Use today's session start; fall back to previous day if before RTH
        session_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now < session_start:
            session_start = session_start - pd.Timedelta(days=1)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=session_start.isoformat(),
            end=now.isoformat(),
            adjustment="raw",
            feed="iex",
        )
        bars = client.get_stock_bars(req)[symbol]
        if bars is None or len(bars) == 0:
            logger.info("No historical 1-min bars available for backfill.")
            return

        minute_bars = []
        for bar in bars:
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert("America/New_York")
            minute_bars.append(MinuteBar(
                symbol=symbol,
                timestamp=ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            ))

        logger.info("Backfilling %d 1-min bars for warm-up.", len(minute_bars))
        builder.gap_fill(minute_bars)
        logger.info("Warm-up complete. H1 buffer: %d bars.", builder.h1_count)

    except Exception as exc:
        logger.warning("REST backfill failed (proceeding without warm-up): %s", exc)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.replay is not None:
        run_replay(args)
        return

    try:
        asyncio.run(live_main(args))
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
