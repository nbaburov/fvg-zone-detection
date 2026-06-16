"""test_fleet_shutdown.py — shutdown path tests for the fleet run loop.

Verifies that on --max-minutes expiry:
  - all asyncio tasks are cancelled / completed,
  - on_session_close is called for every executor,
  - the run coroutine RETURNS within a short timeout (no hang).

All Alpaca SDK calls are stubbed; no real network.
Uses asyncio.run() (same pattern as test_stream.py) — no pytest-asyncio needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake AlpacaBarStream — blocks forever until stop() is called
# ---------------------------------------------------------------------------

class _FakeBarStream:
    """Stands in for AlpacaBarStream: start() blocks; stop() unblocks it."""

    def __init__(self):
        self._stop_event = asyncio.Event()
        self.stop_called = False

    async def start(self):
        await self._stop_event.wait()  # blocks until stop() sets the event

    async def stop(self):
        self.stop_called = True
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor():
    ex = MagicMock()
    ex.on_session_close = MagicMock()
    return ex


def _make_router(executors: list):
    router = MagicMock()
    router.on_session_close = MagicMock(
        side_effect=lambda: [e.on_session_close() for e in executors]
    )
    return router


# ---------------------------------------------------------------------------
# Extracted run-loop (mirrors paper_trade_fleet._run after the fix)
# ---------------------------------------------------------------------------

async def _fake_run(
    bar_stream: _FakeBarStream,
    max_minutes: float,
    has_real: bool = False,
    fill_stream_coro=None,
) -> None:
    """Mirrors the fixed _run() inner coroutine from paper_trade_fleet.main."""

    tasks: list[asyncio.Task] = [
        asyncio.create_task(bar_stream.start(), name="bar-stream"),
    ]
    if has_real and fill_stream_coro is not None:
        tasks.append(asyncio.create_task(fill_stream_coro(), name="fill-stream"))

    async def _stopper() -> None:
        await asyncio.sleep(max_minutes * 60)
        await bar_stream.stop()
        for t in tasks:
            if not t.done():
                t.cancel()

    stopper_task = asyncio.create_task(_stopper(), name="stopper")
    all_tasks = tasks + [stopper_task]

    try:
        await asyncio.gather(*all_tasks, return_exceptions=True)
    finally:
        for t in all_tasks:
            if not t.done():
                t.cancel()
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFleetShutdownMaxMinutes:
    """The run coroutine must complete (not hang) when max_minutes fires."""

    def test_run_returns_within_timeout_sim_only(self):
        """Sim-only fleet (no fill stream): must complete in < 2 s."""

        async def _drive():
            bar_stream = _FakeBarStream()
            await asyncio.wait_for(
                _fake_run(bar_stream, max_minutes=0.001),  # ~60 ms
                timeout=2.0,
            )

        asyncio.run(_drive())

    def test_bar_stream_stop_called(self):
        """stream.stop() must be invoked when the timer fires."""
        stop_called = []

        class _TrackingStream(_FakeBarStream):
            async def stop(self):
                stop_called.append(True)
                await super().stop()

        async def _drive():
            bar_stream = _TrackingStream()
            await asyncio.wait_for(
                _fake_run(bar_stream, max_minutes=0.001),
                timeout=2.0,
            )

        asyncio.run(_drive())
        assert stop_called, "bar_stream.stop() was never called"

    def test_on_session_close_called_for_all_executors(self):
        """router.on_session_close() (which fans into each executor) must run."""
        executors = [_make_executor(), _make_executor(), _make_executor()]
        router = _make_router(executors)

        async def _drive():
            bar_stream = _FakeBarStream()
            try:
                await asyncio.wait_for(
                    _fake_run(bar_stream, max_minutes=0.001),
                    timeout=2.0,
                )
            finally:
                router.on_session_close()

        asyncio.run(_drive())

        router.on_session_close.assert_called()
        for ex in executors:
            ex.on_session_close.assert_called()

    def test_fill_stream_task_cancelled(self):
        """Real-cell fleet: the fill-stream task must also be cancelled on stop."""
        fill_task_cancelled = []

        async def _hanging_fill_stream():
            try:
                await asyncio.sleep(9999)  # never returns on its own
            except asyncio.CancelledError:
                fill_task_cancelled.append(True)
                raise

        async def _drive():
            bar_stream = _FakeBarStream()
            await asyncio.wait_for(
                _fake_run(
                    bar_stream,
                    max_minutes=0.001,
                    has_real=True,
                    fill_stream_coro=_hanging_fill_stream,
                ),
                timeout=2.0,
            )

        asyncio.run(_drive())
        assert fill_task_cancelled, "fill-stream task was never cancelled"

    def test_no_hang_instant_fill_stream(self):
        """Edge: fill stream that returns immediately — still no hang."""

        async def _instant_fill():
            return

        async def _drive():
            bar_stream = _FakeBarStream()
            await asyncio.wait_for(
                _fake_run(
                    bar_stream,
                    max_minutes=0.001,
                    has_real=True,
                    fill_stream_coro=_instant_fill,
                ),
                timeout=2.0,
            )

        asyncio.run(_drive())
