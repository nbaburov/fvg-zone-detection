"""test_fleet_state.py — TDD coverage for FleetState.

Tests
-----
- add_open / open_positions round-trip
- resolve appends to ledger, removes from open
- durable reload: open positions survive across FleetState instances
- idempotent re-apply: resolving same intent_id twice is a no-op
- open position already in ledger is skipped on reload
- resolved_trades returns full ledger
- schema_version present in serialised lines
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.live.fleet_state import (
    SCHEMA_VERSION,
    FleetState,
    OpenSimPosition,
    SimTrade,
    make_intent_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TS = pd.Timestamp("2024-01-15 10:30:00", tz="America/New_York")
TS2 = pd.Timestamp("2024-01-15 11:30:00", tz="America/New_York")


def _pos(
    intent_id: str = "SPY:lstm:fixed_2r:2024-01-15T10:30:00-05:00",
    direction: int = 1,
    fill_price: float = 470.0,
    sl: float = 468.0,
    tp: float = 474.0,
) -> OpenSimPosition:
    return OpenSimPosition(
        intent_id=intent_id,
        ticker="SPY",
        model="lstm",
        strategy="fixed_2r",
        direction=direction,
        fill_price=fill_price,
        sl=sl,
        tp=tp,
        fill_ts=TS,
        accumulated_bars=[],
        cell_key="SPY:lstm:fixed_2r",
    )


def _trade(
    intent_id: str = "SPY:lstm:fixed_2r:2024-01-15T10:30:00-05:00",
    outcome: str = "tp",
    r_multiple: float = 2.0,
) -> SimTrade:
    return SimTrade(
        intent_id=intent_id,
        ticker="SPY",
        model="lstm",
        strategy="fixed_2r",
        direction=1,
        fill_price=470.0,
        sl=468.0,
        tp=474.0,
        fill_ts=TS,
        close_price=474.0,
        close_ts=TS2,
        outcome=outcome,
        r_multiple=r_multiple,
        cell_key="SPY:lstm:fixed_2r",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_add_open_and_open_positions(tmp_path):
    fs = FleetState("fleet1", base_dir=tmp_path)
    p = _pos()
    fs.add_open(p)
    positions = fs.open_positions()
    assert len(positions) == 1
    assert positions[0].intent_id == p.intent_id
    assert positions[0].fill_price == pytest.approx(470.0)


def test_resolve_moves_to_ledger(tmp_path):
    fs = FleetState("fleet1", base_dir=tmp_path)
    p = _pos()
    fs.add_open(p)
    t = _trade()
    fs.resolve(t)

    assert len(fs.open_positions()) == 0
    trades = fs.resolved_trades()
    assert len(trades) == 1
    assert trades[0].intent_id == t.intent_id
    assert trades[0].r_multiple == pytest.approx(2.0)


def test_durable_reload_restores_open_positions(tmp_path):
    """Open positions survive across FleetState instances (cross-session)."""
    fs1 = FleetState("fleet1", base_dir=tmp_path)
    p = _pos()
    fs1.add_open(p)

    # New instance — should reload
    fs2 = FleetState("fleet1", base_dir=tmp_path)
    positions = fs2.open_positions()
    assert len(positions) == 1
    assert positions[0].intent_id == p.intent_id
    assert positions[0].fill_price == pytest.approx(470.0)


def test_durable_reload_restores_multiple_positions(tmp_path):
    fs1 = FleetState("fleet1", base_dir=tmp_path)
    p1 = _pos("id1")
    p2 = _pos("id2", fill_price=471.0)
    fs1.add_open(p1)
    fs1.add_open(p2)

    fs2 = FleetState("fleet1", base_dir=tmp_path)
    ids = {p.intent_id for p in fs2.open_positions()}
    assert ids == {"id1", "id2"}


def test_idempotent_resolve(tmp_path):
    """Re-resolving the same intent_id is a no-op — no duplicate in ledger."""
    fs = FleetState("fleet1", base_dir=tmp_path)
    p = _pos()
    fs.add_open(p)
    t = _trade()
    fs.resolve(t)
    fs.resolve(t)  # second call — no-op

    trades = fs.resolved_trades()
    assert len(trades) == 1


def test_reload_skips_already_resolved_open_position(tmp_path):
    """An open position that was resolved in a previous session is not reloaded."""
    fs1 = FleetState("fleet1", base_dir=tmp_path)
    p = _pos()
    fs1.add_open(p)
    t = _trade()
    fs1.resolve(t)

    # Simulate a corrupt state: manually re-write open_positions.jsonl with
    # the resolved intent (shouldn't happen normally, but guard against it).
    open_path = fs1.state_dir / "open_positions.jsonl"
    with open_path.open("w") as fh:
        fh.write(json.dumps(p.to_dict()) + "\n")

    fs2 = FleetState("fleet1", base_dir=tmp_path)
    # Should NOT appear in open positions since it's in the ledger
    assert len(fs2.open_positions()) == 0


def test_resolved_trades_empty_when_no_ledger(tmp_path):
    fs = FleetState("fleet1", base_dir=tmp_path)
    assert fs.resolved_trades() == []


def test_schema_version_in_serialised_lines(tmp_path):
    fs = FleetState("fleet1", base_dir=tmp_path)
    p = _pos()
    fs.add_open(p)
    t = _trade()
    fs.resolve(t)

    # Open positions file
    open_path = fs.state_dir / "open_positions.jsonl"
    # After resolve, open file is empty but should have been flushed
    # (resolve removes from open then rewrites)

    # Ledger file
    ledger_path = fs.state_dir / "resolved_ledger.jsonl"
    with ledger_path.open() as fh:
        line = fh.readline()
    d = json.loads(line)
    assert d["schema_version"] == SCHEMA_VERSION


def test_make_intent_id_deterministic():
    ts = pd.Timestamp("2024-01-15 10:30:00-05:00")
    id1 = make_intent_id("SPY", "lstm", "fixed_2r", ts)
    id2 = make_intent_id("SPY", "lstm", "fixed_2r", ts)
    assert id1 == id2
    assert "SPY" in id1
    assert "lstm" in id1


def test_open_position_serialisation_round_trip():
    p = _pos()
    p.accumulated_bars = [{"open": 470.0, "high": 471.0, "low": 469.5, "close": 470.5, "volume": 1000, "ts": "2024-01-15T10:31:00-05:00"}]
    d = p.to_dict()
    p2 = OpenSimPosition.from_dict(d)
    assert p2.intent_id == p.intent_id
    assert p2.fill_price == pytest.approx(p.fill_price)
    assert len(p2.accumulated_bars) == 1


def test_sim_trade_serialisation_round_trip():
    t = _trade()
    d = t.to_dict()
    t2 = SimTrade.from_dict(d)
    assert t2.intent_id == t.intent_id
    assert t2.r_multiple == pytest.approx(t.r_multiple)
    assert t2.outcome == t.outcome
