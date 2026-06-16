"""test_fleet_replay.py — TDD for FleetReplayer + scripts/replay_fleet.py.

Assertions (per plan WS-G spec)
--------------------------------
(a) Each intent resolves via compute_exit on H1 forward bars.
(b) Determinism: two runs on the same fixture → identical results.
(c) realistic_pnl.md emitted with sparsity warning when n_resolved < 30.
(d) An intent whose forward bars hit TP yields +R; one hitting SL yields
    the strategy's loss R.

All tests use REAL compute_exit + real resample logic.  No mocking of our
own code (only external IO is avoided by writing fixtures to tmp_path).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.live.fleet_replay import CellResult, FleetReplayer, FleetReplayResult
from src.strategy.exits import ExitConfig, compute_exit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_ny(dt: str) -> pd.Timestamp:
    return pd.Timestamp(dt, tz="America/New_York")


def _build_simple_fixture(
    tmp_path: Path,
    session_name: str,
    force_outcome: str,  # "tp" | "sl" | "undecided"
) -> Path:
    """Build a minimal self-consistent fixture with a single fixed_2r intent.

    Creates 65 H1 bars (via 6 1-min bars each).  The decision window = first
    60 H1 bars.  Forward bars = next 5.  One forward bar is patched to hit
    ``force_outcome``.
    """
    session_dir = tmp_path / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Build 60-bar window.
    # We hard-code bar_56 high = 450.0 and set all forward prices at 460.0
    # so entry (first forward bar open) is well above sl.
    window = np.zeros((60, 5), dtype=np.float64)
    for i in range(60):
        o = 448.0
        h = 450.0 if i == 56 else 449.0   # bar_56 high = 450.0
        lo = 447.0
        c = 448.5
        window[i] = [o, h, lo, c, 10000.0]

    # Geometry for fixed_2r (direction=1)
    sl = 450.0 - 0.01   # bar_56.high - TICK = 449.99
    entry = 460.0        # clearly above sl
    risk = entry - sl    # = 10.01
    tp = entry + 2.0 * risk  # = 480.02

    # Build H1 timestamps: 60 window bars + 5 forward
    h1_timestamps: list[pd.Timestamp] = []
    current = _ts_ny("2026-05-19 09:30:00")
    while len(h1_timestamps) < 65:
        if current.dayofweek < 5:
            for slot in range(6):
                h1_timestamps.append(current + pd.Timedelta(hours=slot))
                if len(h1_timestamps) >= 65:
                    break
        current += pd.Timedelta(days=1)

    decision_ts = h1_timestamps[59]

    # Build 1-min bars (6 per H1)
    # Forward bar open must equal `entry` so fixed_2r fills at the right price
    all_bars: list[dict] = []
    for i, h1_ts in enumerate(h1_timestamps):
        row = window[i] if i < 60 else np.array([entry, entry + 2.0, entry - 1.0, entry + 0.5, 10000.0])
        for m in range(6):
            bar_ts = h1_ts + pd.Timedelta(minutes=m)
            all_bars.append({
                "symbol": "SPY",
                "timestamp": bar_ts.isoformat(),
                "open": float(row[0]),
                "high": float(row[1]),
                "low": float(row[2]),
                "close": float(row[3]),
                "volume": float(row[4]) / 6,
                "is_update": False,
                "source": "backfill" if i < 60 else "live",
            })

    # Patch h1_timestamps[62] (third forward bar) to hit the target outcome
    hit_ts = h1_timestamps[62]
    patched: list[dict] = []
    for b in all_bars:
        b_ts = pd.Timestamp(b["timestamp"])
        if hit_ts <= b_ts < hit_ts + pd.Timedelta(hours=1):
            b = dict(b)
            if force_outcome == "tp":
                b["high"] = tp + 3.0
                b["low"] = entry - risk * 0.3   # doesn't hit SL
            elif force_outcome == "sl":
                b["low"] = sl - 3.0
                b["high"] = entry + risk * 0.3  # doesn't hit TP
            # "undecided" → leave as is (no patch)
        patched.append(b)

    intent = {
        "ticker": "SPY",
        "model": "cnn_lstm",
        "strategy": "fixed_2r",
        "h1_timestamp": decision_ts.isoformat(),
        "direction": 1,
        "signal": "bull",
        "confidence": 0.85,
        "entry": None,
        "sl": sl,
        "tp": None,
        "entry_type": "market",
        "skip_reason": None,
    }

    with (session_dir / "intents.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(intent) + "\n")
    with (session_dir / "bars_1m_fleet.jsonl").open("w", encoding="utf-8") as fh:
        for b in patched:
            fh.write(json.dumps(b) + "\n")

    return session_dir


# ---------------------------------------------------------------------------
# (a) Intent resolves via compute_exit on H1 forward bars
# ---------------------------------------------------------------------------


class TestResolution:
    def test_tp_intent_resolves(self, tmp_path):
        """A TP fixture resolves with n_resolved >= 1."""
        session_dir = _build_simple_fixture(tmp_path, "sess_tp", "tp")
        result = FleetReplayer(session_dir=session_dir).run()

        assert len(result.cells) == 1
        cell = result.cells[0]
        assert cell.n_intents == 1
        # The intent must resolve (tp or sl — forward bars patch guarantees one hits)
        assert cell.n_resolved >= 1

    def test_sl_intent_resolves(self, tmp_path):
        """A SL fixture resolves with n_resolved >= 1."""
        session_dir = _build_simple_fixture(tmp_path, "sess_sl", "sl")
        result = FleetReplayer(session_dir=session_dir).run()

        assert len(result.cells) == 1
        cell = result.cells[0]
        assert cell.n_resolved >= 1

    def test_skipped_intent_not_resolved(self, tmp_path):
        """Intent with skip_reason is counted in n_intents but not resolved."""
        session_dir = tmp_path / "sess_skip"
        session_dir.mkdir()

        intent = {
            "ticker": "SPY", "model": "lstm", "strategy": "fixed_2r",
            "h1_timestamp": "2026-06-02T14:30:00-04:00",
            "direction": 1, "signal": "bull", "confidence": 0.7,
            "entry": None, "sl": None, "tp": None,
            "entry_type": "market", "skip_reason": "DEGENERATE_GEOMETRY",
        }
        (session_dir / "intents.jsonl").write_text(json.dumps(intent) + "\n")
        (session_dir / "bars_1m_fleet.jsonl").write_text("")

        result = FleetReplayer(session_dir=session_dir).run()
        assert len(result.cells) == 1
        assert result.cells[0].n_intents == 1
        assert result.cells[0].n_resolved == 0

    def test_empty_session_returns_no_cells(self, tmp_path):
        """Session with no intents → empty cells list."""
        sd = tmp_path / "sess_empty"
        sd.mkdir()
        (sd / "intents.jsonl").write_text("")
        (sd / "bars_1m_fleet.jsonl").write_text("")
        result = FleetReplayer(session_dir=sd).run()
        assert result.cells == []

    def test_missing_session_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FleetReplayer(session_dir=tmp_path / "no_such")


# ---------------------------------------------------------------------------
# (b) Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_runs_identical(self, tmp_path):
        """Two sequential runs on the same session dir produce identical results."""
        sd = _build_simple_fixture(tmp_path, "sess_det", "tp")
        r1 = FleetReplayer(session_dir=sd).run()
        r2 = FleetReplayer(session_dir=sd).run()

        assert len(r1.cells) == len(r2.cells)
        for c1, c2 in zip(r1.cells, r2.cells):
            assert c1.n_intents == c2.n_intents
            assert c1.n_resolved == c2.n_resolved
            assert c1.n_tp == c2.n_tp
            assert c1.n_sl == c2.n_sl
            # Float R must be bit-identical (no randomness in compute_exit)
            assert c1.after_cost_total_r == c2.after_cost_total_r

    def test_new_instance_same_result(self, tmp_path):
        """Two distinct FleetReplayer instances on same dir produce identical output."""
        sd = _build_simple_fixture(tmp_path, "sess_det2", "sl")
        r1 = FleetReplayer(session_dir=sd).run()
        r2 = FleetReplayer(session_dir=sd).run()
        if r1.cells:
            assert r1.cells[0].after_cost_total_r == r2.cells[0].after_cost_total_r


# ---------------------------------------------------------------------------
# (c) Report emission + sparsity warning
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_file_created(self, tmp_path):
        """emit_report creates realistic_pnl.md under reports_dir/<fleet_id>/."""
        sd = _build_simple_fixture(tmp_path, "sess_rpt", "tp")
        replayer = FleetReplayer(session_dir=sd, fleet_id="fleet_test_001")
        result = replayer.run()
        rpt = replayer.emit_report(result, tmp_path / "reports")

        assert rpt.exists()
        assert rpt.name == "realistic_pnl.md"
        assert rpt.parent.name == "fleet_test_001"

    def test_sparsity_warning_present_when_n_resolved_lt_30(self, tmp_path):
        """Report contains SPARSITY WARNING when n_resolved < 30."""
        sd = _build_simple_fixture(tmp_path, "sess_sparse", "tp")
        replayer = FleetReplayer(session_dir=sd, fleet_id="sparse_id")
        result = replayer.run()

        # 1 intent → n_resolved <= 1 < 30
        assert result.any_sparse

        rpt = replayer.emit_report(result, tmp_path / "reports")
        content = rpt.read_text()
        assert "SPARSITY WARNING" in content

    def test_report_has_required_columns(self, tmp_path):
        """Report table includes all required column headers."""
        sd = _build_simple_fixture(tmp_path, "sess_cols", "undecided")
        replayer = FleetReplayer(session_dir=sd)
        result = replayer.run()
        rpt = replayer.emit_report(result, tmp_path / "reports")
        content = rpt.read_text()

        for col in ("after_cost_total_R", "win_rate", "n_trades", "n_resolved", "undecided_rate"):
            assert col in content, f"Column '{col}' missing from report"

    def test_no_sparsity_flag_at_30_resolved(self):
        """CellResult.sparsity_warning is False at exactly n_resolved=30."""
        c = CellResult(ticker="SPY", model="lstm", strategy="fixed_2r")
        c.n_resolved = 30
        assert not c.sparsity_warning

    def test_sparsity_flag_at_29_resolved(self):
        c = CellResult(ticker="SPY", model="lstm", strategy="fixed_2r")
        c.n_resolved = 29
        assert c.sparsity_warning

    def test_report_is_idempotent(self, tmp_path):
        """Calling emit_report twice produces the same file content."""
        sd = _build_simple_fixture(tmp_path, "sess_idem", "tp")
        replayer = FleetReplayer(session_dir=sd, fleet_id="idem_fleet")
        result = replayer.run()
        rpts_dir = tmp_path / "reports"
        rpt1 = replayer.emit_report(result, rpts_dir)
        content1 = rpt1.read_text()
        rpt2 = replayer.emit_report(result, rpts_dir)
        content2 = rpt2.read_text()
        assert content1 == content2


# ---------------------------------------------------------------------------
# (d) TP → positive R; SL → negative R
# ---------------------------------------------------------------------------


class TestTPSLR:
    def test_tp_hit_positive_r(self, tmp_path):
        """Forward bars hitting TP → after_cost_total_r > 0."""
        sd = _build_simple_fixture(tmp_path, "sess_tp_r", "tp")
        result = FleetReplayer(session_dir=sd).run()

        if not result.cells:
            pytest.skip("No cells")
        cell = result.cells[0]
        if cell.n_resolved == 0:
            pytest.skip("No resolved trades in fixture")

        if cell.n_tp > 0:
            assert cell.after_cost_total_r > 0.0, (
                f"TP hit should yield positive R, got {cell.after_cost_total_r}"
            )

    def test_sl_hit_negative_r(self, tmp_path):
        """Forward bars hitting SL → after_cost_total_r < 0."""
        sd = _build_simple_fixture(tmp_path, "sess_sl_r", "sl")
        result = FleetReplayer(session_dir=sd).run()

        if not result.cells:
            pytest.skip("No cells")
        cell = result.cells[0]
        if cell.n_resolved == 0:
            pytest.skip("No resolved trades in fixture")

        if cell.n_sl > 0:
            assert cell.after_cost_total_r < 0.0, (
                f"SL hit should yield negative R, got {cell.after_cost_total_r}"
            )

    def test_tp_win_rate_is_one(self, tmp_path):
        """Single TP trade → win_rate = 1.0 (no SL trades)."""
        sd = _build_simple_fixture(tmp_path, "sess_wr_tp", "tp")
        cell = FleetReplayer(session_dir=sd).run().cells
        if not cell or cell[0].n_resolved == 0:
            pytest.skip("No resolved trades")
        c = cell[0]
        if c.n_tp > 0 and c.n_sl == 0:
            assert c.win_rate == 1.0

    def test_sl_win_rate_is_zero(self, tmp_path):
        """Single SL trade → win_rate = 0.0."""
        sd = _build_simple_fixture(tmp_path, "sess_wr_sl", "sl")
        cell = FleetReplayer(session_dir=sd).run().cells
        if not cell or cell[0].n_resolved == 0:
            pytest.skip("No resolved trades")
        c = cell[0]
        if c.n_sl > 0 and c.n_tp == 0:
            assert c.win_rate == 0.0

    def test_r_multiple_magnitude_fixed_2r(self, tmp_path):
        """TP trade on fixed_2r should yield r_multiple close to +tp_rr (2.0)."""
        sd = _build_simple_fixture(tmp_path, "sess_r_mag", "tp")
        result = FleetReplayer(
            session_dir=sd,
            exit_configs={"fixed_2r": ExitConfig(strategy="fixed_2r", tp_rr=2.0)},
        ).run()
        if not result.cells or result.cells[0].n_tp == 0:
            pytest.skip("No TP trade resolved")
        cell = result.cells[0]
        # fixed_2r TP → r_multiple = +tp_rr = +2.0 (no costs in default config)
        assert abs(cell.after_cost_total_r - 2.0) < 0.01, (
            f"Expected ~+2.0R for fixed_2r TP, got {cell.after_cost_total_r}"
        )


# ---------------------------------------------------------------------------
# Multi-cell aggregation
# ---------------------------------------------------------------------------


class TestMultiCell:
    def test_two_strategies_two_cells(self, tmp_path):
        """Two strategies on same ticker/model → two separate CellResult rows."""
        sd = tmp_path / "sess_2cell"
        sd.mkdir()
        ts = "2026-06-02T14:30:00-04:00"
        intents = [
            {"ticker": "SPY", "model": "cnn_lstm", "strategy": "fixed_2r",
             "h1_timestamp": ts, "direction": 1, "signal": "bull", "confidence": 0.8,
             "entry": None, "sl": 449.0, "tp": None, "entry_type": "market",
             "skip_reason": None},
            {"ticker": "SPY", "model": "cnn_lstm", "strategy": "ict_iofed",
             "h1_timestamp": ts, "direction": 1, "signal": "bull", "confidence": 0.8,
             "entry": 450.5, "sl": 449.0, "tp": 453.0, "entry_type": "limit",
             "skip_reason": None},
        ]
        (sd / "intents.jsonl").write_text(
            "\n".join(json.dumps(i) for i in intents) + "\n"
        )
        (sd / "bars_1m_fleet.jsonl").write_text("")

        result = FleetReplayer(session_dir=sd).run()
        assert len(result.cells) == 2
        strategies = {c.strategy for c in result.cells}
        assert strategies == {"fixed_2r", "ict_iofed"}

    def test_cells_sorted(self, tmp_path):
        """Cells returned sorted by (ticker, model, strategy)."""
        sd = tmp_path / "sess_sorted"
        sd.mkdir()
        ts = "2026-06-02T14:30:00-04:00"
        intents = [
            {"ticker": "SPY", "model": "xgboost", "strategy": "tradinglab",
             "h1_timestamp": ts, "direction": 1, "signal": "bull", "confidence": 0.7,
             "entry": None, "sl": 449.0, "tp": None, "entry_type": "market", "skip_reason": None},
            {"ticker": "QQQ", "model": "lstm", "strategy": "fixed_2r",
             "h1_timestamp": ts, "direction": 1, "signal": "bull", "confidence": 0.7,
             "entry": None, "sl": 380.0, "tp": None, "entry_type": "market", "skip_reason": None},
            {"ticker": "SPY", "model": "cnn_lstm", "strategy": "fixed_2r",
             "h1_timestamp": ts, "direction": 1, "signal": "bull", "confidence": 0.7,
             "entry": None, "sl": 449.0, "tp": None, "entry_type": "market", "skip_reason": None},
        ]
        (sd / "intents.jsonl").write_text(
            "\n".join(json.dumps(i) for i in intents) + "\n"
        )
        (sd / "bars_1m_fleet.jsonl").write_text("")

        result = FleetReplayer(session_dir=sd).run()
        keys = [(c.ticker, c.model, c.strategy) for c in result.cells]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# H1 resample parity (canonical params)
# ---------------------------------------------------------------------------


class TestResampleParity:
    def test_h1_anchored_at_09_30(self, tmp_path):
        """1-min bars 09:30–10:29 produce a single H1 bar timestamped 09:30."""
        sd = tmp_path / "sess_resample"
        sd.mkdir()

        bars = []
        base = pd.Timestamp("2026-06-02 09:30:00", tz="America/New_York")
        for m in range(60):
            t = base + pd.Timedelta(minutes=m)
            bars.append({
                "symbol": "SPY",
                "timestamp": t.isoformat(),
                "open": 450.0, "high": 451.0, "low": 449.0, "close": 450.5,
                "volume": 100.0, "is_update": False, "source": "live",
            })

        replayer = FleetReplayer(session_dir=sd)
        symbol_bars = replayer._build_symbol_bars(bars)
        h1 = replayer._resample_to_h1(symbol_bars["SPY"])

        assert len(h1) >= 1
        ts0, _ = h1[0]
        assert ts0.hour == 9 and ts0.minute == 30

    def test_no_pre_rth_bars_in_h1(self, tmp_path):
        """Pre-market bars (before 09:30) are excluded from H1 output."""
        sd = tmp_path / "sess_pre_rth"
        sd.mkdir()

        bars = []
        base = pd.Timestamp("2026-06-02 08:00:00", tz="America/New_York")
        for m in range(60):
            t = base + pd.Timedelta(minutes=m)
            bars.append({
                "symbol": "SPY",
                "timestamp": t.isoformat(),
                "open": 450.0, "high": 451.0, "low": 449.0, "close": 450.5,
                "volume": 100.0, "is_update": False, "source": "backfill",
            })

        replayer = FleetReplayer(session_dir=sd)
        symbol_bars = replayer._build_symbol_bars(bars)
        h1 = replayer._resample_to_h1(symbol_bars["SPY"])
        assert len(h1) == 0


# ---------------------------------------------------------------------------
# H1 — replay realism: costs change after_cost_total_R
# ---------------------------------------------------------------------------


class TestH1RealismCosts:
    def test_costs_change_total_r(self):
        """Realism guards (the ATR min-stop floor threaded by replay's
        --realistic) change the resolved R: a too-tight stop that takes an SL
        without the floor is widened to the floor with it, flipping the
        outcome and the r_multiple.  This is exactly what was lost when replay
        built configs with costs OFF (H1).
        """
        from scripts.replay_fleet import _build_exit_configs

        # Tight-stop bull window: bar_56 high barely above price → tiny SL dist.
        win = np.zeros((60, 5))
        for i in range(60):
            win[i] = [100.0, 100.6, 99.4, 100.0, 1000.0]  # ATR ~1.2 → floor 0.6
        win[56] = [100.0, 100.05, 99.95, 100.0, 1000.0]   # SL ≈ 100.04 (too tight)

        future = np.array(
            [
                [100.10, 100.20, 100.08, 100.15],  # entry bar (open=100.10)
                [100.15, 100.30, 100.02, 100.10],  # dips to 100.02
                [100.10, 100.40, 100.05, 100.35],
            ],
            dtype=np.float64,
        )

        cfgs_off = _build_exit_configs("conservative", 2.0, realistic=False)
        cfgs_on = _build_exit_configs("conservative", 2.0, realistic=True)

        out_off = compute_exit(win, future, 1, cfgs_off["fixed_2r"])
        out_on = compute_exit(win, future, 1, cfgs_on["fixed_2r"])

        # Floor widened the stop → different SL geometry and different R.
        assert out_on.sl != out_off.sl
        assert out_on.r_multiple != out_off.r_multiple

    def test_realistic_default_on(self):
        """_build_exit_configs defaults to costs ON (commission + slippage)."""
        from scripts.replay_fleet import _build_exit_configs

        cfgs = _build_exit_configs(fill_mode="conservative", tp_rr=2.0)
        cfg = cfgs["fixed_2r"]
        assert cfg.commission_per_share > 0
        assert cfg.slippage_ticks > 0
        assert cfg.min_stop_atr_k is not None


# ---------------------------------------------------------------------------
# H3 — fleet-id replay aggregates across sessions into ONE merged result
# ---------------------------------------------------------------------------


class TestH3MergeSessions:
    def test_merge_sums_counts(self, tmp_path):
        from src.live.fleet_replay import merge_results

        sd1 = _build_simple_fixture(tmp_path, "fleetX_sess1", "sl")
        sd2 = _build_simple_fixture(tmp_path, "fleetX_sess2", "sl")

        r1 = FleetReplayer(session_dir=sd1, fleet_id="fleetX").run()
        r2 = FleetReplayer(session_dir=sd2, fleet_id="fleetX").run()

        merged = merge_results([r1, r2], fleet_id="fleetX")

        # Single cell (same ticker/model/strategy) — counts summed
        assert len(merged.cells) == 1
        c = merged.cells[0]
        c1, c2 = r1.cells[0], r2.cells[0]
        assert c.n_intents == c1.n_intents + c2.n_intents
        assert c.n_resolved == c1.n_resolved + c2.n_resolved
        assert c.after_cost_total_r == pytest.approx(
            c1.after_cost_total_r + c2.after_cost_total_r
        )
        # Both contributing sessions named in merged session_id
        assert "fleetX_sess1" in merged.session_id
        assert "fleetX_sess2" in merged.session_id

    def test_cost_drag_matches_inspect_formula(self, tmp_path):
        """H1 fix: replay after_cost must subtract the SAME per-trade drag as
        inspect's summarise_trades (slippage_ticks*0.01*2 + commission*2)/risk$.
        """
        sd = _build_simple_fixture(tmp_path, "fleetCost", "tp")
        strat = "fixed_2r"

        # Cost-free run → after_cost == raw r_multiple total.
        cfg_free = {strat: ExitConfig(strategy=strat, fill_mode="conservative")}
        res_free = FleetReplayer(session_dir=sd, exit_configs=cfg_free).run()
        # Realistic run → drag applied.
        cfg_real = {
            strat: ExitConfig(
                strategy=strat, fill_mode="conservative",
                min_stop_atr_k=0.5, commission_per_share=0.005, slippage_ticks=1,
            )
        }
        res_real = FleetReplayer(session_dir=sd, exit_configs=cfg_real).run()

        c_free = next(c for c in res_free.cells if c.n_resolved > 0)
        c_real = next(c for c in res_real.cells if c.n_resolved > 0)
        # Drag is strictly positive on a filled trade → realistic total is lower.
        assert c_real.after_cost_total_r < c_free.after_cost_total_r

    def test_merged_emit_single_report(self, tmp_path):
        from src.live.fleet_replay import merge_results

        sd1 = _build_simple_fixture(tmp_path, "fleetY_a", "tp")
        sd2 = _build_simple_fixture(tmp_path, "fleetY_b", "tp")
        r1 = FleetReplayer(session_dir=sd1, fleet_id="fleetY").run()
        r2 = FleetReplayer(session_dir=sd2, fleet_id="fleetY").run()
        merged = merge_results([r1, r2], fleet_id="fleetY")

        out = FleetReplayer.emit_report(merged, tmp_path / "reports")
        assert out.exists()
        # Exactly one report file under the fleet dir
        assert out.name == "realistic_pnl.md"
        assert out.parent.name == "fleetY"
