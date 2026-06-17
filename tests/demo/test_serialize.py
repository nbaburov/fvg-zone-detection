"""test_serialize.py — TDD tests for src/demo/serialize.py.

Tests:
  1. payload_round_trip: json.dumps round-trips without error
  2. no_lookahead: every zone t_detect >= pattern-close bar timestamp
  3. zone_x_ordering: x0 <= x1 for all zones
  4. frame_times_sorted_unique: frame_times = sorted unique bar timestamps in window
  5. zone_state_matches_outcome: state/t_resolve match Trade.outcome
  6. bars_within_window: all bar timestamps fall within window start/end
  7. determinism: byte-identical JSON on two calls (no RNG/Date.now, sorted keys)
  8. empty_models: empty tracks list raises ValueError
  9. zone_y_ordering: y0 <= y1 for all zones
 10. pending_has_null_resolve: pending zones have t_resolve=null
 11. zone_conf: every zone has a conf field in [0.0, 1.0]

Fixture design
--------------
We generate a flat 1-h DatetimeIndex. The ModelTrack.pred_ts
corresponds to the LAST bar of each 60-bar window. With stride=1 on a sequence
of T bars, window i has last-bar index = 59+i, so pred_ts[i] = ts[59+i].

For testing we use a simpler direct model: we create a large pool of timestamps,
set pred_ts = ts[60:60+n_windows], and set display bars = ts[bar_start:bar_start+display_bars]
where bar_start < 60+n_windows so display bars overlap pred_ts.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.demo.predict import ModelTrack
from src.demo.serialize import build_demo_payload, build_multi_scene_payload, payload_to_json, _ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, start: str = "2025-01-02 09:30") -> pd.DatetimeIndex:
    """Return n consecutive hourly timestamps (no tz gaps — fine for unit tests)."""
    return pd.date_range(start, periods=n, freq="1h", tz="America/New_York")


def _make_window_raw(
    bar56_high: float = 502.0,
    bar56_low: float = 498.0,
    bar58_high: float = 504.0,
    bar58_low: float = 500.5,
) -> np.ndarray:
    """Build a (60, 5) raw window with known gap bars."""
    w = np.zeros((60, 5), dtype=np.float32)
    for i in range(56):
        w[i] = [500.0, 502.0, 498.0, 500.5, 500.0]
    w[56] = [499.0, bar56_high, bar56_low, 501.0, 1000.0]
    w[57] = [501.0, 505.0, 499.0, 504.0, 1500.0]
    w[58] = [503.0, bar58_high, bar58_low, 503.5, 800.0]
    w[59] = [503.5, 506.0, 503.0, 505.0, 900.0]
    return w


def _make_trade(
    window_idx: int,
    direction: int,
    outcome: str,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp | None = None,
) -> Any:
    """Build a minimal TradeOutcome."""
    from src.strategy.exits import TradeOutcome
    ets: pd.Timestamp | None
    if exit_ts is not None:
        ets = exit_ts
    elif outcome in ("tp", "sl", "undecided"):
        ets = entry_ts + pd.Timedelta(hours=3)
    else:
        ets = None

    return TradeOutcome(
        window_idx=window_idx,
        entry_ts=entry_ts,
        direction=direction,
        entry=500.0,
        sl=495.0 if direction == 1 else 505.0,
        tp=510.0 if direction == 1 else 490.0,
        outcome=outcome,
        exit_idx_in_future=2 if outcome in ("tp", "sl") else -1,
        exit_ts=ets,
        exit_price=510.0 if outcome == "tp" else 495.0,
        r_multiple=2.0 if outcome == "tp" else (-1.0 if outcome == "sl" else 0.0),
        filled=outcome not in ("no_fill",),
        swing_fallback=False,
    )


def _make_track(
    n_total_ts: int = 200,
    display_start: int = 60,   # index into total ts pool
    display_bars: int = 65,
    trade_at_window_idx: int | None = None,
    trade_direction: int = 1,
    trade_outcome: str = "tp",
    arch: str = "xgboost",
    label: str = "XGBoost · H1 · multisym · F1 0.738",
    bar56_high: float = 502.0,
    bar56_low: float = 498.0,
    bar58_high: float = 504.0,
    bar58_low: float = 500.5,
) -> ModelTrack:
    """Build a synthetic ModelTrack.

    Architecture:
    - all_ts = n_total_ts hourly timestamps
    - pred_ts = all_ts[60 : 60 + n_windows]  (last-bar timestamps)
    - bars = all_ts[display_start : display_start + display_bars]
    - n_windows = n_total_ts - 60

    For a zone to appear in the display window, pred_ts[window_idx] must be
    one of the display bar timestamps.  Since pred_ts[i] = all_ts[60+i] and
    display bars start at all_ts[display_start], window_idx = display_start - 60
    is the first index whose pred_ts falls in the display range.
    """
    all_ts = _make_timestamps(n_total_ts)
    n_windows = n_total_ts - 60
    pred_ts = all_ts[60 : 60 + n_windows]   # last-bar ts for each window

    bars_ts = all_ts[display_start : display_start + display_bars]
    assert len(bars_ts) == display_bars

    bars_data = pd.DataFrame(
        {
            "open":   np.linspace(500.0, 510.0, display_bars),
            "high":   np.linspace(502.0, 512.0, display_bars),
            "low":    np.linspace(498.0, 508.0, display_bars),
            "close":  np.linspace(501.0, 511.0, display_bars),
            "volume": np.ones(display_bars) * 1000.0,
        },
        index=bars_ts,
    )

    preds = np.zeros(n_windows, dtype=np.int64)
    probas = np.zeros((n_windows, 3), dtype=np.float32)
    probas[:, 0] = 1.0

    windows_raw = np.zeros((n_windows, 60, 5), dtype=np.float32)
    for i in range(n_windows):
        windows_raw[i] = _make_window_raw(bar56_high, bar56_low, bar58_high, bar58_low)

    trades: list = []
    if trade_at_window_idx is not None:
        widx = trade_at_window_idx
        preds[widx] = trade_direction
        probas[widx, 0] = 0.0
        probas[widx, trade_direction] = 1.0
        entry_ts = pred_ts[widx]
        t = _make_trade(widx, trade_direction, trade_outcome, entry_ts)
        trades.append(t)

    return ModelTrack(
        name=arch,
        label=label,
        bars=bars_data,
        pred_ts=pred_ts,
        preds=preds,
        probas=probas,
        trades=trades,
        windows_raw_all=windows_raw,
    )


# ---------------------------------------------------------------------------
# Convenience: first window index whose pred_ts is inside the display window
# ---------------------------------------------------------------------------

def _first_display_window_idx(display_start: int = 60) -> int:
    """Return window_idx whose pred_ts == display bars' first timestamp."""
    # pred_ts[i] = all_ts[60+i]; display starts at all_ts[display_start]
    return display_start - 60


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPayloadRoundTrip:
    """Test 1: json.dumps round-trips without error."""

    def test_basic_round_trip(self):
        track = _make_track()
        payload = build_demo_payload([track])
        dumped = json.dumps(payload)
        reloaded = json.loads(dumped)
        assert reloaded["window"]["start"] is not None
        assert isinstance(reloaded["frame_times"], list)
        assert len(reloaded["models"]) == 1

    def test_multi_model_round_trip(self):
        t1 = _make_track(arch="xgboost", label="XGB")
        t2 = _make_track(arch="lstm", label="LSTM")
        payload = build_demo_payload([t1, t2])
        dumped = json.dumps(payload)
        reloaded = json.loads(dumped)
        assert len(reloaded["models"]) == 2


class TestNoLookahead:
    """Test 2: every zone t_detect >= pattern-close bar timestamp.

    pred_ts[window_idx] is the window last-bar timestamp (bar 59 of the window,
    i.e. the N+2 reaction candle close). This is the detection boundary — zone
    is not revealed before this time.
    """

    def test_t_detect_equals_pred_ts(self):
        """t_detect must equal pred_ts for the window (the label bar timestamp)."""
        display_start = 80   # bars start at all_ts[80]
        widx = _first_display_window_idx(display_start)  # = 80-60 = 20
        track = _make_track(
            display_start=display_start,
            trade_at_window_idx=widx,
            trade_direction=1,
            trade_outcome="tp",
        )
        payload = build_demo_payload([track])
        zones = payload["models"][0]["zones"]
        assert len(zones) == 1, f"Expected 1 zone, got {len(zones)}"
        z = zones[0]

        all_ts = _make_timestamps(200)
        pred_ts = all_ts[60 : 60 + 140]
        expected_t_detect = _ms(pred_ts[widx])
        assert z["t_detect"] == expected_t_detect, (
            f"t_detect={z['t_detect']} != expected={expected_t_detect}"
        )

    def test_zone_before_display_window_excluded(self):
        """Window_idx=0 has pred_ts < display_start → zone must be excluded."""
        display_start = 80   # display starts at all_ts[80]
        # window_idx=0 pred_ts = all_ts[60], which is before all_ts[80]
        track = _make_track(
            display_start=display_start,
            trade_at_window_idx=0,  # pred_ts[0] = all_ts[60] < all_ts[80]
            trade_direction=1,
            trade_outcome="tp",
        )
        payload = build_demo_payload([track])
        zones = payload["models"][0]["zones"]
        assert len(zones) == 0, "Zone before display window must be excluded"


class TestZoneXOrdering:
    """Test 3: x0 <= x1 for all zones."""

    def test_bull_zone_x_ordering(self):
        widx = _first_display_window_idx(80)
        track = _make_track(display_start=80, trade_at_window_idx=widx, trade_direction=1,
                            trade_outcome="pending")
        payload = build_demo_payload([track])
        for z in payload["models"][0]["zones"]:
            assert z["x0"] <= z["x1"], f"x0={z['x0']} > x1={z['x1']}"

    def test_bear_zone_x_ordering(self):
        widx = _first_display_window_idx(80)
        track = _make_track(display_start=80, trade_at_window_idx=widx, trade_direction=2,
                            trade_outcome="sl")
        payload = build_demo_payload([track])
        for z in payload["models"][0]["zones"]:
            assert z["x0"] <= z["x1"], f"x0={z['x0']} > x1={z['x1']}"


class TestFrameTimes:
    """Test 4: frame_times = sorted unique H1 bar timestamps in window."""

    def test_frame_times_sorted(self):
        track = _make_track()
        payload = build_demo_payload([track])
        ft = payload["frame_times"]
        assert ft == sorted(ft), "frame_times not sorted"

    def test_frame_times_unique(self):
        track = _make_track()
        payload = build_demo_payload([track])
        ft = payload["frame_times"]
        assert len(ft) == len(set(ft)), "frame_times has duplicates"

    def test_frame_times_match_bar_times(self):
        track = _make_track(display_bars=65)
        payload = build_demo_payload([track])
        ft = payload["frame_times"]
        bar_ms_set = {b["t"] for b in payload["models"][0]["bars"]}
        assert set(ft) == bar_ms_set, "frame_times does not match bar timestamps"

    def test_frame_count_matches_display_window(self):
        track = _make_track(display_bars=65)
        payload = build_demo_payload([track])
        assert len(payload["frame_times"]) == 65


class TestZoneStateAndResolve:
    """Test 5: zone state/t_resolve match Trade.outcome."""

    @pytest.mark.parametrize("outcome,expected_state,has_resolve", [
        ("tp", "tp", True),
        ("sl", "sl", True),
        ("undecided", "timeout", True),
        ("no_future", "pending", False),
    ])
    def test_zone_state(self, outcome, expected_state, has_resolve):
        widx = _first_display_window_idx(80)
        track = _make_track(
            display_start=80,
            trade_at_window_idx=widx,
            trade_direction=1,
            trade_outcome=outcome,
        )
        payload = build_demo_payload([track])
        zones = payload["models"][0]["zones"]
        assert len(zones) == 1, f"outcome={outcome}: expected 1 zone, got {len(zones)}"
        z = zones[0]
        assert z["state"] == expected_state, (
            f"outcome={outcome}: state={z['state']!r} != {expected_state!r}"
        )
        if has_resolve:
            assert z["t_resolve"] is not None, f"outcome={outcome}: t_resolve should not be null"
        else:
            assert z["t_resolve"] is None, f"outcome={outcome}: t_resolve should be null"


class TestBarsWithinWindow:
    """Test 6: all bar timestamps fall within window start/end."""

    def test_bars_within_window_bounds(self):
        track = _make_track(display_bars=65)
        payload = build_demo_payload([track])
        window_start = payload["window"]["start"]
        window_end = payload["window"]["end"]
        for m in payload["models"]:
            for b in m["bars"]:
                assert b["t"] >= window_start, f"bar t={b['t']} before window start"
                assert b["t"] <= window_end, f"bar t={b['t']} after window end"

    def test_bar_count_matches_display_window(self):
        track = _make_track(display_bars=65)
        payload = build_demo_payload([track])
        for m in payload["models"]:
            assert len(m["bars"]) == 65


class TestDeterminism:
    """Test 7: byte-identical JSON on two calls."""

    def test_byte_identical_on_rerun(self):
        widx = _first_display_window_idx(80)
        track = _make_track(display_start=80, trade_at_window_idx=widx, trade_direction=1,
                            trade_outcome="tp")
        p1 = payload_to_json(build_demo_payload([track]))
        p2 = payload_to_json(build_demo_payload([track]))
        assert p1 == p2, "Two calls produced different JSON output"

    def test_sorted_keys_in_json(self):
        track = _make_track()
        j = payload_to_json(build_demo_payload([track]))
        first_key_positions = {}
        for key in ("footer", "frame_times", "models", "window"):
            first_key_positions[key] = j.index(f'"{key}"')
        sorted_by_pos = sorted(first_key_positions.items(), key=lambda x: x[1])
        keys_in_order = [k for k, _ in sorted_by_pos]
        assert keys_in_order == sorted(first_key_positions.keys()), (
            f"Top-level keys not sorted: {keys_in_order}"
        )


class TestEmptyModels:
    """Test 8: empty tracks list raises ValueError."""

    def test_empty_tracks_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_demo_payload([])


class TestZoneYOrdering:
    """Test 9: y0 <= y1 for all zones."""

    def test_bull_zone_y_ordering(self):
        # bull: y0 = bar_58.low = 500.5, y1 = bar_56.high = 502 → y0 < y1 ✓
        widx = _first_display_window_idx(80)
        track = _make_track(
            display_start=80,
            trade_at_window_idx=widx,
            trade_direction=1,
            trade_outcome="pending",
            bar56_high=502.0, bar58_low=500.5,
        )
        payload = build_demo_payload([track])
        for z in payload["models"][0]["zones"]:
            assert z["y0"] <= z["y1"], f"y0={z['y0']} > y1={z['y1']}"

    def test_bear_zone_y_ordering(self):
        # bear: y0 = bar_56.low = 498.0, y1 = bar_58.high = 504.0 → y0 < y1 ✓
        widx = _first_display_window_idx(80)
        track = _make_track(
            display_start=80,
            trade_at_window_idx=widx,
            trade_direction=2,
            trade_outcome="pending",
            bar56_low=498.0, bar58_high=504.0,
        )
        payload = build_demo_payload([track])
        for z in payload["models"][0]["zones"]:
            assert z["y0"] <= z["y1"], f"y0={z['y0']} > y1={z['y1']}"


class TestBuildMultiScenePayload:
    """Tests for build_multi_scene_payload: single-pass multi-scene assembly."""

    def _make_wide_track(self) -> ModelTrack:
        """Wide track covering 400 hourly timestamps (ample for 2 scene slices)."""
        return _make_track(
            n_total_ts=400,
            display_start=100,   # wide bars start at index 100
            display_bars=250,    # wide window = 250 bars
            trade_at_window_idx=50,
            trade_direction=1,
            trade_outcome="tp",
        )

    def _scenes(self) -> list[dict]:
        """Two non-overlapping scenes within the wide track's bar range."""
        all_ts = _make_timestamps(400)
        # Scene 1: bars 100-164 of wide track (first 65)
        # Scene 2: bars 165-229 of wide track (next 65)
        return [
            {
                "id": "scene_a",
                "label": "Scene A",
                "start": str(all_ts[100]),
                "end":   str(all_ts[164]),
            },
            {
                "id": "scene_b",
                "label": "Scene B",
                "start": str(all_ts[165]),
                "end":   str(all_ts[229]),
            },
        ]

    def test_top_level_structure(self):
        track = self._make_wide_track()
        payload = build_multi_scene_payload([track], self._scenes())
        assert "scenes" in payload
        assert "default_scene" in payload
        assert payload["default_scene"] == 0
        assert len(payload["scenes"]) == 2

    def test_scene_ids_and_labels(self):
        track = self._make_wide_track()
        payload = build_multi_scene_payload([track], self._scenes())
        assert payload["scenes"][0]["id"] == "scene_a"
        assert payload["scenes"][0]["label"] == "Scene A"
        assert payload["scenes"][1]["id"] == "scene_b"
        assert payload["scenes"][1]["label"] == "Scene B"

    def test_stats_keys_present(self):
        track = self._make_wide_track()
        payload = build_multi_scene_payload([track], self._scenes())
        for sc in payload["scenes"]:
            assert "stats" in sc
            assert set(sc["stats"].keys()) == {"tp", "sl", "timeout"}

    def test_scene_contains_payload_keys(self):
        track = self._make_wide_track()
        payload = build_multi_scene_payload([track], self._scenes())
        required_keys = {"window", "frame_times", "footer", "models"}
        for sc in payload["scenes"]:
            assert required_keys.issubset(sc.keys()), (
                f"Scene {sc['id']} missing keys: {required_keys - sc.keys()}"
            )

    def test_scene_bars_within_scene_window(self):
        track = self._make_wide_track()
        scenes = self._scenes()
        payload = build_multi_scene_payload([track], scenes)
        for sc in payload["scenes"]:
            w_start = sc["window"]["start"]
            w_end   = sc["window"]["end"]
            for m in sc["models"]:
                for b in m["bars"]:
                    assert b["t"] >= w_start, f"bar before scene window"
                    assert b["t"] <= w_end,   f"bar after scene window"

    def test_scenes_have_independent_bar_sets(self):
        track = self._make_wide_track()
        payload = build_multi_scene_payload([track], self._scenes())
        ft_a = payload["scenes"][0]["frame_times"]
        ft_b = payload["scenes"][1]["frame_times"]
        # Scene A starts before scene B: first timestamp of A < first timestamp of B
        assert ft_a[0] < ft_b[0], "Scene A should start before scene B"
        # Scene B's detection window starts after scene A's scene_end.
        # Note: scene A's frame_times may extend into the tail (past scene_end),
        # so we only assert the scene start ordering, not strict disjointness.
        all_ts = _make_timestamps(400)
        # Scene A detection end = all_ts[164]; scene B detection start = all_ts[165]
        scene_a_detect_end_ms = int(all_ts[164].value // 1_000_000)
        scene_b_start_ms = ft_b[0]
        assert scene_b_start_ms > scene_a_detect_end_ms, (
            "Scene B must start after scene A's detection window ends"
        )

    def test_stats_counts_non_negative(self):
        track = self._make_wide_track()
        payload = build_multi_scene_payload([track], self._scenes())
        for sc in payload["scenes"]:
            s = sc["stats"]
            assert s["tp"] >= 0 and s["sl"] >= 0 and s["timeout"] >= 0

    def test_empty_wide_tracks_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_multi_scene_payload([], self._scenes())

    def test_empty_scenes_raises(self):
        track = self._make_wide_track()
        with pytest.raises(ValueError, match="non-empty"):
            build_multi_scene_payload([track], [])

    def test_round_trips_json(self):
        track = self._make_wide_track()
        payload = build_multi_scene_payload([track], self._scenes())
        import json
        dumped = json.dumps(payload)
        reloaded = json.loads(dumped)
        assert len(reloaded["scenes"]) == 2


class TestPendingNullResolve:
    """Test 10: pending zones have t_resolve=null."""

    def test_no_future_zone_null_resolve(self):
        widx = _first_display_window_idx(80)
        track = _make_track(display_start=80, trade_at_window_idx=widx,
                            trade_direction=1, trade_outcome="no_future")
        payload = build_demo_payload([track])
        zones = payload["models"][0]["zones"]
        assert len(zones) == 1
        assert zones[0]["t_resolve"] is None
        assert zones[0]["state"] == "pending"

    def test_zone_with_no_exit_ts_is_pending(self):
        from src.strategy.exits import TradeOutcome
        widx = _first_display_window_idx(80)
        all_ts = _make_timestamps(200)
        pred_ts = all_ts[60 : 60 + 140]

        trade = TradeOutcome(
            window_idx=widx,
            entry_ts=pred_ts[widx],
            direction=1,
            entry=500.0, sl=495.0, tp=510.0,
            outcome="no_future",
            exit_idx_in_future=-1,
            exit_ts=None,
            exit_price=500.0,
            r_multiple=0.0,
            filled=True,
            swing_fallback=False,
        )

        track = _make_track(display_start=80, trade_at_window_idx=widx,
                            trade_direction=1, trade_outcome="no_future")
        # Replace the auto-generated trade with our explicit one
        track = ModelTrack(
            name=track.name,
            label=track.label,
            bars=track.bars,
            pred_ts=track.pred_ts,
            preds=track.preds,
            probas=track.probas,
            trades=[trade],
            windows_raw_all=track.windows_raw_all,
        )
        payload = build_demo_payload([track])
        zones = payload["models"][0]["zones"]
        assert len(zones) == 1
        assert zones[0]["t_resolve"] is None


class TestZoneConf:
    """Test 11: every zone has a conf field in [0.0, 1.0]."""

    def test_conf_field_present_and_in_range(self):
        """All zones must have a 'conf' key with value in [0.0, 1.0]."""
        widx = _first_display_window_idx(80)
        track = _make_track(
            display_start=80,
            trade_at_window_idx=widx,
            trade_direction=1,
            trade_outcome="tp",
        )
        payload = build_demo_payload([track])
        zones = payload["models"][0]["zones"]
        assert len(zones) == 1, "Expected 1 zone"
        z = zones[0]
        assert "conf" in z, "Zone must have a 'conf' field"
        assert isinstance(z["conf"], float), f"conf must be float, got {type(z['conf'])}"
        assert 0.0 <= z["conf"] <= 1.0, f"conf={z['conf']} out of [0, 1]"

    def test_conf_matches_probas(self):
        """conf must equal max(probas[widx][1], probas[widx][2])."""
        widx = _first_display_window_idx(80)
        track = _make_track(
            display_start=80,
            trade_at_window_idx=widx,
            trade_direction=1,
            trade_outcome="tp",
        )
        # In _make_track, when trade_at_window_idx is set:
        #   probas[widx, 0] = 0.0, probas[widx, trade_direction] = 1.0
        # So for direction=1: probas[widx] = [0, 1, 0] -> conf = max(1.0, 0.0) = 1.0
        payload = build_demo_payload([track])
        z = payload["models"][0]["zones"][0]
        expected_conf = float(max(track.probas[widx][1], track.probas[widx][2]))
        assert z["conf"] == expected_conf, f"conf={z['conf']} != expected={expected_conf}"

    def test_conf_all_zones_in_range(self):
        """All zones across multiple models must have conf in [0.0, 1.0]."""
        t1 = _make_track(arch="xgboost", label="XGB",
                         display_start=80, trade_at_window_idx=_first_display_window_idx(80),
                         trade_direction=1, trade_outcome="tp")
        t2 = _make_track(arch="lstm", label="LSTM",
                         display_start=80, trade_at_window_idx=_first_display_window_idx(80),
                         trade_direction=2, trade_outcome="sl")
        payload = build_demo_payload([t1, t2])
        for m in payload["models"]:
            for z in m["zones"]:
                assert "conf" in z, f"Zone in {m['name']} missing 'conf'"
                assert 0.0 <= z["conf"] <= 1.0, (
                    f"conf={z['conf']} out of range in {m['name']}"
                )

    def test_conf_in_multi_scene_payload(self):
        """Zones in multi-scene payload must also carry conf in [0.0, 1.0]."""
        track = _make_track(
            n_total_ts=400,
            display_start=100,
            display_bars=250,
            trade_at_window_idx=50,
            trade_direction=1,
            trade_outcome="tp",
        )
        all_ts = _make_timestamps(400)
        scenes = [
            {
                "id": "scene_a",
                "label": "Scene A",
                "start": str(all_ts[100]),
                "end":   str(all_ts[164]),
            },
        ]
        payload = build_multi_scene_payload([track], scenes)
        for sc in payload["scenes"]:
            for m in sc["models"]:
                for z in m["zones"]:
                    assert "conf" in z, f"Zone in multi-scene {m['name']} missing 'conf'"
                    assert 0.0 <= z["conf"] <= 1.0, (
                        f"conf={z['conf']} out of range in multi-scene {m['name']}"
                    )
