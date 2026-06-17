"""serialize.py — Build the JSON payload consumed by the ECharts demo driver.

This is the tested core.  Pure, no IO, no ECharts dependency.

JSON schema
-----------
{
  "window": {"start": <ms>, "end": <ms>},
  "frame_times": [<ms>, ...],          # sorted H1 bar-close times in window
  "footer": "...",
  "models": [
    {
      "name": "...", "label": "...",
      "bars":  [{"t":<ms>,"o":.,"h":.,"l":.,"c":.}, ...],
      "zones": [
        {
          "dir": 1|2,
          "y0": .., "y1": ..,            # gap price bounds
          "x0": <ms>, "x1": <ms>,        # pattern span (bar_56.open to bar_58.close)
          "t_detect": <ms>,              # timestamp of bar_58 (pattern-close bar)
          "t_resolve": <ms> | null,      # when outcome known; null if pending
          "state": "tp"|"sl"|"timeout"|"pending",
          "conf": float                  # model softmax conviction: max(p_bull, p_bear)
        }, ...
      ]
    }, ...
  ]
}

Gap geometry (from exits.py conventions):
  bar_56 = windows_raw[i][56]  candle-1 (pre-impulse)
  bar_57 = windows_raw[i][57]  candle-2 (impulse)
  bar_58 = windows_raw[i][58]  candle-3 (reaction / gap near-edge)
  Column order: [open=0, high=1, low=2, close=3, volume=4]

  bull FVG gap = [bar_58.low, bar_56.high]  → y0=bar_58.low, y1=bar_56.high
  bear FVG gap = [bar_56.low, bar_58.high]  → y0=bar_56.low, y1=bar_58.high

Pattern x-span: x0 = bar_56 timestamp, x1 = bar_58 timestamp.
The window last-bar timestamp (pred_ts[i]) = bar_59 timestamp for a 60-bar
window where bar indices 0..59 map to df rows, and the label bar is row 59
(N+2).  Bar_58 is the reaction candle at position -2 from the window end;
bar_59 is the window's last bar (index 59), which is when the label first
becomes knowable (N+2 close).

Detection time = pred_ts[i] = timestamp of the window's last bar (bar index 59),
which is the N+2 reaction candle.  The pattern closes at bar_58 (index 58), but
the label (and prediction) is knowable at bar 59 close.  So t_detect = pred_ts[i].
This is the correct no-lookahead boundary: zone is not shown until bar 59 closes.

Outcome state mapping:
  "tp"        → state="tp",      t_resolve = exit_ts
  "sl"        → state="sl",      t_resolve = exit_ts
  "undecided" → state="timeout", t_resolve = exit_ts (last bar checked)
  "no_future" → state="pending", t_resolve = null
  "no_fill"   → skipped (limit-order miss; we use fixed_2r so won't occur)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.demo.predict import ModelTrack

FOOTER = (
    "xLSTM underperformed (F1 0.369), no saved checkpoint, not shown"
    " · SPY test 2023 to 2025, unseen data"
)

_OUTCOME_TO_STATE = {
    "tp": "tp",
    "sl": "sl",
    "undecided": "timeout",
    "no_future": "pending",
    "no_fill": "pending",  # fallback; fixed_2r never emits no_fill
}


def _ms(ts: "pd.Timestamp | None") -> "int | None":
    """Convert a Timestamp to epoch milliseconds (int). None stays None."""
    if ts is None or pd.isna(ts):
        return None
    return int(ts.value // 1_000_000)


def _bar_dict(row: pd.Series, ts: pd.Timestamp) -> dict[str, Any]:
    return {
        "t": _ms(ts),
        "o": float(row["open"]),
        "h": float(row["high"]),
        "l": float(row["low"]),
        "c": float(row["close"]),
    }


def _zone_for_trade(
    trade: Any,
    pred_ts_i: pd.Timestamp,
    window_raw_i: np.ndarray,
    ts_all: pd.DatetimeIndex,
    window_idx: int,
    window_start_ts: pd.Timestamp | None = None,
    conf: float = 0.0,
) -> dict[str, Any] | None:
    """Build a zone dict from a single Trade.

    Parameters
    ----------
    trade : TradeOutcome
    pred_ts_i : pd.Timestamp
        Window last-bar timestamp (t_detect).
    window_raw_i : np.ndarray, shape (60, 5)
        Raw OHLCV for this window.
    ts_all : pd.DatetimeIndex
        All window timestamps (for x0/x1 time lookup).
    window_idx : int
        Index of this window in ts_all (for bar_56/bar_58 timestamp lookup).
    window_start_ts : pd.Timestamp | None
        First bar timestamp of the display window.  When provided, x0 and x1
        are clamped to be >= this value so zones whose pattern precedes the
        display window do not shift the ECharts x-axis left and break
        time-alignment across the 4 facets (Fix A).
    conf : float
        Model softmax conviction for this zone: max(p_bull, p_bear).
    """
    direction = int(trade.direction)

    bar_56 = window_raw_i[56]
    bar_58 = window_raw_i[58]

    # Gap price bounds
    if direction == 1:
        y0 = float(bar_58[2])   # bar_58.low
        y1 = float(bar_56[1])   # bar_56.high
    else:
        y0 = float(bar_56[2])   # bar_56.low
        y1 = float(bar_58[1])   # bar_58.high

    if y0 > y1:
        y0, y1 = y1, y0  # ensure y0 <= y1

    # x-span timestamps: bar_56 is 3 bars before the last bar (index 56 of 60-bar window)
    # pred_ts_i = last-bar timestamp (bar 59, index 59); bar_56 is pred_ts_i - 3 steps in ts_all.
    # We use position window_idx in ts_all to back out bar-level timestamps:
    #   ts_all[window_idx] = window last bar (bar 59)
    #   bar_56 timestamp = ts_all[window_idx - 3]  (if within bounds)
    #   bar_58 timestamp = ts_all[window_idx - 1]
    # Assumes stride=1 (consecutive windows), which is enforced by the runner call in predict.py.
    n_ts = len(ts_all)
    idx_56 = window_idx - 3
    idx_58 = window_idx - 1

    x0_ts = ts_all[idx_56] if 0 <= idx_56 < n_ts else pred_ts_i
    x1_ts = ts_all[idx_58] if 0 <= idx_58 < n_ts else pred_ts_i

    # Fix A: clamp x0/x1 so they are >= display window start.  A zone detected
    # near the start of the display window may have its 3-bar pattern (bar_56–
    # bar_58) extending BEFORE the window, which shifts the ECharts x-axis left
    # on that one facet and breaks time-alignment across all 4 panels.  Clamping
    # ensures every zone starts at or after the left edge of the display window.
    if window_start_ts is not None:
        if x0_ts < window_start_ts:
            x0_ts = window_start_ts
        if x1_ts < window_start_ts:
            x1_ts = window_start_ts
        # Maintain x0 <= x1 after clamping
        if x0_ts > x1_ts:
            x0_ts = x1_ts

    # Outcome state
    outcome = trade.outcome
    state = _OUTCOME_TO_STATE.get(outcome, "pending")

    # Resolve time
    exit_ts = getattr(trade, "exit_ts", None)
    if state in ("tp", "sl"):
        t_resolve = _ms(exit_ts)
    elif state == "timeout":
        t_resolve = _ms(exit_ts)
    else:
        t_resolve = None

    return {
        "dir": direction,
        "y0": y0,
        "y1": y1,
        "x0": _ms(x0_ts),
        "x1": _ms(x1_ts),
        "t_detect": _ms(pred_ts_i),
        "t_resolve": t_resolve,
        "state": state,
        "conf": float(max(0.0, min(1.0, conf))),
    }


def _build_zones_for_model(
    track: ModelTrack,
    window_ts_in_display: set,
    window_start_ts: pd.Timestamp | None = None,
) -> list[dict]:
    """Build zone list for a model, restricted to zones detectable within the display window.

    A zone is included iff pred_ts (t_detect) is within the display window.
    window_start_ts is passed through to _zone_for_trade for x-axis clamping (Fix A).
    """
    # Build a lookup: window_idx -> trade (only one trade per window, since
    # compute_trades_for_model iterates windows and skips pred==0)
    trade_by_widx: dict[int, Any] = {}
    for trade in track.trades:
        widx = int(trade.window_idx)
        trade_by_widx[widx] = trade

    zones = []
    ts_all = track.pred_ts  # Fix E: was window_ts_all (alias removed from dataclass)
    probas = track.probas   # shape (N, 3): index 1=bull, 2=bear

    for widx, pred_ts in enumerate(ts_all):
        pred_ts_tz = pred_ts
        if pred_ts_tz not in window_ts_in_display:
            continue
        if widx not in trade_by_widx:
            continue
        trade = trade_by_widx[widx]
        raw_i = track.windows_raw_all[widx]

        # Compute model conviction: max(p_bull, p_bear) for this window
        if 0 <= widx < len(probas):
            conf_val = float(max(probas[widx][1], probas[widx][2]))
        else:
            conf_val = 0.0

        zone = _zone_for_trade(trade, pred_ts_tz, raw_i, ts_all, widx,
                               window_start_ts=window_start_ts,
                               conf=conf_val)
        if zone is not None:
            zones.append(zone)

    return zones


def build_demo_payload(tracks: list[ModelTrack]) -> dict[str, Any]:
    """Build the full JSON payload for the ECharts demo driver.

    Parameters
    ----------
    tracks : list[ModelTrack]
        One track per model, as returned by predict.build_model_tracks().

    Returns
    -------
    dict
        Fully serializable dict (no Timestamp / ndarray objects).
        Deterministic: same input → same output; no RNG, no clock.
    """
    if not tracks:
        raise ValueError("tracks must be non-empty")

    # The display window is defined by the bars in the first track
    # (all tracks share the same bars DataFrame since they use the same window)
    bars_df = tracks[0].bars
    bar_times = bars_df.index  # DatetimeIndex

    window_start_ms = _ms(bar_times[0])
    window_end_ms = _ms(bar_times[-1])
    window_start_ts = bar_times[0]  # Fix A: pass to zone builder for x-axis clamping

    # frame_times = sorted unique H1 bar timestamps in the display window
    # These are the "reveal clock" ticks: JS shows bars/zones with t <= frame_t
    frame_times = sorted({_ms(ts) for ts in bar_times})

    # Set of display-window bar timestamps for fast membership test
    window_ts_set = set(bar_times)

    models_payload = []
    for track in tracks:
        # Bars within the display window
        bars_list = [
            _bar_dict(row, ts)
            for ts, row in track.bars.iterrows()
        ]

        zones = _build_zones_for_model(track, window_ts_set, window_start_ts=window_start_ts)

        models_payload.append({
            "name": track.name,
            "label": track.label,
            "bars": bars_list,
            "zones": zones,
        })

    payload: dict[str, Any] = {
        "window": {"start": window_start_ms, "end": window_end_ms},
        "frame_times": frame_times,
        "footer": FOOTER,
        "models": models_payload,
    }

    return payload


_RESOLUTION_TAIL_BARS = 20  # extra display bars after scene_end so late gaps resolve


def build_multi_scene_payload(
    wide_tracks: list[ModelTrack],
    scenes: list[dict],
) -> dict[str, Any]:
    """Build a multi-scene payload from a single wide-inference pass.

    Parameters
    ----------
    wide_tracks : list[ModelTrack]
        Tracks produced by a single build_model_tracks() call over a wide
        window covering all scenes.  Each track's pred_ts/preds/probas/trades
        span the full wide range; only .bars needs to be sliced per scene.
    scenes : list[dict]
        Each dict must have keys:
          id       : str  — unique scene identifier
          label    : str  — human-readable scene label
          start    : str  — ISO timestamp for scene window start
          end      : str  — ISO timestamp for scene window end
        The timezone of start/end must be compatible with the track index tz.

    Returns
    -------
    dict
        {
          "scenes": [
            {
              "id": ...,
              "label": ...,
              "stats": {"tp": N, "sl": N, "timeout": N},
              <all build_demo_payload keys: window, frame_times, footer, models>
            }, ...
          ],
          "default_scene": 0
        }

    Resolution tail
    ---------------
    Each scene's display bars are extended by _RESOLUTION_TAIL_BARS bars past
    the scene's end timestamp so that gaps detected near the scene boundary
    have time to resolve visually.  Zones are still only DETECTED within the
    original scene window (t_detect <= scene_end); the tail bars just extend
    the animation so outcomes can play out on screen.
    """
    if not wide_tracks:
        raise ValueError("wide_tracks must be non-empty")
    if not scenes:
        raise ValueError("scenes must be non-empty")

    from dataclasses import replace as _dc_replace

    tz = wide_tracks[0].bars.index.tz
    wide_bar_index = wide_tracks[0].bars.index  # shared bar index across all tracks

    scene_payloads = []
    for scene in scenes:
        ts_start = pd.Timestamp(scene["start"])
        ts_start = ts_start.tz_localize(tz) if ts_start.tzinfo is None else ts_start.tz_convert(tz)
        ts_end = pd.Timestamp(scene["end"])
        ts_end = ts_end.tz_localize(tz) if ts_end.tzinfo is None else ts_end.tz_convert(tz)

        # Find where scene_end falls in the wide bar index and extend by tail bars
        try:
            end_pos = wide_bar_index.get_loc(ts_end)
        except KeyError:
            # ts_end not exact match; find the rightmost position <= ts_end
            end_pos = wide_bar_index.searchsorted(ts_end, side="right") - 1
        tail_end_pos = min(end_pos + _RESOLUTION_TAIL_BARS, len(wide_bar_index) - 1)
        ts_end_tail = wide_bar_index[tail_end_pos]

        # Slice bars for this scene from the wide DataFrame, including the tail
        scene_tracks: list[ModelTrack] = []
        for track in wide_tracks:
            scene_bars = track.bars.loc[ts_start:ts_end_tail]
            # Replace only .bars; keep pred_ts/preds/probas/trades/windows_raw_all
            # so serialize can filter zones by the scene's bar timestamps.
            scene_track = _dc_replace(track, bars=scene_bars)
            scene_tracks.append(scene_track)

        scene_payload = build_demo_payload(scene_tracks)

        # Post-filter each model's zones to only those detected within the
        # original scene window (t_detect <= _ms(ts_end)).  The tail bars are
        # display-only; we do not want new gap detections in the tail.
        ts_end_ms = _ms(ts_end)
        for m in scene_payload["models"]:
            m["zones"] = [z for z in m["zones"] if z["t_detect"] <= ts_end_ms]

        # Tally tp/sl/timeout counts across all models (from filtered zones)
        tp_count = sl_count = timeout_count = 0
        for m in scene_payload["models"]:
            for z in m["zones"]:
                if z["state"] == "tp":
                    tp_count += 1
                elif z["state"] == "sl":
                    sl_count += 1
                elif z["state"] == "timeout":
                    timeout_count += 1

        scene_entry: dict[str, Any] = {
            "id": scene["id"],
            "label": scene["label"],
            "stats": {"tp": tp_count, "sl": sl_count, "timeout": timeout_count},
        }
        scene_entry.update(scene_payload)
        scene_payloads.append(scene_entry)

    return {
        "scenes": scene_payloads,
        "default_scene": 0,
    }


def payload_to_json(payload: dict[str, Any]) -> str:
    """Serialize payload to deterministic JSON string (sorted keys, no indent)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
