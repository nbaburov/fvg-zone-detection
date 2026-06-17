"""demo_animator.py — CLI for the FVG detection bar-by-bar demo animation.

Generates a self-contained offline HTML that replays real SPY H1 windows
bar-by-bar, showing how each best-per-arch model draws FVG zones live, with
zones recoloring on outcome.

Four scan-selected scenes (each ~65 H1 bars), selectable via scene buttons.
Chosen for more/larger visible gaps, all 4 models firing, and to contrast the
two lead models (XGBoost vs CNN-LSTM):
  0. Good:      2024-03-07 to 2024-03-20 (both models do well; ~14 held, 5 failed)
  1. XGB edge:  2025-03-14 to 2025-03-27 (XGBoost holds 3, CNN-LSTM holds 0)
  2. CNN edge:  2024-04-15 to 2024-04-26 (CNN-LSTM holds 2, XGBoost holds 0)
  3. Bad:       2025-06-06 to 2025-06-20 (both struggle; 0 held, 13 failed)

SINGLE INFERENCE PASS: build_model_tracks is called once over the full wide
window spanning all three scenes; each scene then slices from the resulting
wide tracks without re-running inference.

NOTE: scenes are NOT chronologically ordered (good=2024-03 is earliest,
bad=2025-06 is latest). _WIDE_START = min(scene starts), _WIDE_END is set
well past the latest scene end so every scene has a full 20-bar resolution tail.

Models (H1 multisym, best seed-0 checkpoint per arch):
  XGBoost    F1 0.738   checkpoints/xgboost_h1_multisym/
  CNN-LSTM   F1 0.675   checkpoints/cnn_lstm_h1_multisym/  (best-ever 5m-tuned=0.713)
  LSTM       F1 0.640   checkpoints/lstm_h1_multisym/
  Transformer F1 0.577  checkpoints/transformer_h1_multisym/
  xLSTM: omitted - no saved checkpoint (F1 0.369, underperformed)

Usage:
  python scripts/demo_animator.py
  python scripts/demo_animator.py --out reports/demo/fvg_animation.html
  python scripts/demo_animator.py --models xgboost cnn_lstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Model table (hardcoded)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent

_DEFAULT_OUT = _REPO_ROOT / "reports" / "demo" / "fvg_animation.html"

# ---------------------------------------------------------------------------
# Hand-picked scenes (~65 H1 bars each)
# ---------------------------------------------------------------------------
# Each scene window chosen via scan: larger/visible gaps, clean outcomes,
# low noise, all 4 models fire. Labels use plain hyphens/colons (no em-dashes).
#
# Per-model contrast is the point: good = both win, xgb_edge = XGBoost holds
# where CNN-LSTM misses, cnn_edge = CNN-LSTM catches winners XGBoost misses,
# bad = both struggle. ~13-20 gaps each, all 4 models fire.
#
# NOTE: scenes are NOT chronologically ordered.
#   good = 2024-03 (earliest), cnn_edge = 2024-04, xgb_edge = 2025-03, bad = 2025-06 (latest).
# Inference runs ONCE over the wide window spanning all three scenes
# (_WIDE_START = min of scene starts = good 2024-03-07;
#  _WIDE_END   = 2025-07-31, ~6 weeks past the latest scene end = bad 2025-06-20).
# Each scene slices from the wide tracks; no re-inference per scene.
# Default scene = good (index 0).
SCENES = [
    {
        "id": "good",
        "label": "Both models do well: gaps mostly held",
        "start": "2024-03-07T11:30:00-05:00",
        "end":   "2024-03-20T12:30:00-04:00",
    },
    {
        "id": "xgb_edge",
        "label": "XGBoost edge: its picks hold, CNN-LSTM's miss",
        "start": "2025-03-14T10:30:00-04:00",
        "end":   "2025-03-27T11:30:00-04:00",
    },
    {
        "id": "cnn_edge",
        "label": "CNN-LSTM edge: catches gaps XGBoost misses",
        "start": "2024-04-15T13:30:00-04:00",
        "end":   "2024-04-26T14:30:00-04:00",
    },
    {
        "id": "bad",
        "label": "Tough tape: both models struggle",
        "start": "2025-06-06T12:30:00-04:00",
        "end":   "2025-06-20T13:30:00-04:00",
    },
]

# Wide inference window: EARLIEST across all scenes (good 2024-03-07) to
# well past the LATEST scene end (bad 2025-06-20). Using min(scene starts)
# so the wide pass covers all 4 scenes regardless of their order in the list.
# _WIDE_END is set ~6 weeks past bad's end to give a full 20-bar resolution tail.
_WIDE_START = min(s["start"] for s in SCENES)  # 2024-03-07 = earliest scene (good)
_WIDE_END   = "2025-07-31T16:00:00-04:00"      # ~6 weeks past latest scene end (bad 2025-06-20)

# All 4 H1-multisym models (xLSTM excluded — no checkpoint)
_ALL_MODEL_SPECS = {
    "xgboost": {
        "arch": "xgboost",
        "tf": "h1",
        "dataset": "multisym",
        "tuned": False,
        "label": "XGBoost · H1 · multi-symbol · F1 0.738",
    },
    "cnn_lstm": {
        "arch": "cnn_lstm",
        "tf": "h1",
        "dataset": "multisym",
        "tuned": False,
        "label": "CNN-LSTM · H1 · multi-symbol · F1 0.675 (best-ever 5m-tuned 0.713)",
    },
    "lstm": {
        "arch": "lstm",
        "tf": "h1",
        "dataset": "multisym",
        "tuned": False,
        "label": "LSTM · H1 · multi-symbol · F1 0.640",
    },
    "transformer": {
        "arch": "transformer",
        "tf": "h1",
        "dataset": "multisym",
        "tuned": False,
        "label": "Transformer · H1 · multi-symbol · F1 0.577 (untuned)",
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate FVG detection bar-by-bar multi-scene animation HTML."
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Output HTML path (default: reports/demo/fvg_animation.html)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(_ALL_MODEL_SPECS.keys()),
        default=list(_ALL_MODEL_SPECS.keys()),
        help="Which models to include (default: all 4)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from src.demo.predict import ModelSpec, build_model_tracks
    from src.demo.render import read_vendored_echarts, write_demo_html
    from src.demo.serialize import build_multi_scene_payload

    # Build ModelSpec list in the order given
    specs = [
        ModelSpec(**_ALL_MODEL_SPECS[name])
        for name in args.models
    ]

    print("=" * 60)
    print(f"FVG Demo Animator  (multi-scene, {len(SCENES)} scenes)")
    print("=" * 60)
    print(f"Scenes:  {len(SCENES)} ({', '.join(s['id'] for s in SCENES)})")
    print(f"Wide window: {_WIDE_START}  ->  {_WIDE_END}")
    print(f"Models:  {', '.join(args.models)}")
    print(f"Output:  {args.out}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: single inference pass over the wide window
    # -----------------------------------------------------------------------
    print("Phase 1/3 — Running inference (single pass over wide window)...")
    wide_tracks = build_model_tracks(
        window_start=_WIDE_START,
        window_end=_WIDE_END,
        model_specs=specs,
        lookahead_bars=20,
    )

    print(f"  Wide windows built: {len(wide_tracks[0].pred_ts)}")
    print()

    # -----------------------------------------------------------------------
    # Phase 2: build multi-scene payload (slices from wide tracks, no re-inference)
    # -----------------------------------------------------------------------
    print("Phase 2/3 — Building multi-scene payload (slicing from wide tracks)...")
    payload = build_multi_scene_payload(wide_tracks, SCENES)

    for sc in payload["scenes"]:
        stats = sc["stats"]
        bars_count = len(sc["frame_times"])
        total_zones = sum(len(m["zones"]) for m in sc["models"])
        print(
            f"  [{sc['id']:12s}]  bars={bars_count}  zones={total_zones}  "
            f"tp={stats['tp']}  sl={stats['sl']}  timeout={stats['timeout']}"
        )
        for m in sc["models"]:
            n_bull = sum(1 for z in m["zones"] if z["dir"] == 1)
            n_bear = sum(1 for z in m["zones"] if z["dir"] == 2)
            tp_m = sum(1 for z in m["zones"] if z["state"] == "tp")
            sl_m = sum(1 for z in m["zones"] if z["state"] == "sl")
            print(
                f"    {m['name']:12s}  zones={len(m['zones'])}  "
                f"(bull={n_bull}, bear={n_bear})  tp={tp_m}  sl={sl_m}"
            )
    print()

    # -----------------------------------------------------------------------
    # Phase 3: render
    # -----------------------------------------------------------------------
    print("Phase 3/3 — Reading ECharts + writing HTML...")
    echarts_js = read_vendored_echarts()
    out_path = Path(args.out)
    write_demo_html(payload, out_path, echarts_js=echarts_js)

    print(f"  Written: {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print()
    print("Done. Open in a browser for the animation.")
    print()
    print("Note: xLSTM omitted - F1 0.369 (underperformed), no saved checkpoint.")


if __name__ == "__main__":
    main()
