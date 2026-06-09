"""D9 depth probe — gate before the Phase 4 multi-symbol full pull.

Answers two questions cheaply (short windows, no multi-year pull) BEFORE committing
to the multi-day historical pull + training:

  1. Reachability — does each candidate symbol's 1-min history reach back to 2016
     (SPY's start), or is it shallower?
  2. Usable-bar multiplier — after the full validation pipeline (RTH filter, H1
     resample, zero-volume + OHLC-integrity drop), how many usable H1 bars does each
     symbol yield per month vs SPY? This converts the plan's optimistic "~4x" into a
     measured number, exposing low-liquidity symbols (e.g. DIA) before they silently
     skew the pool.

Reuses src.data.download.download_h1 (so the probe validates bars exactly like the real
pipeline will), but points the minute cache at a throwaway scratch dir so the canonical
data/raw/ cache is never partially filled ahead of the real pull.

Run (needs live Alpaca creds in .env):
    python scripts/data/depth_probe.py

Decision per .nb/plan/09-Jun-26/multi-symbol-expansion.md (D9 gate):
  - all symbols reach 2016 + comparable usable bars  -> proceed with full pull
  - a symbol is shallow / much lower usable count     -> drop it, or truncate ALL
                                                          symbols to the shortest common
                                                          history. Decide here, record it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

# Make the repo root importable when run as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download import download_h1  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
BASELINE = "SPY"

# Short windows (< 365 days each, so _sanity_check_bar_count is exempt).
# EARLY probes 2016 reachability; RECENT probes current liquidity.
EARLY = ("2016-01-04", "2016-01-29")
RECENT = ("2025-04-01", "2025-04-30")

SCRATCH = ROOT / "data" / "raw" / "_probe"
REPORT_DIR = ROOT / "reports" / "rigor" / "09-Jun-26"


def _probe_window(symbol: str, start: str, end: str, tag: str) -> dict:
    """Pull one short window for *symbol*, return usable-H1 diagnostics.

    Never touches the canonical data/raw cache — uses a per-symbol scratch path and
    use_cache=False so each call hits the API fresh and writes only to scratch.
    """
    cache_path = SCRATCH / f"{symbol.lower()}_{tag}.parquet"
    try:
        h1 = download_h1(
            symbol,
            start=start,
            end=end,
            use_cache=False,
            cache_path=str(cache_path),
        )
    except Exception as exc:  # noqa: BLE001 - probe must not abort on one symbol
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if h1 is None or len(h1) == 0:
        return {"ok": False, "error": "no bars returned"}

    return {
        "ok": True,
        "usable_h1": int(len(h1)),
        "first_bar": str(h1.index.min()),
        "last_bar": str(h1.index.max()),
    }


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    print(f"D9 depth probe — symbols {SYMBOLS}")
    print(f"  early window  {EARLY[0]} .. {EARLY[1]}  (2016 reachability)")
    print(f"  recent window {RECENT[0]} .. {RECENT[1]}  (current liquidity)\n")

    for sym in SYMBOLS:
        early = _probe_window(sym, EARLY[0], EARLY[1], "early")
        recent = _probe_window(sym, RECENT[0], RECENT[1], "recent")
        results[sym] = {"early": early, "recent": recent}
        reach = "YES" if early["ok"] else f"NO ({early['error']})"
        e_cnt = early.get("usable_h1", 0)
        r_cnt = recent.get("usable_h1", 0)
        first = early.get("first_bar", "-")
        print(f"  {sym:4s}  2016-reach={reach:24s}  early_H1={e_cnt:4d}  recent_H1={r_cnt:4d}  first={first}")

    # Relative usable-bar multiplier vs SPY (recent window = like-for-like liquidity).
    spy_recent = results[BASELINE]["recent"].get("usable_h1", 0) or 1
    spy_early = results[BASELINE]["early"].get("usable_h1", 0) or 1

    print("\nUsable-H1 ratio vs SPY (1.0 = same as SPY for that window):")
    summary: dict[str, dict] = {}
    for sym in SYMBOLS:
        r = results[sym]["recent"].get("usable_h1", 0) / spy_recent
        e = results[sym]["early"].get("usable_h1", 0) / spy_early
        reaches_2016 = results[sym]["early"]["ok"]
        summary[sym] = {
            "reaches_2016": reaches_2016,
            "recent_ratio_vs_spy": round(r, 3),
            "early_ratio_vs_spy": round(e, 3),
        }
        print(f"  {sym:4s}  recent={r:5.2f}x  early={e:5.2f}x  reaches_2016={reaches_2016}")

    # Verdict hint (human decides the final call).
    shallow = [s for s in SYMBOLS if not summary[s]["reaches_2016"]]
    low_liq = [s for s in SYMBOLS if s != BASELINE and summary[s]["recent_ratio_vs_spy"] < 0.8]
    print("\nVERDICT HINT (you decide):")
    if not shallow and not low_liq:
        print("  All symbols reach 2016 and have comparable liquidity -> PROCEED with full pull.")
    else:
        if shallow:
            print(f"  Shallow (no 2016 history): {shallow} -> drop, or truncate ALL to shortest common history.")
        if low_liq:
            print(f"  Lower liquidity (<0.8x SPY usable bars): {low_liq} -> expect a smaller real multiplier; keep or drop.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "depth_probe.json"
    payload = {
        "symbols": SYMBOLS,
        "early_window": EARLY,
        "recent_window": RECENT,
        "raw": results,
        "summary": summary,
    }
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"\nReport written: {report_path}")

    # Clean up scratch minute caches so they can't be mistaken for the real pull.
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
