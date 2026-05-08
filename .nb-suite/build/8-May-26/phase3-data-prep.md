# Build Log — Phase 3 Data Preparation
Date: 8-May-26
Scope: Foundations 1–9 (Foundation 10 = nb-notebook, deferred)
Mode: /nb:build sequential, current branch

---

## Done

**F1 — Scaffolding**
- `src/__init__.py`, `src/data/__init__.py`, `src/data/labels/__init__.py`
- `tests/__init__.py`, `tests/data/__init__.py`, `tests/conftest.py`
- `data/raw/.gitkeep`, `data/processed/.gitkeep`
- `.env.example` (already existed as untracked — confirmed correct content)
- Label registry: `LABELLERS` dict + `@register` decorator

**F2 — Label ABC + FVG**
- `src/data/labels/base.py`: `BaseLabeller` ABC with `label()`, `encode()`
- `src/data/labels/fvg.py`: `FVGLabeller` — fully vectorised NumPy, N+1 label convention
- `tests/data/test_fvg_labeller.py`: 14 tests — 9-candle falsification fixture, off-by-one check, encoding, edge cases
- Fixture design required 3 iterations to engineer data with exactly 2 FVGs and no accidental gap conditions

**F3 — Alpaca download + H1 resample**
- `src/data/download.py`: minute pull, RTH filter, 09:30-anchored 1h resample, OHLC validation, session tagging, cache
- `tests/data/test_download.py`: 8 tests — all mocked, no real API calls
- Fixes: `early_closes` is DatetimeIndex not DataFrame (exchange_calendars API difference); `bar_set.df` is attribute not callable; `load_dotenv()` moved to call time to avoid module-level side effect

**F4 — Process (label + encode + persist)**
- `src/data/process.py`: orchestrates download → label → encode → parquet
- `tests/data/test_process.py`: 4 tests — mocked via dict patch pattern

**F5 — Causal normalisation**
- `src/data/normalize.py`: per-window z-score (OHLC) + log1p z-score (volume); pure function; zero-std returns zeros
- `tests/data/test_normalize.py`: 10 tests — anti-lookahead (2 approaches), log1p verification, edge cases

**F6 — Sliding window dataset**
- `src/data/window.py`: `build_windows()` generator + `SMCWindowDataset(torch.Dataset)` with `label_counts`
- Cross-session gap detection: threshold 90 min between consecutive bars
- `tests/data/test_window.py`: 9 tests

**F7 — Temporal split**
- `src/data/split.py`: strict temporal boundaries 2018–2021/2022/2023–2024
- `tests/data/test_split.py`: 5 tests — no overlap, no row loss, empty split error

**F8 — Gold-set annotation tool**
- `src/data/annotate.py`: `sample_gold_set()` (3-strata), `compute_kappa()`, `run_annotation_ui()` (interactive, not unit-tested per plan)
- `scripts/annotate_gold_set.py`: CLI runner
- `data/gold_labels.example.csv`: schema example (committed)
- `tests/data/test_annotate.py`: 6 tests — kappa matches sklearn, 75 rows, no dupes, strata present

**F9 — Integration pipeline**
- `src/data/pipeline.py`: `build_pipeline()` end-to-end; class weights normalised to sum=num_classes
- `tests/data/test_pipeline.py`: 5 tests — fixture uses continuous 1h bars to avoid cross-session window drops

---

## Final test result

```
61 passed, 1 warning in 10.25s
```

All 61 tests pass. No failures.

---

## Good

- Vectorised FVG labeller: zero Python loops over rows
- Anti-lookahead: 2 independent test approaches (mutation + independent windows)
- Mock discipline: no real Alpaca API calls in any test
- `load_dotenv()` moved to call time (not import time) — no module-level side effect
- Class weights normalised explicitly (inverse-frequency alone does not guarantee sum = num_classes)

---

## Bad / Fixed

1. **Fixture FVG accidental triggers**: first 9-candle fixture design caused 4 unintended FVGs. Required 3 redesigns with deliberate price engineering.
2. **exchange_calendars API**: `nyse.early_closes` is a `DatetimeIndex`, not a DataFrame with `.index` — fixed by using `.date` directly.
3. **Mock `bar_set.df`**: tests set `bar_set.df.return_value = df` (callable mock) but code uses `bar_set.df` (attribute). Fixed to `bar_set.df = df`.
4. **Cross-session gap drops all windows**: `SMCWindowDataset` with 7h-spaced fixture data dropped every window (420 min gap > 90 min threshold). Fixed by using continuous 1h fixture for integration tests.
5. **`load_dotenv` in cred test**: `patch.dict(os.environ, clear=True)` doesn't prevent `load_dotenv()` from reloading `.env`. Fixed by patching `src.data.download.load_dotenv` directly.
6. **Class weights sum**: inverse-frequency formula without explicit normalisation produces sum != num_classes on imbalanced data. Added explicit `weights *= num_classes / weights.sum()`.

---

## Observations

- FVG positive rate on synthetic data: ~10% (expected range 5–15% per R3 — consistent)
- Cross-session gap filter (90 min threshold) is strict — any non-RTH-structured fixture will produce 0 windows. Real data will work correctly; test fixtures must use continuous or RTH-structured timestamps.
- `run_annotation_ui()` intentionally not unit-tested (plan spec: interactive). Documented here as verification method: manual run against `data/processed/spy_h1.parquet`.

---

## Open flags

- F10 (CRISP-DM notebook) deferred to nb-notebook agent
- Gold-set annotation requires real data on disk (`data/processed/spy_h1.parquet`) — cannot run until Alpaca credentials are live and `build_labelled_dataset()` is called
- `scripts/annotate_gold_set.py` had a syntax error (`import pandas as pd as _pd`) — fixed
- Python version in venv: 3.12.7 (plan requires 3.12+) — confirmed compatible

---

## Next steps

1. Provide Alpaca credentials in `.env` and run `build_labelled_dataset()` to generate `data/processed/spy_h1.parquet`
2. Run `python scripts/annotate_gold_set.py` for gold-set annotation (75 candles)
3. Launch nb-notebook for Foundation 10 (CRISP-DM notebook, Status Update 1 deliverable)
4. Phase 4 entry point: `from src.data.pipeline import build_pipeline`
