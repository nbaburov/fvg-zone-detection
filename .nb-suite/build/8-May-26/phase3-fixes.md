# Build Log: Phase 3 Fix-up Cycle — 2026-05-08

Agent: nb-build (sonnet)
Prior: phase3-data-prep.md + 3 review artifacts (spec-pass1, quality-pass2, phase3-verification)
Result: PASS — 64 tests, 0 failed (was 61 / +3 new tests)

---

## Findings addressed

### MAJOR (8 fixed)

| # | Location | Fix |
|---|----------|-----|
| 1 | `download.py:204` | `< 5000` check moved unconditional via `_sanity_check_bar_count()` helper; runs on both cache + live paths; guards against date range heuristic using `start`/`end`. Extracted `_MIN_MULTIYEAR_BAR_COUNT = 5000`. |
| 2 | `download.py:18` | Removed `_NYSE` global mutable sentinel. Replaced with `@functools.lru_cache(maxsize=1)` on `_get_nyse()`. |
| 3 | `split.py:39` | Explicit clip to `test_end` via `filtered_df`. `len(train)+len(val)+len(test) == len(filtered_df)` now holds for production data with 2025+ bars. |
| 4 | `labels/base.py` | Added `ClassVar` annotations to all 4 class attrs (`label_index_offset`, `num_classes`, `class_names`, `encoded_map`). |
| 5 | `pipeline.py:57` | Removed dead `seed: int = 42` parameter. |
| 6 | `annotate.py:55` | Vectorised stratum 1 neighbourhood expansion: `(pos_indices[:,None] + np.arange(-5,6)).ravel()` + `np.unique`. |
| 7 | `annotate.py:162` | Replaced `print()` in `compute_kappa()` with `logger.warning()` / `logger.info()`. |
| 8 | `annotate.py:285` | Split `sample_gold_set` into 3 private helpers: `_sample_fvg_rich`, `_sample_low_vol`, `_sample_random_stratified`. Each < 30 lines. |

### MINOR (13 fixed)

| # | Location | Fix |
|---|----------|-----|
| 1 | `normalize.py` | Extracted `_DEGENERATE_STD_THRESHOLD = 1e-8`; used both occurrences. |
| 2 | `process.py` | Extracted `_MIN_POSITIVE_RATE = 0.01`, `_WARN_POSITIVE_RATE = 0.03`. |
| 3 | `annotate.py` | Extracted `_LOW_VOL_QUANTILE = 0.25`, `_KAPPA_GATE_THRESHOLD = 0.6`. |
| 4 | `download.py` | Extracted `_MIN_MULTIYEAR_BAR_COUNT = 5000` (deduped with MAJOR #1). |
| 5 | `labels/__init__.py` | Removed `TYPE_CHECKING` guard; added `Callable`, `TypeVar` type hints on `register()` and `decorator()`. |
| 6 | `test_fvg_labeller.py` | Removed duplicate `spy_9candle_fvg` fixture (was 109 lines); relies on conftest. |
| 7 | `test_window.py:125` | Changed `<=` to strict `<`; added comment explaining 3-day gap guarantees at least one drop. |
| 8 | `test_split.py` | Added `test_post_test_end_rows_are_clipped`: injects 2025-01-02 bar, asserts it is excluded, asserts row-count invariant on filtered_df. |
| 9 | `test_process.py` | Added `test_raises_when_below_1pct_boundary` (1/200 = 0.5%) and `test_does_not_raise_above_1pct_boundary` (3/200 = 1.5%). |
| 10 | `test_download.py:184` | Removed silent `if len(half_day_bars) > 0` guard; added assertive `assert len(half_day_bars) > 0` first. |
| 11 | `test_download.py:69` | Deleted dead `_mock_alpaca_bars()` helper. |
| 12 | `split.py` | Date-normalised boundary comparison via `.normalize()` throughout; removes fragile `+1 Timedelta` pattern. |
| 13 | `test_pipeline.py` | Removed `seed=42` kwarg from all `build_pipeline()` calls (parameter no longer exists). |

### NIT (4 fixed)

| # | Location | Fix |
|---|----------|-----|
| 1 | `annotate.py:199` | Moved `import csv` to module top. Plotly guard-import kept inside function (optional dep). |
| 2 | `annotate.py:176` | Extracted `_resume_annotated_set()`, `_display_candle_chart()`, `_prompt_label()` as private helpers. `run_annotation_ui()` now ~35 lines. |
| 3 | `pipeline.py:39` | Replaced for-loop with `np.bincount(labels.astype(int), minlength=num_classes)`. |
| 4 | `window.py` | Documented `window_size` parameter in `build_windows()` docstring as planned-spec extension. |

---

## Test delta

| Metric | Before | After |
|--------|--------|-------|
| Total tests | 61 | 64 |
| Passed | 61 | 64 |
| Failed | 0 | 0 |
| New tests | — | +3 (test_process boundary ×2, test_split clip ×1) |

---

## Deferred items

None. All major, minor, and nit findings from all 3 review artifacts addressed.

---

## Files modified

- `src/data/download.py`
- `src/data/split.py`
- `src/data/labels/base.py`
- `src/data/labels/__init__.py`
- `src/data/annotate.py`
- `src/data/normalize.py`
- `src/data/process.py`
- `src/data/pipeline.py`
- `src/data/window.py`
- `tests/data/test_download.py`
- `tests/data/test_fvg_labeller.py`
- `tests/data/test_split.py`
- `tests/data/test_process.py`
- `tests/data/test_window.py`
- `tests/data/test_pipeline.py`
