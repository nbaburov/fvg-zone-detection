# Test Log: Phase 3 Data Prep — Independent Verification — 2026-05-08

## Result
PASS — 61 passed / 0 failed / 0 skipped (12.02s)
1 deprecation warning (websockets.legacy — irrelevant, from alpaca-trade-api transitive dep)

---

## Test distribution per module

| Module | Tests | Scope |
|--------|-------|-------|
| test_annotate.py | 6 | F8 — gold set sampling + kappa |
| test_download.py | 8 | F3 — Alpaca download + resample |
| test_fvg_labeller.py | 14 | F2 — FVG labeller |
| test_normalize.py | 10 | F5 — causal normalisation |
| test_pipeline.py | 5 | F9 — end-to-end integration |
| test_process.py | 4 | F4 — build_labelled_dataset |
| test_split.py | 5 | F7 — temporal split |
| test_window.py | 9 | F6 — sliding window dataset |
| **Total** | **61** | |

---

## Quality audit

### test_fvg_labeller.py — STRONG
- Falsification fixture: 9-candle synthetic with exactly 2 engineered FVGs (bull at index 3→label 4, bear at index 6→label 7). Design rationale documented in fixture comments.
- Off-by-one check: `test_label_not_at_fvg_candle_itself` explicitly asserts labels[3]==0 and labels[6]==0 — N+1 convention verified, not just asserted at the label position.
- `test_no_other_nonzero_labels` checks the full label series for unexpected non-zero values — catches false positives.
- Boundary cases: first/last candle always zero tested separately.
- Parametrized edge case: 1-, 2-, 3-row DataFrames all return all-zeros.
- Encode mapping tested against specific values (not just in-range check).
- No mocking of internal logic — real FVGLabeller called throughout.
- No `assert True` patterns. No truthy-only assertions.

### test_normalize.py — STRONG
- Anti-lookahead tested via two independent approaches:
  1. Mutation test: verifies function is pure (identical input → identical output).
  2. Independent windows: row 200 mutation cannot affect window at rows 100-160. This is the meaningful lookahead check — correctly simulates the build_windows pipeline.
- Volume log1p test verifies the output differs from raw z-score (not just "runs without error").
- Zero-std and zero-volume edge cases handled — no NaN guarantee tested.
- Realistic inputs: prices ~400-800, volumes 5k-20k (matches actual SPY scale).

### test_split.py — STRONG
- No-overlap test uses set intersection — strict, catches any shared index.
- No-row-loss test: len(train)+len(val)+len(test)==len(df) exactly.
- Boundary date ranges verified year by year.
- Empty split raises ValueError tested with deliberate bad boundaries.
- Temporal ordering tested with .max() < .min() (not just year comparison).

### test_pipeline.py — GOOD
- End-to-end integration: mocks only `build_labelled_dataset` (Alpaca boundary) and `PROCESSED_DIR` — internal logic (split, window, normalise, weights) all runs real.
- Parquet files read back and overlap-checked — tests actual disk output, not just return values.
- Class weights: shape (3,) and sum≈3.0 both checked (sum check validates the explicit normalisation fix).
- Stride assertion: `len(train_ds) > len(val_ds)` is weak — does not verify stride=1 vs stride=60 exactly, only relative ordering. Sufficient for integration but won't catch if both use wrong stride.

### test_download.py — GOOD
- All 8 tests mock Alpaca at SDK boundary (StockHistoricalDataClient) — correct level.
- `test_no_after_hours_bars` injects an explicit 16:30 bar and checks it's removed — behavioural test.
- `test_ohlc_violation_bar_dropped` injects a violation and asserts OHLC integrity on output.
- `test_cache_hit_skips_api` calls `assert_not_called()` — correct assertion for cache logic.
- `test_half_day_tagged_as_half` has a conditional: `if len(half_day_bars) > 0`. If the filtering logic drops all half-day bars, the assert is skipped silently. Minor: should assert `len(half_day_bars) > 0` first.
- `mock_bar_set.df = MagicMock(return_value=minute_df)` in `_mock_alpaca_bars()` helper sets df as a callable mock, but the fixture in individual tests uses `bar_set.df = two_days_minute_bars` (attribute directly). The helper `_mock_alpaca_bars` is defined but NEVER CALLED in any test — dead code. Tests work because individual tests set the mock correctly, but the helper would be broken if used.

### test_process.py — ADEQUATE
- Labeller mocked via dict patch — correct approach (tests orchestration logic, not labeller).
- `test_raises_value_error_when_too_few_positives` tests 0 positives (extreme case). Does not test the 1% boundary (e.g., 1 positive in 200 rows = 0.5% < 1% should raise; 3 in 200 = 1.5% should not). Boundary behaviour untested.
- 4 tests is thin for an orchestration function, but coverage of core contracts is present.

### test_annotate.py — GOOD
- `test_compute_kappa_matches_sklearn` uses known exact labels and compares against sklearn — strong ground truth test.
- Gold set: 75 rows, no duplicates, all strata present, required columns — all four contracts tested.
- `run_annotation_ui` intentionally untested (interactive — documented in build log).

### test_window.py — ADEQUATE
- `test_build_windows_label_alignment` checks first 10 windows — verifies label is at position k+59, not k. Explicit alignment check, not just "returns something".
- `test_cross_session_window_dropped`: asserts `len(drop) <= len(keep)`. Weak — does not assert that any windows were actually dropped (could pass if the gap filter drops nothing). Gap is 7 days (idx1 ends Mon, idx2 starts following Mon), which at 1h freq is ~168h between consecutive bars — well above 90min threshold. Test likely works but assertion direction only, not count.
- `test_build_windows_stride_60_non_overlapping`: does not verify non-overlap directly — uses count comparison (stride=60 < stride=1). Non-overlap is an implicit consequence of the logic, but not explicitly proven.

---

## Coverage gaps

| Gap | Severity | Detail |
|-----|----------|--------|
| `test_process.py`: 1% boundary not tested | Minor | Only 0% (raises) tested. 0.5% (should raise) and 1.5% (should not) unverified. Boundary condition. |
| `test_download.py`: half-day silent pass | Minor | `if len(half_day_bars) > 0` can silently skip the assert. Should assert length > 0 first. |
| `test_window.py`: cross-session drop count | Minor | Only asserts `<=`, not that any windows were actually dropped. A no-op filter would pass. |
| `test_window.py`: stride=60 non-overlap | Nit | Non-overlap for stride=60 not explicitly proven — only window count compared. |
| `test_pipeline.py`: stride exact value | Nit | Only relative count check (train > val), not that stride values are exactly 1 and 60. |
| `_mock_alpaca_bars()` helper dead code | Nit | Defined but never used. If called, `bar_set.df` would be callable mock, not attribute — would break. |
| `run_annotation_ui()` untested | Known/accepted | Interactive function. Documented as manual verification only. Acceptable. |

---

## Bad test patterns found

| Pattern | Location | Detail |
|---------|----------|--------|
| Dead helper function | test_download.py L69-74 `_mock_alpaca_bars` | Defined, never called. Contains a bug (`bar_set.df = MagicMock(return_value=...)` but real code uses attribute, not callable). |
| Silent conditional assert | test_download.py L183-187 `test_half_day_tagged_as_half` | `if len(half_day_bars) > 0: assert ...` — empty result silently passes. |
| Weak inequality assertion | test_window.py L124-126 `test_cross_session_window_dropped` | `assert len(result_drop) <= len(result_keep)` — equality would pass (no windows dropped). |

No `assert True` patterns. No pure truthy assertions. No hardcoded absolute paths. All paths use `tempfile.TemporaryDirectory()`. Tests are independent (no shared mutable state between tests).

---

## Bugs found (real code bugs discovered via tests)

None — all tests pass and audit found no evidence of implementation bugs not already caught and fixed during build (build log documents 6 bugs fixed during TDD cycle).

---

## Flagged (tests not written / untestable)

- `run_annotation_ui()`: interactive CLI — requires terminal input. Not unit-testable. Accepted.
- Real Alpaca API integration: no live credentials in CI. Accepted (mocked at correct boundary).

---

## Next steps

1. [Minor] Fix silent conditional in `test_half_day_tagged_as_half`: add `assert len(half_day_bars) > 0` before the session_type assert.
2. [Minor] Strengthen `test_cross_session_window_dropped`: assert `len(result_drop) < len(result_keep)` (strict less-than) and optionally assert a specific drop count.
3. [Minor] Add `test_process.py` boundary test: 1 positive in 200 rows should raise; 3 positives in 200 should not.
4. [Nit] Remove or fix dead `_mock_alpaca_bars()` helper in test_download.py.
