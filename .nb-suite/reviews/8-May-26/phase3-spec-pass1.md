# Review — Phase 3 Data Preparation — Spec Compliance Pass 1

Date: 8-May-26
Reviewer: nb-review (sonnet)
Scope: Pass 1 — spec compliance only
Build log read: yes
Plan read: yes

---

REVIEW DOMAIN: code
REVIEW PASS: spec
RESULT: FAIL
ISSUES: 7 total — 0 blocker / 3 major / 3 minor / 1 nit

---

## Findings

### [MAJOR] src/data/download.py:204–214 — sanity check < 5000 bars logic is wrong

Plan contract: "Raises: ValueError if returned bar count < 5000 (sanity check)."
Implementation raises only when `use_cache is False AND end != start AND (end - start) > 365 days`.
Cache-hit path never raises even if the cached data has 3 bars. Also, when `use_cache=True` and cache is corrupt/tiny, no sanity check runs at all. The plan's contract says unconditionally raise, not conditionally based on date range.

Fix: move the `< 5000` check to after `h1` is produced, regardless of cache/live path:
```python
if len(h1) < 5000:
    raise ValueError(f"Sanity check failed: only {len(h1)} H1 bars ...")
```

---

### [MAJOR] src/data/download.py:74 — mock in test sets `bar_set.df` as callable, but code accesses it as attribute — test coverage gap persists

`test_download.py:74`: `mock_bar_set.df = MagicMock(return_value=minute_df)` — this sets `df` to a callable mock, NOT a DataFrame. But `download.py:179` does `minute_df = bar_set.df` (attribute access, not a call). The build log documents this was "Fixed" but the fix was applied only to tests that use `bar_set.df = two_days_minute_bars` directly (e.g. `test_output_columns` line 104). The helper `_mock_alpaca_bars()` at line 70–76 still sets `mock_bar_set.df = MagicMock(return_value=minute_df)` (callable), yet this helper is never used by any test (no test calls `_mock_alpaca_bars`). The helper is dead code — not a functional bug in tests that run, but the fix is incomplete: the helper was the intended abstraction but was abandoned and the actual tests bypass it, creating inconsistency.

Fix: delete `_mock_alpaca_bars()` dead helper (lines 70–76 in `test_download.py`) or fix it to `mock_bar_set.df = minute_df` and wire it into the tests uniformly.

---

### [MAJOR] src/data/split.py:39 — temporal_split does not guarantee `len(train)+len(val)+len(test)==len(df)` when data exists outside 2024-12-31

Plan: "No bar appears in more than one subset" and test `test_no_rows_lost_or_duplicated` asserts total row count. But `split.py` silently drops any bar with index > `2024-12-31 + 1 day` (i.e., 2025+). If `full_df` contains 2025 data (possible if `download_spy_h1` end defaults to today = 2026), those bars are excluded from all three splits and the total check fails.

The test fixture only covers 2018–2024 so the test passes. But the plan's contract "No bar appears in more than one subset" (by extension, no bars lost) is violated for real production data.

Fix: add an explicit assertion or trim `df` to `test_end` at the start of `temporal_split`:
```python
df = df.loc[df.index <= (test_end + pd.Timedelta(days=1))]
```
Or document the contract: only bars within defined boundary range are guaranteed to appear.

---

### [MINOR] src/data/window.py — label at position 59 (plan spec) vs label at `i + window_size - 1` (implementation)

Plan spec says "label is the encoded class of the bar at i+59" (Foundation 6). Implementation uses `label_pos = i + window_size - 1` which equals `i + 59` when `window_size=60`. This is correct when `window_size=60` but the class accepts arbitrary `window_size`. The plan's contract hardcodes window_size=60. No bug for standard use, but the spec says "(60,5)" is fixed — the implementation adds flexibility not in the plan. Minor scope creep.

Fix: either document the window_size parameter as extension, or enforce `window_size=60` as default-only with an assertion.

---

### [MINOR] tests/data/test_split.py — `test_no_rows_lost_or_duplicated` not valid for production data range

The test fixture uses `pd.bdate_range("2018-01-02", "2024-12-31")` which stays within split boundaries. The test passes trivially because no out-of-range data exists in the fixture. The plan requires this invariant to hold for real data. Test does not falsify the actual production failure mode (data after 2024-12-31).

Fix: add a fixture row at 2025-01-02 and assert total row count includes it or assert it is explicitly excluded.

---

### [MINOR] src/data/annotate.py:sample_gold_set — stratum `fvg_rich` uses a Python for-loop over `pos_indices` (lines 55–59)

Plan (Foundation 2) explicitly forbids Python row-loops for labellers. The annotation sampling isn't a labeller but the plan's anti-loop stance applies to vectorised operations generally. The ±5 window expansion loop over `pos_indices` is O(n_positive * 11) and will be slow on 9000+ positive candles in real data. Not in plan scope to fix, but technically a deviation from the vectorised mandate.

Fix: use `np.unique(np.concatenate([pos_indices + offset for offset in range(-5, 6)]).clip(0, n-1))` — single vectorised operation.

---

### [NIT] src/data/labels/__init__.py:7–9 — TYPE_CHECKING guard is unnecessary complexity

Plan contract shows `LABELLERS: dict[str, type["BaseLabeller"]] = {}`. Implementation wraps the import in `TYPE_CHECKING` to avoid circular import. The string annotation `"BaseLabeller"` already avoids the circular import at runtime — the `TYPE_CHECKING` block adds nothing functional and deviates from the plan's clean contract.

Fix: remove `TYPE_CHECKING` block; the string annotation `dict[str, type["BaseLabeller"]]` works without it.

---

## Contract verification results

| Requirement | Status | Notes |
|---|---|---|
| F1: `LABELLERS` dict exported | PASS | |
| F1: `@register` decorator | PASS | |
| F2: `BaseLabeller` ABC with all 4 class attrs | PASS | |
| F2: `FVGLabeller` registered as `"fvg"` | PASS | |
| F2: label_index_offset=1, num_classes=3, class_names correct, encoded_map correct | PASS | |
| F2: label at N+1 not N | PASS | falsification fixture verifies |
| F2: vectorised (no Python row-loop) | PASS | |
| F3: TimeFrame.Minute (not Hour) | PASS | line 176 |
| F3: RTH filter 09:30–15:59 | PASS | `between_time("09:30", "15:59")` line 53 |
| F3: resample `closed='left', label='left'` | PASS | line 56 |
| F3: OHLC integrity 5 rules | PASS | |
| F3: session_type column Categorical | PASS | |
| F3: cache + EnvironmentError on missing creds | PASS (partial) | EnvironmentError raised; < 5000 check wrong — MAJOR above |
| F4: ValueError if positive rate < 1% | PASS | |
| F4: warns if < 3% | PASS | |
| F4: writes parquet | PASS | |
| F5: causal — anti-lookahead test exists and passes | PASS | 2 approaches: mutation + independent windows |
| F6: SMCWindowDataset with stride config | PASS | |
| F6: cross-session windows dropped | PASS | 90 min threshold |
| F6: label at position 59 (window_size=60) | PASS | |
| F7: strict temporal ordering | PASS | |
| F7: ValueError on empty split | PASS | |
| F7: no rows lost/duplicated | PARTIAL — MAJOR above | |
| F8: sample_gold_set returns 75 stratified rows | PASS | |
| F8: compute_kappa matches sklearn | PASS | |
| F8: UI not unit-tested (intentional) | PASS | documented in build log |
| F9: build_pipeline returns 4-tuple | PASS | |
| F9: class weights on TRAIN only | PASS | |
| F9: all parquets + class_weights.json persisted | PASS | |
| F9: class_weights sum ≈ num_classes | PASS | |

---

## UNVERIFIED

- Whether `exchange_calendars` correctly identifies 2020-11-27 (Black Friday) as early close in the test environment — test for `test_half_day_tagged_as_half` has a guard `if len(half_day_bars) > 0` that silently passes if the session tag doesn't work. Cannot verify without running tests in this review.
- Actual Alpaca API behavior with `TimeFrame.Minute` on the full 8-year pull — only verifiable when credentials are live.
- Whether the `test_no_rows_lost_or_duplicated` test actually passes for the default plan-specified production date range (2018–today) given the split bug above.

---

APPROVE: no
