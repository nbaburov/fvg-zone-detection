# Review: Phase 3 Code Quality — Pass 2

Date: 8-May-26
Scope: src/ + tests/ — Foundations 1–9
Pass: quality (STANDARDS.md code quality rules)
Result: FAIL

---

## Summary

Issues: 17 total — 0 blocker / 5 major / 8 minor / 4 nit

---

## MAJOR

**[MAJOR] src/data/annotate.py:162–171 — `print()` in `compute_kappa()` for structured logging**
compute_kappa() uses print() to emit the kappa result and gate failure, bypassing the logging module entirely. The file has `logger = logging.getLogger(__name__)` already set up.
Fix: replace both print() calls in compute_kappa() with `logger.warning(...)` (gate failure) and `logger.info(...)` (gate pass). run_annotation_ui() print() calls are acceptable (interactive UI).

**[MAJOR] src/data/pipeline.py:57 — `seed` parameter accepted but never used**
`build_pipeline(seed: int = 42)` — seed appears nowhere in the function body. Dead parameter, YAGNI violation; callers passing `seed=42` get silent no-op.
Fix: remove the parameter. If seeding is needed in future (e.g. for shuffling), add it then.

**[MAJOR] src/data/annotate.py:55–57 — Python for-loop over `pos_indices` in `sample_gold_set()`**
`for pi in pos_indices: for offset in range(-5, 6): ...` builds fvg_rich_candidates with a double Python loop. For large datasets (25k+ candles) this is O(n_positives × 11). Should be vectorised.
Fix:
```python
# vectorised neighbourhood expansion
offsets = np.arange(-5, 6)
neighbour_idx = (pos_indices[:, None] + offsets).ravel()
fvg_rich_candidates = np.unique(np.clip(neighbour_idx, 0, n - 1)).tolist()
```

**[MAJOR] src/data/annotate.py:285 — `sample_gold_set()` exceeds 50-line function limit**
Function is 125 lines (line 15–139). Three distinct responsibilities: stratum 1 sampling, stratum 2 sampling, stratum 3 sampling + merge. SOLID single-responsibility violated.
Fix: extract `_sample_fvg_rich()`, `_sample_low_vol()`, `_sample_random_stratified()` as private helpers; `sample_gold_set` calls them and merges results. Each helper < 30 lines.

**[MAJOR] src/data/labels/base.py:11–14 — class attributes lack `ClassVar` annotation**
`label_index_offset`, `num_classes`, `class_names`, `encoded_map` declared as plain annotations on BaseLabeller. Without `ClassVar`, static type checkers (mypy, pyright) treat them as abstract instance attributes that must be set in `__init__`, triggering false positives on every concrete subclass.
Fix:
```python
from typing import ClassVar
class BaseLabeller(ABC):
    label_index_offset: ClassVar[int]
    num_classes: ClassVar[int]
    class_names: ClassVar[list[str]]
    encoded_map: ClassVar[dict[int, int]]
```

---

## MINOR

**[MINOR] src/data/normalize.py:34,47 — magic epsilon `1e-8` not extracted as constant**
Used twice inline with no named constant. If threshold is ever tuned, both sites must be updated.
Fix: add `_DEGENERATE_STD_THRESHOLD: float = 1e-8` at module top; reference it both times.

**[MINOR] src/data/process.py:57,63 — magic thresholds `0.01`, `0.03` not extracted**
Positive label rate gates (1% error, 3% warning) are inline literals. Not self-documenting.
Fix: add `_MIN_POSITIVE_RATE = 0.01` and `_WARN_POSITIVE_RATE = 0.03` as module-level constants.

**[MINOR] src/data/annotate.py:73,162 — magic numbers `0.25`, `0.6` not extracted**
Low-vol ATR quantile (0.25) and kappa gate threshold (0.6) are inline. Especially kappa threshold is a domain decision that belongs in a named constant.
Fix: `_LOW_VOL_QUANTILE = 0.25`, `_KAPPA_GATE_THRESHOLD = 0.6` at module top.

**[MINOR] src/data/download.py:205,213 — magic number `5000` not extracted**
Minimum bar count sanity check (5000) appears twice inline.
Fix: `_MIN_MULTIYEAR_BAR_COUNT = 5000` at module top.

**[MINOR] src/data/download.py:18–25 — `_NYSE` global mutable state**
`_NYSE = None` is a module-level mutable sentinel, mutated by `_get_nyse()` via `global _NYSE`. This is implicit shared state; breaks thread safety and makes testing/isolation harder.
Fix: use `functools.lru_cache(maxsize=1)` on `_get_nyse()` and remove the global:
```python
import functools
@functools.lru_cache(maxsize=1)
def _get_nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")
```

**[MINOR] src/data/labels/__init__.py:13–19 — `register()` decorator missing type hints**
`def register(name: str)` has no return type. Inner `def decorator(cls)` has no param or return type.
Fix:
```python
from typing import Callable, TypeVar
T = TypeVar("T", bound="BaseLabeller")
def register(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
```

**[MINOR] tests/data/test_fvg_labeller.py:24 — `spy_9candle_fvg` fixture duplicated from conftest.py**
Fixture defined identically in both `tests/conftest.py:17` and `tests/data/test_fvg_labeller.py:24`. DRY violation. The local definition shadows the conftest version, hiding the duplication but not eliminating it.
Fix: remove the local `@pytest.fixture def spy_9candle_fvg` in test_fvg_labeller.py; rely on conftest.

**[MINOR] tests/data/test_window.py:125 — `test_cross_session_window_dropped` assertion is weak**
`assert len(result_drop) <= len(result_keep)` passes even if `result_drop == result_keep` (i.e., if no cross-session windows were actually generated). The gap fixture does produce a 3-day gap, so windows WILL straddle it, but the assertion doesn't verify that any were dropped.
Fix: `assert len(result_drop) < len(result_keep)` — strict less-than proves something was actually filtered. Add a comment: "3-day gap at row 100 guarantees windows 41–100 are dropped".

---

## NIT

**[NIT] src/data/annotate.py:176 — `run_annotation_ui()` function is 109 lines**
Exceeds 50-line limit. UI loop, resume logic, and CSV writing are mixed. Not a blocker because it's an interactive/scripting tool, but warrants splitting.
Fix: extract `_resume_annotated_set()` (lines 208–212), `_display_candle_chart()` (lines 232–252), and `_prompt_label()` (lines 254–271) as private helpers.

**[NIT] src/data/annotate.py:199–202 — late `import csv` and `import plotly` inside function body**
`import csv` and the plotly import occur inside `run_annotation_ui()`. `csv` is stdlib and should be at module top. Plotly guard-import is acceptable (optional dep).
Fix: move `import csv` to module imports; keep the plotly try/except inside the function.

**[NIT] src/data/pipeline.py:39–44 — `_compute_class_weights()` Python for-loop, not vectorised**
`for c in range(num_classes): count = (labels == c).sum()` — not hot path (runs once per pipeline call), but easily vectorised.
Fix:
```python
counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
counts = np.where(counts > 0, counts, 1.0)  # unseen class default
weights = total / (counts * num_classes)
```

**[NIT] src/data/split.py:39 — `+ pd.Timedelta(days=1)` boundary logic is fragile**
Adding 1 day to include the boundary date is implicit. If H1 data ever has bars at exact midnight on the boundary (e.g., from a different source), those bars would appear in both train and val simultaneously, silently breaking the no-overlap guarantee. The test_no_rows_lost_or_duplicated test passes only because bdate_range skips Jan 1.
Fix: use exclusive upper bound with `< (train_end + pd.Timedelta(days=1))` → same effect, or replace the +1day pattern with:
```python
train_df = df.loc[df.index.normalize() <= pd.Timestamp(boundaries["train_end"], tz=df.index.tz)].copy()
```
This keeps only bars whose date (ignoring time) is on or before train_end.

---

## What I couldn't verify

- Circular import check: could not run `python -m src` due to torch not installed in system env. Import graph inspection (manual) shows no cycles — pipeline → process → download (terminal); pipeline → split, window; window → labels.base, normalize. No cycles detected.
- test_half_day_tagged_as_half: depends on NYSE exchange_calendars correctly identifying 2020-11-27 as early-close. Could not execute test.
- annotate.py loop at line 225 (`for _, gold_row in gold_set_df.iterrows()`) — flagged as iterrows() but this is in an interactive annotation UI, not a data-processing hot path. Tolerable here; still a minor code-quality concern (iterrows is always slow, even for 75 rows if called in a tight loop).

---

## APPROVE: no
