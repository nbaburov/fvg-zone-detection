# Valid FVG Labeller — Implementation Plan

> **Ready for /nb:build.**
> Work type: Improvement/refactor — replaces `src/data/labels/fvg.py` with 6-criteria valid FVG implementation. Extends existing labeller architecture.
> Recommended mode: /nb:build standard (single branch, sequential phases). All phases depend on prior.
> Test strategy: TDD (pytest). Each criterion gets an independent synthetic fixture.
> Depth: Deep

**Goal:** Replace the raw geometric FVG labeller (κ = 0.284 vs human gold) with TradingLab's 6-criteria "valid FVG" rule. Change label index from N+1 to N+2. Discard existing 75-candle gold set and re-annotate under the new definition. Gate progress on empirical class-count check before annotation effort.

---

## Locked decisions (do not re-open)

| Decision | Value |
|----------|-------|
| Label index | N+2 (reaction candle must be closed) |
| Output schema | Ternary {0=none, 1=bull-valid, 2=bear-valid} |
| Swing lookback | 50 bars |
| ATR confluence threshold | 0.5 × ATR(14) |
| Mitigation rule | Body close inside gap — inference only, not in training labels |
| Ablation logging | Mandatory — per-criterion boolean, exportable CSV |
| Two-model split | Model A: labels exclude mitigation. Model B / inference: applies mitigation live |

---

## Three open risks — resolved here

### Risk 1: BOS window tautology

**Problem:** If BOS lookback = 50 bars and swing is defined over the same 50-bar window, a BOS is "break of the prior 50-bar swing high." Then checking "did a BOS occur in the past 50 bars?" is tautologically nearly always true, because every meaningful price move breaks at some local extreme within 50 bars.

**Decision: Use 20-bar BOS lookback for the "recent BOS" check (criterion #6), while keeping 50-bar lookback for swing definition (criterion #4, #5).**

Reasoning: The swing window (50 bars) defines the structural range for priority and Gann position. The BOS check (20 bars) enforces that a structure break occurred in the immediate past — not just anywhere in the last 50 bars. 20 bars ≈ 3 trading days on H1. This is "recent" in practitioner SMC usage. The asymmetry is intentional and documented in CLAUDE.md. If ablation shows criterion #6 still barely filters, widen BOS lookback to test sensitivity.

```python
BOS_LOOKBACK: int = 20   # recent structure break window (separate from SWING_LOOKBACK)
SWING_LOOKBACK: int = 50  # swing range for priority + gann
```

### Risk 2: Class sparsity gate

**Decision tree — run BEFORE gold re-annotation:**

1. Run `ValidFVGLabeller.label()` on the full processed H1 dataset (`data/processed/spy_h1.parquet`).
2. Count positives: `n_pos = (labels != 0).sum()`.
3. Apply gate:
   - `n_pos >= 150` → proceed to gold re-annotation.
   - `75 <= n_pos < 150` → relax criterion #3 (S/R confluence) first: increase `confluence_atr_mult` from 0.5 to 1.0. Recount. If still < 150, drop criterion #3 entirely. Log which relaxation was applied.
   - `n_pos < 75` → relax criterion #3 as above AND reduce `bos_lookback` from 20 to 50 (making BOS nearly always pass, effectively dropping criterion #6). Log. Surface to user before proceeding.

**Why criterion #3 first:** It is the most aggressive filter (estimated 60% rejection) and the most parameter-sensitive. ATR multiplier is a soft knob, not a binary gate. Criteria #2 (reaction candle) and #6 (BOS) are structurally important for thesis argumentation and should be preserved as long as possible.

**Script:** Add `scripts/count_valid_fvg.py` — loads parquet, runs labeller, prints count and per-criterion breakdown from ablation log. Run this before starting gold annotation. Output is an ablation CSV at `.nb-suite/analysis/criterion-ablation-YYYY-MM-DD.csv`.

### Risk 3: Reaction candle ambiguity

**Decision: Default = "closes inside gap" only. Log "closes in direction" as ablation variant.**

Criterion #2 has two possible definitions:
- **Strict:** Reaction candle (N+2) `close` falls within `[fvg_bottom, fvg_top]`. Requires price to re-enter the gap zone before closing.
- **Loose:** Reaction candle close is on the correct side of the gap (above `fvg_bottom` for bull, below `fvg_top` for bear). Nearly any candle passes.

The strict version is more meaningful ("price accepted the zone") and consistent with ICT's "price trades into the gap" requirement. The loose version risks labelling any candle following a geometric FVG.

**Default implementation:** strict ("closes inside gap"). The ablation log records `crit2_strict_pass` and `crit2_loose_pass` as separate columns so thesis analysis can compare both.

```python
# Strict (default)
bull_react = (react_close >= fvg_bottom) & (react_close <= fvg_top)
# Loose (ablation column only, not used in label)
bull_react_loose = react_close >= fvg_bottom
```

---

## Architecture overview

The existing labeller architecture (`BaseLabeller` → `FVGLabeller`) is retained unchanged. A new concrete class `ValidFVGLabeller` is added alongside the existing `FVGLabeller`. The existing raw FVG labeller is NOT deleted — it is kept for ablation comparison and registered under `"fvg_raw"`. The new labeller registers under `"fvg_valid"`.

Pipeline flag: `build_pipeline(labeller_name="fvg_valid")` switches to the new labeller. No other changes to `pipeline.py` are needed — the registry handles dispatch.

Helper modules for S/R levels and BOS are extracted into `src/data/labels/sr.py` and `src/data/labels/bos.py` as pure-function modules (not classes). `ValidFVGLabeller` imports from both.

```
src/data/labels/
  base.py          — unchanged
  fvg.py           — renamed internals: FVGLabeller → keep as-is, re-register as "fvg_raw"
  valid_fvg.py     — NEW: ValidFVGLabeller, registers as "fvg_valid"
  sr.py            — NEW: causal pivot high/low S/R computation
  bos.py           — NEW: causal BOS detection
  __init__.py      — unchanged (registry auto-populates on import)
```

Gold annotation pipeline:
```
scripts/count_valid_fvg.py   — run first: empirical sparsity check
scripts/annotate_gold_set.py — run after gate passes: re-annotate with new guide
```

Documentation:
```
docs/fvg-label-guide.md      — full rewrite: 6-criteria decision tree
notebooks/02-gold-annotation.ipynb — new chart overlays (swing markers, S/R lines, Gann box)
```

---

## Data model / types

### ValidFVGLabeller config

```python
@register("fvg_valid")
class ValidFVGLabeller(BaseLabeller):
    label_index_offset: ClassVar[int] = 2         # N+2 (reaction candle)
    num_classes: ClassVar[int] = 3
    class_names: ClassVar[list[str]] = ["none", "bullish", "bearish"]
    encoded_map: ClassVar[dict[int, int]] = {0: 0, 1: 1, -1: 2}

    # Configurable via constructor — not ClassVar
    swing_lookback: int = 50
    bos_lookback: int = 20
    sr_lookback: int = 5
    atr_period: int = 14
    confluence_atr_mult: float = 0.5

    def label(self, df: pd.DataFrame) -> pd.Series: ...
    def label_with_ablation(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]: ...
```

### Ablation DataFrame schema

Returned by `label_with_ablation()` — NOT part of the label Series:

```python
# One row per candle. All columns are bool.
ablation_df.columns = [
    "geometric_bull",      # raw geometric FVG passes, bullish
    "geometric_bear",      # raw geometric FVG passes, bearish
    "crit2_strict_bull",   # reaction candle closes inside gap (bull)
    "crit2_strict_bear",   # reaction candle closes inside gap (bear)
    "crit2_loose_bull",    # reaction candle closes in gap direction (bull)
    "crit2_loose_bear",    # reaction candle closes in gap direction (bear)
    "crit3_sr_bull",       # S/R confluent (bull)
    "crit3_sr_bear",       # S/R confluent (bear)
    "crit5_gann_bull",     # Gann box position passes (bull)
    "crit5_gann_bear",     # Gann box position passes (bear)
    "crit6_bos_bull",      # recent BOS precedes (bull)
    "crit6_bos_bear",      # recent BOS precedes (bear)
    "valid_bull",          # all criteria pass, bullish
    "valid_bear",          # all criteria pass, bearish
]
```

### S/R module types

```python
# src/data/labels/sr.py
def compute_pivot_levels(
    df: pd.DataFrame,
    lookback: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (pivot_high, pivot_low) as causal Series.
    pivot_high[i] = max(high[i-lookback..i-1]) — rolling max, shift(1).
    pivot_low[i]  = min(low[i-lookback..i-1])  — rolling min, shift(1).
    NaN for first `lookback` bars.
    """
```

### BOS module types

```python
# src/data/labels/bos.py
def compute_bos(
    df: pd.DataFrame,
    swing_lookback: int = 50,
    bos_lookback: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (bull_bos_recent, bear_bos_recent) as bool Series.
    bull_bos_recent[i] = True if any close > rolling_50_max occurred in past bos_lookback bars.
    bear_bos_recent[i] = True if any close < rolling_50_min occurred in past bos_lookback bars.
    Both use shift(1) — BOS must precede bar i.
    """
```

---

## Component breakdown

### `src/data/labels/sr.py`

**What it does:** Computes causal pivot high/low S/R levels. No class — pure functions. Uses `df["high"].rolling(lookback).max().shift(1)` and `df["low"].rolling(lookback).min().shift(1)`. Returns two Series with the same index as df. NaN for first `lookback` bars (not enough history). Does not depend on any other src module.

**Owns:** Pivot high/low computation only. Does not compute ATR, does not decide confluence.

**Exposes:** `compute_pivot_levels(df, lookback) -> (pd.Series, pd.Series)`

**Depends on:** pandas only.

### `src/data/labels/bos.py`

**What it does:** Computes causal BOS indicator. A bullish BOS at bar j = `close[j] > max(high[j-swing_lookback..j-1])`. Rolls this into a "any BOS in past bos_lookback bars" boolean. Uses `.rolling().max().shift(1)` twice — once for the swing reference level, once for the recency window. Fully vectorised.

**Owns:** BOS detection. Does not compute swing range for other criteria — that is done inline in ValidFVGLabeller.

**Exposes:** `compute_bos(df, swing_lookback, bos_lookback) -> (pd.Series, pd.Series)`

**Depends on:** pandas only.

### `src/data/labels/valid_fvg.py`

**What it does:** Implements `ValidFVGLabeller`. Imports `compute_pivot_levels` and `compute_bos`. In `label()`, runs the 6-step vectorised pipeline (geometry → criterion 2 → 3 → 5 → 6 → assemble at N+2). Returns ternary Series {0, 1, -1} with label at N+2 for each passing FVG. In `label_with_ablation()`, runs same pipeline but also assembles and returns the ablation DataFrame.

**Owns:** All 6-criteria logic, ATR computation (inline, 3 lines), Gann box check, priority mode (label all passing candidates), label assembly at N+2.

**Exposes:** `label(df) -> pd.Series`, `label_with_ablation(df) -> tuple[pd.Series, pd.DataFrame]`

**Depends on:** `sr.py`, `bos.py`, `base.py`, numpy, pandas.

### `src/data/labels/fvg.py` — re-registration change

**Change:** Add `@register("fvg_raw")` alongside existing `@register("fvg")`. Existing `FVGLabeller` class is unchanged. Pipeline calls with `"fvg_raw"` get the geometric labeller. This preserves ablation comparison capability.

**No other changes to `fvg.py`.**

### `scripts/count_valid_fvg.py`

**What it does:** Loads `data/processed/spy_h1.parquet`, instantiates `ValidFVGLabeller`, calls `label_with_ablation()`, prints: total candles, n_pos (bull+bear), positive rate %, per-criterion pass counts. Saves ablation DataFrame to `.nb-suite/analysis/criterion-ablation-<date>.csv`. Must be run before starting gold re-annotation. Exits with code 1 if n_pos < 75 to signal human intervention needed.

### `docs/fvg-label-guide.md` — full rewrite

Content outline (builder writes the full document):

```
# How to Label Valid FVG — 6-Step Decision Guide

## The pattern in one sentence

## What you see on the chart
  - FVG zone highlighted (box from bottom to top of gap)
  - Swing markers (purple triangles, 50-bar pivot highs/lows)
  - S/R lines (horizontal, 5-bar pivot levels)
  - Gann box overlay (midpoint line of 50-bar swing)

## Step 1: Is there a geometric gap? (criterion #1 baseline)
  - Visual definition
  - "None" cases — when to reject immediately

## Step 2: Does bar N+2 close inside the gap? (criterion #2)
  - N+2 = 2 bars AFTER the red dashed line
  - Visual examples: yes / no
  - ELI5: "Did price come back into the zone before leaving?"

## Step 3: Is there an S/R level nearby? (criterion #3)
  - What the horizontal lines show
  - "Nearby" = line overlaps the highlighted box
  - ELI5: "Is this zone near a recent local high or low?"

## Step 4: Is this the lowest (bull) or highest (bear) gap in the range? (criterion #4 — priority)
  - Mark all valid — annotator does not need to assess priority

## Step 5: Is the gap in the lower half (bull) or upper half (bear) of the range? (criterion #5)
  - What the Gann box midpoint line shows
  - Bull FVG bottom must be below the midpoint
  - Bear FVG top must be above the midpoint

## Step 6: Was there a structure break recently to the left? (criterion #6)
  - Look left 20 bars. Did price close above/below a prior swing extreme?
  - Visual: swing triangles show the prior highs/lows
  - ELI5: "Did the market break structure before creating this gap?"

## Decision table
  all 6 pass → Confirm (bullish or bearish per direction)
  any fails  → None

## What to mark Ambiguous
## Reference definitions
```

### `notebooks/02-gold-annotation.ipynb` — UI changes

Three chart overlays added to the existing Plotly candlestick:
1. **Swing markers:** vertical markers at bars where `pivot_high` or `pivot_low` are locally extreme (within the displayed 30-bar window).
2. **S/R lines:** horizontal lines at `pivot_high` and `pivot_low` levels visible on the chart.
3. **Gann box:** shaded rectangle covering the FVG zone [bottom, top] with a horizontal midpoint line at `swing_mid`.

Chart window widens from 7 bars to 30 bars (enough to see recent swing structure and BOS context).

Display schema remains: NO programmatic label shown. New overlays are computed on-the-fly from the raw OHLCV df, not from the labelled df.

---

## Sequential build order

All phases are sequential. No parallel workstreams — each foundation depends on the prior.

### Phase 1: Helper modules (sr.py, bos.py)

**Model:** haiku
**Files:**
- Create: `src/data/labels/sr.py`
- Create: `src/data/labels/bos.py`
- Create: `tests/data/labels/test_sr.py`
- Create: `tests/data/labels/test_bos.py`

**Acceptance criteria:**
- `compute_pivot_levels` returns two Series with correct NaN prefix (first `lookback` bars are NaN).
- Pivot high at bar i uses only bars i-lookback..i-1 (assert by mutating bar i itself and checking output is unchanged).
- `compute_bos` returns bool Series where bull_bos_recent[i] is True only when a close > prior_swing_high occurred in the past bos_lookback bars.
- BOS result at bar i does NOT depend on bar i's close (shift(1) is correctly applied).
- Both modules produce no NaN in their valid range (post-warmup bars).
- Tests pass: `pytest tests/data/labels/test_sr.py tests/data/labels/test_bos.py`

**Effort:** 1–2 hours

### Phase 2: ValidFVGLabeller core (valid_fvg.py)

**Model:** sonnet
**Depends on:** Phase 1
**Files:**
- Create: `src/data/labels/valid_fvg.py`
- Create: `tests/data/labels/test_valid_fvg.py`

**Acceptance criteria:**
- Label appears at N+2, not N+1 (9-candle falsification fixture — pattern at bars 3–5 → label at bar 7 = N+2).
- Bullish label = 1, bearish label = -1. First 2 and last 2 rows always 0.
- No NaN in output.
- Each criterion independently filters: for each criterion C, a synthetic fixture where ALL criteria pass except C must return label=0.
- `label_with_ablation()` returns ablation DataFrame with all 14 columns listed in data model section.
- `crit2_strict` fails when reaction candle closes outside gap. `crit2_loose` passes in the same case.
- `crit3_sr` fails when no S/R level is within the expanded zone.
- `crit5_gann` fails when bull FVG bottom is above swing midpoint.
- `crit6_bos` fails when no BOS in past 20 bars.
- Labeller registers as `"fvg_valid"` in the LABELLERS registry.
- `build_pipeline(labeller_name="fvg_valid")` completes without error on a 500-row synthetic fixture.
- Tests pass: `pytest tests/data/labels/test_valid_fvg.py`

**Effort:** 3–5 hours

**Critical implementation note on ATR:** Use the simplified ATR proxy from research (`close.diff().abs().rolling(atr_period).mean()`) with `.shift(1)` to prevent lookahead. Do not introduce `pandas_ta` dependency.

### Phase 3: Re-register raw FVG labeller

**Model:** haiku
**Depends on:** Phase 2
**Files:**
- Modify: `src/data/labels/fvg.py` — add `@register("fvg_raw")` decorator alongside or replace `@register("fvg")` with both.
- Modify: `tests/data/labels/test_fvg_labeller.py` — update fixture to use `"fvg_raw"` key where applicable. Add assertion: `"fvg_raw"` and `"fvg"` both resolve to `FVGLabeller` from registry.

**Acceptance criteria:**
- `LABELLERS["fvg_raw"]` resolves to `FVGLabeller`.
- `LABELLERS["fvg"]` still resolves (kept for backward compat with existing pipeline calls during transition).
- All existing `test_fvg_labeller.py` tests pass unchanged.
- **Effort:** 30 minutes

### Phase 4: Sparsity gate script

**Model:** haiku
**Depends on:** Phase 2
**Files:**
- Create: `scripts/count_valid_fvg.py`
- Create: `.nb-suite/analysis/.gitkeep` (directory marker)

**Acceptance criteria:**
- Script loads `data/processed/spy_h1.parquet` (must pre-exist from prior pipeline run).
- Runs `ValidFVGLabeller().label_with_ablation()` on full dataset.
- Prints: total candles, n_bull, n_bear, n_pos, positive_rate (%).
- Prints per-criterion pass counts from ablation df.
- Saves ablation CSV to `.nb-suite/analysis/criterion-ablation-<YYYY-MM-DD>.csv`.
- Exits with code 0 if n_pos >= 150, code 1 if < 75, code 2 if 75 <= n_pos < 150.
- Human reads the output and applies the decision tree from Risk 2 before proceeding to Phase 5.

**This script produces the go/no-go decision for gold re-annotation. It is run manually, not in CI.**

**Effort:** 1 hour

### Phase 5: Gold annotation overhaul

**Model:** sonnet
**Depends on:** Phase 4 (sparsity gate run and passed), Phase 3
**Files:**
- Modify: `docs/fvg-label-guide.md` — full rewrite per outline in component breakdown section
- Modify: `notebooks/02-gold-annotation.ipynb` — add swing markers, S/R lines, Gann box; widen window from 7 to 30 bars
- Move: `data/gold_labels.csv` → `data/gold_labels_v1_geometric.csv` (backup, do not delete)
- Modify: `src/data/annotate.py` — update `sample_gold_set()` to read `raw_label` from `ValidFVGLabeller` output (the column name is already `raw_label` in the pipeline — no schema change needed). Update `_display_candle_chart()` to render the 3 new overlays and use 30-bar window.

**New gold annotation session:** After this phase is built, annotator runs `notebooks/02-gold-annotation.ipynb` to re-annotate all 75 candles (same candle indices from `data/gold_labels_v1_geometric.csv`, reused for sampling — but annotation decision restarts from scratch). `data/gold_labels.csv` (fresh) is the output.

**Acceptance criteria (builder):**
- `docs/fvg-label-guide.md` covers all 6 criteria with visual examples and decision table.
- `notebooks/02-gold-annotation.ipynb` displays swing markers (purple), S/R horizontal lines (orange), Gann midpoint line (blue dashed) on a 30-bar window.
- Gann box overlay correctly shows the FVG zone [bottom, top] as a shaded rectangle.
- `run_annotation_ui()` in `annotate.py` correctly handles 30-bar window without IndexError at dataset boundaries.
- Old gold labels are backed up at `data/gold_labels_v1_geometric.csv` before any re-annotation begins.
- Manual test: run one annotation step in the notebook; chart renders with all overlays; b/e/n/a keys work; annotation appends to `data/gold_labels.csv`.

**Effort:** 3–4 hours (builder) + 1.5–2.5 hours (annotator running the tool)

### Phase 6: κ re-measurement and gate

**Model:** haiku
**Depends on:** Phase 5 (all 75 candles annotated)
**Files:**
- No new files. Run `scripts/annotate_gold_set.py` (existing) with `compute_kappa()`.

**Acceptance criteria:**
- `compute_kappa("data/gold_labels.csv")` runs without error.
- Output logged to console and to `.nb-suite/analysis/kappa-<YYYY-MM-DD>.txt`.
- If κ ≥ 0.6: proceed to Phase 7.
- If κ < 0.6: do NOT proceed. Investigate 3 worst-disagreement candles. If annotation guide is unclear, update `docs/fvg-label-guide.md` and re-annotate those candles only. Surface to user with specific disagreement examples before any further action.

**Effort:** 30 minutes (tooling); annotation re-run contingency = 1–2 hours

### Phase 7: Pipeline integration and CLAUDE.md update

**Model:** haiku
**Depends on:** Phase 6 (κ gate passed)
**Files:**
- Modify: `src/data/pipeline.py` — confirm `build_pipeline(labeller_name="fvg_valid")` works end-to-end. No code change expected — registry handles dispatch. Add `fvg_valid` to inline docstring as valid option.
- Modify: `CLAUDE.md` — update "Critical Constraints" section: label index N+2 (not N+1), note ValidFVGLabeller is the active labeller, note raw FVG kept as `"fvg_raw"` for ablation.
- Modify: `docs/idea.md` — update labelling section to reference 6-criteria rule if scope description was using raw FVG definition. Do not change scope.
- Run: `pytest` (full suite) — all existing tests must pass.

**Acceptance criteria:**
- `build_pipeline(labeller_name="fvg_valid")` returns `(train_ds, val_ds, test_ds, class_weights)` without error on real data.
- Class weights tensor shape = (3,), sum ≈ 3.0.
- `build_pipeline(labeller_name="fvg_raw")` still works (ablation path).
- Full pytest suite passes.
- CLAUDE.md reflects N+2 convention.

**Effort:** 1 hour

### Phase 8: Ablation analysis notebook section

**Model:** sonnet
**Depends on:** Phase 7
**Files:**
- Modify: `notebooks/02-gold-annotation.ipynb` — add new section: "Criterion Ablation Analysis". Load `.nb-suite/analysis/criterion-ablation-<date>.csv`. Show per-criterion pass counts as a bar chart. Show the rejection funnel (geometric → crit2 → crit3 → crit5 → crit6 → valid). Observation cell: which criterion is the most aggressive filter and why this supports validity of the label design.

**Acceptance criteria:**
- Bar chart renders with correct per-criterion counts.
- Observation cell contains ≥ 3 sentences of substantive analysis.
- The ablation CSV exists on disk (produced by Phase 4 script).

**Effort:** 1–2 hours

---

## Test strategy

### Unit tests — per criterion, synthetic fixtures

All fixtures in `tests/conftest.py` or `tests/data/labels/conftest.py`.

**Mandatory synthetic fixtures (create as pytest fixtures):**

```python
# Canonical 9-candle FVG pattern for N+2 causality test
@pytest.fixture
def fvg_9candle_n2():
    """
    Bars 3-5 form bullish FVG.
    Bar 7 = N+2 (reaction candle).
    ValidFVGLabeller must label bar 7 = 1 (IF all other criteria also pass).
    """
```

For each criterion, a "criterion-isolated" fixture where:
- All other criteria pass (via synthetic construction).
- The target criterion fails.
- Expected label = 0.

| Fixture name | Criterion isolated | Failure condition |
|---|---|---|
| `fvg_crit2_fail` | Criterion #2 (reaction candle) | N+2 close is outside gap (reaction candle misses) |
| `fvg_crit3_fail` | Criterion #3 (S/R confluence) | No S/R level within ATR-expanded zone |
| `fvg_crit5_fail` | Criterion #5 (Gann box) | Bull FVG bottom above swing midpoint |
| `fvg_crit6_fail` | Criterion #6 (BOS) | No close > prior 50-bar high in past 20 bars |

For each fixture, a parallel "criterion-pass" variant to confirm the logic inverts correctly.

### Causality test (mandatory)

In `test_valid_fvg.py`:

```python
def test_no_future_bar_dependency(fvg_h1_fixture):
    """
    Mutate bar at index K+1. Assert labels at indices 0..K are unchanged.
    Run for K = len(df) // 2 (middle of dataset).
    """
    labeller = ValidFVGLabeller()
    labels_before = labeller.label(fvg_h1_fixture.copy())
    mutated = fvg_h1_fixture.copy()
    mutated.iloc[len(mutated) // 2 + 1, :] = 9999.0  # corrupt future bar
    labels_after = labeller.label(mutated)
    # Labels before the mutation point must be identical
    cutoff = len(mutated) // 2
    pd.testing.assert_series_equal(labels_before.iloc[:cutoff], labels_after.iloc[:cutoff])
```

This is the mandatory pytest fixture that asserts no lookahead. Must pass before Phase 7.

### Integration test

In `tests/data/test_pipeline.py` (extend existing):

```python
def test_pipeline_valid_fvg_labeller(spy_h1_fixture):
    """build_pipeline(labeller_name='fvg_valid') on 500-row fixture returns valid datasets."""
    train_ds, val_ds, test_ds, weights = build_pipeline(labeller_name="fvg_valid")
    assert weights.shape == (3,)
    assert abs(weights.sum().item() - 3.0) < 0.01
```

### Sparsity gate test

In `tests/data/labels/test_valid_fvg.py`:

```python
def test_positive_rate_on_large_fixture(spy_h1_500row_fixture):
    """On 500-row fixture with injected known valid FVGs, positive rate > 0."""
    labeller = ValidFVGLabeller()
    labels = labeller.label(spy_h1_500row_fixture)
    n_pos = (labels != 0).sum()
    assert n_pos > 0, "No valid FVGs detected — check fixture or labeller"
```

### S/R and BOS unit tests

`test_sr.py`:
- `compute_pivot_levels` output NaN for first `lookback` rows.
- `pivot_high[i]` = known max of prior `lookback` bars for a hand-constructed fixture.
- Mutating bar `i`'s high does not change `pivot_high[i]` (only bars i+1... are affected).

`test_bos.py`:
- `bull_bos_recent[i]` = False when no close > prior 50-bar high in past 20 bars.
- `bull_bos_recent[i]` = True exactly when a BOS occurred in the defined window.
- `bull_bos_recent[i]` is based on bars before i (shift(1) correct).

---

## Edge cases and constraints

- **Boundary:** Label index = N+2. `i` ranges `1..n-3`. First 2 rows always label 0. Last 2 rows always label 0. The builder must assert `label[:2].eq(0).all()` and `label[-2:].eq(0).all()`.

- **ATR warmup:** ATR(14) is NaN for first 14 bars. S/R levels are NaN for first `sr_lookback` bars. BOS is NaN for first `swing_lookback` bars. All criteria produce NaN for their warmup period. The labeller must map any NaN criterion to False (criterion fails = label 0). Use `.fillna(False)` on all criterion masks before combining.

- **Gann edge case:** If `swing_high ≈ swing_low` (< 0.01 range), `swing_mid` collapses. Guard: `(swing_high - swing_low) >= 0.01`. If guard fails, mark `crit5` as False (cannot assess position).

- **Reaction candle at boundary:** If the geometric FVG's middle candle i = n-2, then N+2 = n, which is out of bounds. The `i` range `1..n-3` prevents this — the loop never reaches n-2.

- **Conflict resolution:** If the same N+2 index is the label position for both a bull and a bear FVG (geometrically possible if two different patterns converge), bull wins (same as raw labeller convention). Note: this is extremely rare on H1 SPY.

- **Vectorisation constraint:** No per-bar Python loops. All criteria computed as array operations. The rolling-argmin for priority (criterion #4) is handled by the "label all passing" mode — no loop needed.

- **No pandas_ta dependency:** ATR computed inline. Do not introduce new dependencies.

- **Mitigation module is inference-only:** The body-close mitigation function from research (`is_mitigated`) is NOT implemented in this plan's scope. It is Model B territory. Do not add it to any training-path code. If builder sees a natural place to put it, create `src/data/labels/mitigation.py` as a stub with docstring only, no implementation.

---

## Rollback plan

If κ < 0.6 after full re-annotation and the disagreement analysis reveals the 6-criteria rule is fundamentally unworkable within the timeline:

1. Restore `data/gold_labels.csv` from `data/gold_labels_v1_geometric.csv` (backup made in Phase 5).
2. Switch pipeline back to `labeller_name="fvg_raw"` (the raw labeller is never deleted).
3. Re-run `build_pipeline(labeller_name="fvg_raw")` — all existing tests still pass (Phase 3 preserved the old labeller).
4. Document the attempt and κ result in the Status Update 2 notebook. The thesis gains methodological evidence even from a negative result.

No code needs to be deleted or reverted — the registry supports both labellers simultaneously.

---

## Risk register

| Risk | Likelihood | Blast radius | Reversibility | Mitigation |
|------|------------|--------------|---------------|------------|
| n_pos < 75 on 13,500-candle dataset (too sparse) | Medium | Gold annotation effort wasted, model may not converge | High — rollback in place, raw labeller preserved | Sparsity gate (Phase 4) blocks annotation start. Criterion relaxation decision tree applied first. |
| κ < 0.6 after re-annotation (guide still ambiguous) | Medium | Phase 4 model blocked | Medium — re-annotate subset | Investigate 3 worst disagreements. Update guide. Re-annotate those candles only. Surface to user before further action. |
| BOS criterion trivially satisfied (20-bar lookback too wide) | Medium | Criterion #6 adds no filtering signal | Low — parameter tweak | Ablation column `crit6_bos` shows pass rate. If > 90%, narrow bos_lookback further in ablation run. |
| S/R criterion too coarse (every bar is "near" an S/R level) | Medium | Criterion #3 adds no filtering signal | Low — parameter tweak | Ablation column `crit3_sr` shows pass rate. If > 90%, increase `sr_lookback` from 5 to 10 bars. |
| Vectorisation bug in rolling-priority logic | Low | Wrong label distribution; thesis claims invalid | Medium — fix and rerun | Isolated unit test with hand-computed expected values. Mandatory pre-integration. |
| May 17 deadline pressure — annotation not complete in time | Medium | Status Update 1 lacks kappa measurement | Medium — partial result is acceptable | Phase 1–4 can complete before May 17. Annotation and kappa result documented as "in progress" in SU1. |
| Chart overlays in annotation notebook crash at dataset boundaries | Low | Annotator blocked | Easy | Window clipping guard already in `_display_candle_chart` (existing code). Extend to 30-bar window with same `max(0, ...)` / `min(n, ...)` guards. |

---

## Pre-mortem

Imagining it is August 2026 and the valid FVG labeller failed:

1. **The sparsity gate was passed with n_pos = 160 (barely above 150), but the model still didn't converge.** Root cause: 160 positives spread over 13,500 candles, with stride=1 on train, created ~160 positive windows out of ~9,000 total — 1.8% positives. Even with weighted loss, gradient signal was too weak for LSTM to learn the minority class. Should have required n_pos >= 300 before proceeding.

2. **Cohen's kappa was 0.61 (just above gate) but annotator disagreements were concentrated in criterion #3 (S/R confluence).** The "S/R level near the zone" check was too subjective. The annotation guide showed static reference images but the annotator's judgment of "near" differed from the programmatic 0.5 ATR threshold by a systematic ±1 bar. Should have added a programmatic S/R overlay directly showing whether `crit3_sr` is True or False for the candidate, not just the raw S/R lines.

3. **The BOS criterion (20-bar lookback) was satisfied for 87% of geometric FVGs, making it nearly a no-op.** Root cause: SPY H1 2018–2024 has frequent structure breaks. The 20-bar window was too wide. Should have used 10 bars and checked if this improved discrimination before locking the parameter.

4. **The N+2 label index caused the Status Update 1 notebook to be inconsistent** with the existing 01-data-understanding.ipynb which documented the N+1 convention. The inconsistency was not caught before peer review because the plan did not include a "search for N+1 mentions in notebooks and update them" task.

5. **Gold re-annotation took 3.5 hours instead of 2.5 because the 30-bar chart window had performance issues** rendering swing markers for every annotated candle. Should have pre-computed swing marker positions as a DataFrame and passed them to the display function, rather than computing them live in each chart render call.

---

## Out of scope (this plan)

- Mitigation function implementation (`is_mitigated`) — inference-only, Phase 4/5 territory.
- Model B architecture — separate plan.
- OB / BOS / CHoCH / Liquidity labellers.
- Multi-timeframe features.
- Notebook `01-data-understanding.ipynb` — not modified in this plan. Update N+1 → N+2 references there separately.
- Live / real-time data path.
- SMOTE or other oversampling — evaluate after seeing actual positive count from Phase 4.
- `pandas_ta` dependency — ATR computed inline.
- XGBoost or model training — Phase 4.

---

## Phase 4 handoff contract

After all 8 phases complete and κ ≥ 0.6:

**Active labeller:** `"fvg_valid"` in registry. `build_pipeline(labeller_name="fvg_valid")` is the canonical Phase 4 entry point.

**Label encoding (unchanged from Phase 3):**
- 0 = none
- 1 = bullish valid FVG
- 2 = bearish valid FVG

**Label index:** N+2. Window of 60 bars ending at the label bar includes the full 3-candle pattern (bars N-1, N, N+1) plus the reaction candle (N+2) as the final bar. The model sees the reaction candle as part of its input.

**New files on disk:**
- `data/gold_labels.csv` — 75-candle re-annotation under 6-criteria definition. κ ≥ 0.6.
- `data/gold_labels_v1_geometric.csv` — archived original labels (κ = 0.284).
- `.nb-suite/analysis/criterion-ablation-<date>.csv` — per-criterion pass counts.
- `data/processed/spy_h1.parquet` — regenerated with `raw_label` from `ValidFVGLabeller`.

**Positive rate expected:** 1.5–3.5% (subject to sparsity gate result from Phase 4).

**Ablation baseline:** `build_pipeline(labeller_name="fvg_raw")` runs the geometric labeller for comparison. Expected positive rate: ~8–12%.

---

## What to flag if found during build

- Any criterion producing a pass rate > 90% on the full dataset — it is not filtering. Report before proceeding to gold annotation.
- `label[:2].any()` or `label[-2:].any()` non-zero — boundary contract violated.
- ATR values of 0 or NaN after warmup period — data integrity issue in processed parquet.
- `LABELLERS["fvg_valid"]` not in registry after import — `__init__.py` import chain broken.
- Any N+1 references in test assertions — must be N+2 for ValidFVGLabeller.
- Reaction candle strict close-inside rate < 10% of geometric FVGs — criterion #2 is too restrictive; surface before gold annotation.
