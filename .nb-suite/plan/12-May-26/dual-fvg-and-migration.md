# Plan: Dual-FVG Baselines + ValidFVG Migration Sub-Sprint

**Position in sprint:** Fits between Phase C (ValidFVG baselines on 2016–2025 done) and Phase D (full rigor rerun on ValidFVG only).

**Objective:** Produce a side-by-side raw-FVG vs ValidFVG comparison table locked in a report, then fully migrate the codebase to ValidFVG as canonical target. Phase D then runs on a clean, single-target codebase.

**Sub-sprint agent:** `@nb-build` orchestrates. Spawns `@nb-test` after Part A, `@nb-review` after Part B.

---

## Current state (input conditions)

- `data/processed/spy_h1_{train,val,test}.parquet` — ValidFVG-labelled (~3% positive). These are the Phase C splits.
- No raw-FVG parquets on disk for the 2016–2025 range.
- `data/processed/class_weights.json` — ValidFVG weights (none=0.024, bull=1.232, bear=1.744).
- `data/processed/spy_h1_labeled.parquet` — copy of `spy_h1.parquet`, currently ValidFVG-labelled.
- Existing Phase C checkpoints: `checkpoints/lstm/lstm_v2_seed42.pt`, `checkpoints/xgboost/xgb_v2_seed42.ubj` (ValidFVG, 2016–2025).
- `src/data/labels/__init__.py` registers `"fvg"` / `"fvg_raw"` → `FVGLabeller`, `"fvg_valid"` → `ValidFVGLabeller`.
- `build_pipeline()` default `labeller_name="fvg"` (stale default — changes in Part B).
- Tests in `tests/data/test_pipeline.py` hardcode `labeller_name="fvg"` in 4 places.

---

## Part A — Dual-FVG Baselines

**Goal:** train LSTM seed42 + XGB seed42 on raw FVG labels from the SAME 2016–2025 data range, capture F1, write comparison report.

### A1 — Generate raw-FVG splits (est. 10 min)

Run `build_pipeline(labeller_name="fvg", ...)` with a separate output dir so ValidFVG splits are NOT overwritten.

**New script:** `scripts/data/build_rawfvg_splits.py`

```
Purpose: build raw-FVG labelled splits from 2016-2025 data.
Output dir: data/processed/rawfvg/
Files written:
  data/processed/rawfvg/spy_h1_train.parquet
  data/processed/rawfvg/spy_h1_val.parquet
  data/processed/rawfvg/spy_h1_test.parquet
  data/processed/rawfvg/class_weights_rawfvg.json
```

Implementation note: call `build_pipeline(labeller_name="fvg", ...)` with `PROCESSED_DIR` patched to `data/processed/rawfvg`. The raw SPY data (`spy_h1.parquet`) is already on disk; `use_cache=True` avoids Alpaca redownload.

Class balance expected: ~75% none / ~15% bull / ~10% bear (matching old 2018–2024 splits). Confirm in script output before proceeding.

**Decision gate A1:** If raw-FVG positive rate < 5% or > 40%, STOP — the data on disk may be mislabelled. Expected range: 22–28% positive.

### A2 — Train LSTM raw-FVG seed42 (est. 25 min compute)

**Invocation:**
```bash
.venv/bin/python scripts/training/train_lstm.py \
  --splits data/processed/rawfvg \
  --seed 42 \
  --label rawfvg \
  --device cpu
```

The `--splits` flag already exists (Phase C). The `--label` flag is new — it controls checkpoint suffix and class weights lookup.

**Changes to `scripts/training/train_lstm.py`:**
- Add `--label {rawfvg,validfvg}` argument (default: `validfvg`).
- Load class weights from `{splits_dir}/class_weights_{label}.json` when `--label` is set, else from `{splits_dir}/class_weights.json` (backwards compat).
- Checkpoint path: `checkpoints/lstm/lstm_seed{N}_{label}.pt` when `--label` != `validfvg`, else `checkpoints/lstm/lstm_seed{N}.pt` (preserves existing naming for ValidFVG).
- Meta JSON path: same suffix pattern as checkpoint.

**Output files:**
```
checkpoints/lstm/lstm_seed42_rawfvg.pt
checkpoints/lstm/lstm_seed42_rawfvg.meta.json
```

**Decision gate A2:** If raw-FVG LSTM F1 < 0.5 or > 0.98, pause and inspect. Expected range: 0.78–0.90 based on archive numbers.

### A3 — Train XGB raw-FVG seed42 (est. 8 min compute)

**Invocation:**
```bash
.venv/bin/python scripts/training/train_xgboost.py \
  --splits data/processed/rawfvg \
  --seed 42 \
  --label rawfvg
```

**Changes to `scripts/training/train_xgboost.py`:**
- Same `--label` flag pattern as LSTM above.
- Class weights from `{splits_dir}/class_weights_{label}.json`.
- Checkpoint: `checkpoints/xgboost/xgb_seed42_rawfvg.ubj`.

**Output files:**
```
checkpoints/xgboost/xgb_seed42_rawfvg.ubj
checkpoints/xgboost/xgb_seed42_rawfvg.meta.json
```

### A4 — Collect naive baselines for both label sets (est. 5 min)

**New script:** `scripts/eval/naive_baselines.py`

Computes majority-class F1 and uniform-random F1 for both label sets. Reads from:
- `data/processed/spy_h1_test.parquet` (ValidFVG)
- `data/processed/rawfvg/spy_h1_test.parquet` (raw FVG)

Prints + writes to `reports/rigor/2026-05-12/baselines/naive_baselines.json`.

### A5 — Write comparison report (est. 5 min)

**New script:** `scripts/eval/dual_fvg_compare.py`

Reads:
- `checkpoints/lstm/lstm_seed42_rawfvg.meta.json`
- `checkpoints/lstm/lstm_v2_seed42.meta.json` (ValidFVG — from Phase C)
- `checkpoints/xgboost/xgb_seed42_rawfvg.meta.json`
- `checkpoints/xgboost/xgb_v2_seed42.meta.json` (ValidFVG)
- `reports/rigor/2026-05-12/baselines/naive_baselines.json`

Writes:
```
reports/rigor/2026-05-12/baselines/dual_fvg_compare.json
reports/rigor/2026-05-12/baselines/dual_fvg_compare.md   (human-readable table)
```

Target table structure:
```
| Model         | Label     | Pos rate | Test Macro F1 | Bull F1 | Bear F1 |
|---------------|-----------|----------|---------------|---------|---------|
| LSTM seed42   | raw FVG   | ~25%     | TBD           | TBD     | TBD     |
| LSTM seed42   | ValidFVG  | ~3%      | 0.5983        | 0.4045  | 0.4085  |
| XGB seed42    | raw FVG   | ~25%     | TBD           | TBD     | TBD     |
| XGB seed42    | ValidFVG  | ~3%      | 0.5222        | 0.4356  | 0.2094  |
| Naive majority| raw FVG   | ~25%     | TBD           | —       | —       |
| Naive majority| ValidFVG  | ~3%      | ~0.328        | —       | —       |
```

**Decision gate A5 (HARD GATE before Part B):**
- If raw-FVG LSTM F1 > ValidFVG LSTM F1 by more than 0.40 AND raw-FVG F1 > 0.92: pause. Numbers this extreme likely indicate a labelling pipeline error in the rawfvg splits (e.g. wrong weights file, ValidFVG splits accidentally fed). Inspect before migrating.
- Normal expectation: raw-FVG F1 higher (~0.83–0.87) than ValidFVG (~0.60) by ~0.25–0.30. This is expected (easier problem).

---

## Part B — Migrate to ValidFVG-only canonical target

Only proceed after Part A comparison report is written and gate A5 passes.

### B1 — Update pipeline default (2 min)

**File:** `src/data/pipeline.py`

Change:
```python
def build_pipeline(
    labeller_name: str = "fvg",
```
To:
```python
def build_pipeline(
    labeller_name: str = "fvg_valid",
```

### B2 — Class weights file naming (5 min)

**File:** `src/data/pipeline.py`

Change the weights persistence path from:
```python
weights_path = out_dir / "class_weights.json"
```
To:
```python
weights_path = out_dir / f"class_weights_{labeller_name}.json"
# Also write the legacy path for backwards compat with existing checkpoint consumers
legacy_path = out_dir / "class_weights.json"
```

Write both files — `class_weights_fvg_valid.json` and `class_weights.json` — so existing checkpoint `.meta.json` references and training scripts that haven't been updated yet don't break silently.

### B3 — Update tests (15 min)

**File:** `tests/data/test_pipeline.py`

Four occurrences of `labeller_name="fvg"` → `labeller_name="fvg_valid"`. Search pattern:
```
grep -n 'labeller_name="fvg"' tests/data/test_pipeline.py
```
Lines: 80, 99, 114, 154 (confirmed from read).

Note: the fixture `spy_9candle_fvg` in `tests/conftest.py` is a data fixture, not a labeller name reference — leave it alone.

**File:** `tests/data/test_fvg_labeller.py`

Read and check whether any tests assert raw-FVG-specific class distributions (~25% positive rate). If so, those tests are validating the `FVGLabeller` class which stays in codebase — tests are fine as-is. Do NOT delete these tests; they validate the labeller implementation that stays for historical comparison.

### B4 — FVGLabeller fate (5 min, decision + comment only)

**Decision:** KEEP `src/data/labels/fvg.py` and both registry entries (`"fvg"`, `"fvg_raw"`). Rationale:
- Raw-FVG checkpoint loading (`lstm_seed42_rawfvg.pt`) requires the labeller to be importable for `SMCWindowDataset` construction during inspect.
- Removing it saves ~72 lines but breaks the historical comparison path.
- Mark the file with a header comment:

```python
# NOTE: FVGLabeller is kept for historical baseline comparison only.
# The canonical training target is ValidFVGLabeller ("fvg_valid").
# Do not use "fvg" or "fvg_raw" for new training runs.
```

### B5 — Retire `spy_h1_labeled.parquet` (5 min)

`data/processed/spy_h1_labeled.parquet` is a copy of `spy_h1.parquet` with no additional transformation. Its purpose was the old `persist_labels.py` workflow before the pipeline was unified.

**Action:** Delete the file. Update `scripts/data/persist_labels.py` to print a deprecation notice and exit 0 cleanly rather than erroring.

```python
# In scripts/data/persist_labels.py, replace main() body:
def main() -> int:
    print(
        "DEPRECATED: spy_h1_labeled.parquet is no longer used. "
        "The canonical labelled dataset is data/processed/spy_h1.parquet, "
        "produced by build_pipeline(labeller_name='fvg_valid')."
    )
    return 0
```

`src/data/persist_labels.py` does not exist (confirmed) — no action needed there.

### B6 — Update CLAUDE.md (5 min)

**File:** `CLAUDE.md`

Change the stale line in "Current label target" (in the Stack section):
```
# current line references raw FVG as training target
```
To correctly state: `ValidFVGLabeller ("fvg_valid")` @ N+2 is the canonical training target. Raw FVG kept for historical comparison only.

### B7 — Update `docs/models-status.md` (10 min)

Current doc header says:
```
> **Current label target (training):** `FVGLabeller` — raw geometric 3-candle FVG at index N+1.
```

Replace with:
```
> **Current label target (training):** `ValidFVGLabeller` ("fvg_valid") — 6-criteria SMC FVG @ N+2.
> Raw `FVGLabeller` ("fvg") results are preserved in the Dual-FVG Baselines section as historical comparison. Not a regression — different problem.
```

Add a new section "Dual-FVG Baseline Comparison" that embeds the table from `dual_fvg_compare.md` (copy/paste, with a note pointing to the report file). This section goes between the existing "V2 Dataset" section and the Phase D results when they arrive.

Add a note to the archive annotation point (see B8).

### B8 — Annotate archive doc (3 min)

**File:** `.nb-suite/archive/2026-05-12-pre-rerun-snapshot.md`

Add a note at the top of the "Key findings worth remembering" section:

```
> **Annotation added 12-May-26 post-migration:** The raw-FVG numbers in this table
> (LSTM 0.836, XGB 0.637) were produced on `FVGLabeller` (geometric, ~25% positive rate).
> ValidFVG results (LSTM ~0.60, XGB ~0.52) are on `ValidFVGLabeller` (6-criteria, ~3% positive).
> These are not regression — they measure different problems. Do not compare directly.
```

---

## Decision gates summary

| Gate | Condition | Action if triggered |
|------|-----------|---------------------|
| A1 | raw-FVG positive rate < 5% or > 40% | STOP — pipeline error, inspect splits |
| A2 | LSTM raw-FVG F1 < 0.50 or > 0.98 | PAUSE — inspect checkpoint + labeller |
| A5 (HARD) | raw-FVG F1 > ValidFVG F1 by > 0.40 AND raw-FVG F1 > 0.92 | STOP before Part B — likely data error |

---

## Wall-clock estimates

| Step | Estimate |
|------|---------|
| A1 — Generate raw-FVG splits | 10 min (CPU, no Alpaca download) |
| A2 — Train LSTM seed42 rawfvg | 20–30 min (CPU, seed42 known to be fast) |
| A3 — Train XGB seed42 rawfvg | 5–8 min |
| A4 — Naive baselines | 2 min |
| A5 — Compare script | 5 min |
| B1–B4 — Code + config changes | 15 min |
| B5–B8 — Doc + annotation updates | 20 min |
| **Total** | **~80–100 min** |

---

## Test strategy

### New tests needed

**`tests/data/test_pipeline_rawfvg.py`** (new file, ~30 lines):
- `test_rawfvg_splits_exist()` — asserts `data/processed/rawfvg/spy_h1_{train,val,test}.parquet` exist after `build_rawfvg_splits.py` runs.
- `test_rawfvg_class_weights_json()` — reads `class_weights_rawfvg.json`, asserts keys 0/1/2, bull+bear weights > none weight (inverse-freq → minority gets higher weight).
- `test_rawfvg_positive_rate()` — reads train parquet, asserts positive rate 0.20–0.35 (raw FVG expected ~25%).

**`tests/scripts/test_dual_fvg_compare.py`** (new file, ~20 lines):
- `test_compare_report_written()` — runs `dual_fvg_compare.py`, asserts `dual_fvg_compare.json` exists and has keys for all 4 model-label combinations.

### Existing tests needing update

**`tests/data/test_pipeline.py`** — 4 occurrences of `labeller_name="fvg"` → `"fvg_valid"`. After change, tests remain structurally identical but now exercise the canonical labeller.

**No other test files need changes.** `tests/data/test_fvg_labeller.py` tests `FVGLabeller` directly — stays as-is since the class is kept. `tests/data/labels/test_valid_fvg.py` already tests `ValidFVGLabeller` — no change.

### Regression check after B3

After updating `test_pipeline.py`, run full suite:
```bash
.venv/bin/python -m pytest tests/ -x -q
```
Expected: all pass. The pipeline logic is unchanged; only the default labeller is different.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Alpaca redownload triggered for rawfvg splits | Low | High (30+ min) | `use_cache=True` in `build_rawfvg_splits.py` — existing `spy_h1.parquet` on disk contains raw OHLCV; the labeller runs on that, no API call needed. Confirm `build_labelled_dataset(..., use_cache=True)` reads the cache before calling Alpaca. |
| `class_weights.json` naming collision | Medium | Medium | Two-file write strategy in B2 (`class_weights_fvg_valid.json` + legacy `class_weights.json`). Existing training scripts that hardcode `class_weights.json` continue to work. |
| Raw-FVG checkpoint naming conflicts with existing checkpoints | Low | Low | Suffix `_rawfvg` on all new checkpoints; ValidFVG checkpoints keep existing `_v2_seed42` naming. No collision possible. |
| XGB early-stop at round 3 repeats on raw-FVG splits (as seen in V2 ValidFVG run) | Low | Medium | Raw-FVG positive rate ~25% means mlogloss has genuine signal — early-stop-at-3 was a ValidFVG-specific failure (97% none). If it does happen, fall back to fixed `n_estimators=527` (from archive HP). |
| `train_lstm.py` `--label` flag breaks existing invocations | Low | Low | Default `--label validfvg` preserves current behaviour. Existing CI / test invocations don't pass `--splits` flag — they use the in-memory pipeline, unaffected. |
| `spy_h1_labeled.parquet` deletion breaks a downstream script | Low | Medium | `grep -r "spy_h1_labeled" scripts/ src/ tests/` before deleting. If any hit found, update that script to use `spy_h1.parquet` first. |
| Archive annotation clobbers git blame | None | None | Annotation is additive text; no existing content removed. |

---

## File manifest

### New files

```
scripts/data/build_rawfvg_splits.py
scripts/eval/naive_baselines.py
scripts/eval/dual_fvg_compare.py
tests/data/test_pipeline_rawfvg.py
tests/scripts/test_dual_fvg_compare.py
data/processed/rawfvg/spy_h1_train.parquet       (generated)
data/processed/rawfvg/spy_h1_val.parquet          (generated)
data/processed/rawfvg/spy_h1_test.parquet         (generated)
data/processed/rawfvg/class_weights_rawfvg.json   (generated)
checkpoints/lstm/lstm_seed42_rawfvg.pt            (generated)
checkpoints/lstm/lstm_seed42_rawfvg.meta.json     (generated)
checkpoints/xgboost/xgb_seed42_rawfvg.ubj        (generated)
checkpoints/xgboost/xgb_seed42_rawfvg.meta.json  (generated)
reports/rigor/2026-05-12/baselines/naive_baselines.json   (generated)
reports/rigor/2026-05-12/baselines/dual_fvg_compare.json  (generated)
reports/rigor/2026-05-12/baselines/dual_fvg_compare.md    (generated)
```

### Modified files

```
scripts/training/train_lstm.py         — add --label flag, class weights lookup, checkpoint naming
scripts/training/train_xgboost.py      — same as above
src/data/pipeline.py                   — default labeller "fvg_valid", dual class_weights write
src/data/labels/fvg.py                 — add deprecation header comment
scripts/data/persist_labels.py         — replace body with deprecation notice
tests/data/test_pipeline.py            — 4× "fvg" → "fvg_valid"
CLAUDE.md                              — update "current label target" line
docs/models-status.md                  — update header, add dual-FVG section
.nb-suite/archive/2026-05-12-pre-rerun-snapshot.md  — add annotation note
```

### Deleted files

```
data/processed/spy_h1_labeled.parquet  — after confirming no downstream references
```

---

## Subagent ownership

**`@nb-build`** orchestrates the full sub-sprint.

Internal spawns (as per `@nb-build` default behaviour):
- **`@nb-test`** after Part A completes — runs `tests/data/test_pipeline_rawfvg.py` and `tests/scripts/test_dual_fvg_compare.py`, reports pass/fail.
- **`@nb-review`** after Part B completes — spec compliance check on: pipeline default change, test updates, class weights naming, deprecated script body, archive annotation.

No `@nb-research` needed — all decisions resolved in this plan.

---

## Handoff brief for `@nb-build`

**What you are building:** A two-part sub-sprint between Phase C and Phase D of the post-research rerun sprint.

**Part A inputs:**
- Raw OHLCV data already on disk: `data/processed/spy_h1.parquet` (do not redownload).
- Registered labellers: `"fvg"` → `FVGLabeller`, `"fvg_valid"` → `ValidFVGLabeller` (in `src/data/labels/__init__.py`).
- Existing split boundaries in `src/data/split.py` `SPLIT_BOUNDARIES` (train 2016–2021, val 2022, test 2023–2025).
- Phase C ValidFVG checkpoints: `checkpoints/lstm/lstm_v2_seed42.pt`, `checkpoints/xgboost/xgb_v2_seed42.ubj` — these are INPUTS to the comparison report, do not retrain them.

**Part A outputs required before Part B starts:**
- `reports/rigor/2026-05-12/baselines/dual_fvg_compare.md` must exist and contain F1 numbers for all 4 model-label pairs.
- Gate A5 must pass (human review the numbers before executing Part B).

**Part B constraints:**
- Never delete `.nb-suite/` artifacts.
- Never delete `src/data/labels/fvg.py` — add deprecation comment only.
- ValidFVG checkpoint naming (`lstm_v2_seed42.pt`, `xgb_v2_seed42.ubj`) is NOT changed — these are canonical and other scripts reference them by name.
- `class_weights.json` (no suffix) must remain as a legacy alias — write it alongside the new suffixed file, do not remove it.
- Run `.venv/bin/python -m pytest tests/ -x -q` after B3 and confirm all tests pass before proceeding to B4+.

**Device:** CPU only for all training (`--device cpu`). No MPS.

**Python binary:** `.venv/bin/python` throughout.

**Commit:** None. Plan ends here. User commits after reviewing Part B diff.
