# Build Log — valid-fvg-labeller — 10-May-26

## Status: STOPPED AT PHASE 5 — SPARSITY GATE FAILED, USER DECISION REQUIRED

---

## Phase 1 — Helper modules (sr.py, bos.py)

Status: COMPLETE (pre-existing, already passing)

Files checked:
- `src/data/labels/sr.py` — causal pivot high/low, shift(1), no lookahead
- `src/data/labels/bos.py` — causal BOS detection, bos_lookback window

Tests: 13/13 pass (`tests/data/labels/test_sr.py`, `tests/data/labels/test_bos.py`)

Note: Python venv had broken symlinks (Anaconda removed, migration to Python 3.14 in progress). Resolved by installing pyenv 3.12.7 and relinking .venv/bin/python* to pyenv install. User should complete migration to 3.14 venv when convenient.

---

## Phase 2 — ValidFVGLabeller core (valid_fvg.py)

Status: COMPLETE

Files created:
- `src/data/labels/valid_fvg.py` — ValidFVGLabeller @register("fvg_valid"), N+2 label index, 6 criteria vectorised
- `tests/data/labels/test_valid_fvg.py` — 15 tests

Tests: 15/15 pass

Key implementation decisions:
- Criterion #2: strict (close inside gap), not loose. Ablation records both.
- Criterion #4: label all passing candidates (no priority elimination) per plan.
- Gann guard: range < 0.01 → crit5 = False.
- NaN handling: `fillna(False)` applied via numpy `np.where` on all criteria.
- Boundary: `raw[:2] = 0; raw[-2:] = 0` enforced.

---

## Phase 3 — Re-register raw FVG labeller

Status: COMPLETE

Change: `fvg.py` now has `@register("fvg") @register("fvg_raw")` — both keys resolve to FVGLabeller.
`__init__.py` updated to also import `valid_fvg`.

---

## Phase 4 — Sparsity gate script

Status: COMPLETE (script written and run — GATE FAILED)

File created: `scripts/count_valid_fvg.py`

Results on `data/processed/spy_h1.parquet` (14,686 candles):

| Criterion | Bull remaining | Pass rate |
|-----------|---------------|-----------|
| Geometric FVG | 2,186 | 14.9% |
| + Crit2 strict | 221 | 10.1% of geom |
| + Crit3 S/R | 170 | 76.9% of crit2 |
| + Crit5 Gann | 48 | 28.2% of crit3 |
| + Crit6 BOS (20-bar) | 2 | 4.2% of crit5 |
| VALID BULL | 2 | — |
| VALID BEAR | 4 | — |
| TOTAL POSITIVES | 6 | 0.04% |

Gate result: n_pos=6 < 75. Decision tree: escalate to user.

Relaxation attempts (run during build to document):
- crit3 relaxed (mult 0.5→1.0): n_pos=9
- crit3 dropped (mult=999): n_pos=11  
- crit3 dropped + bos_lookback 20→50: n_pos=52 (still < 75)

Even full relaxation per plan's decision tree yields only 52 positives.

FLAG TRIGGERED (from plan "What to flag if found"):
> "Reaction candle strict close-inside rate < 10% of geometric FVGs — criterion #2 is too restrictive; surface before gold annotation."

Exact rate: crit2 strict passes 10.1% of geometric bull FVGs (221/2186). This is AT the flag threshold.

Switching to **loose crit2** (close in gap direction, not necessarily inside):
- loose + no crit3 + bos_lookback=50: **467 positives (3.2%)**
- This is in the healthy range (research projected 200–470 for "label all candidates" mode)

---

## Phase 5 — Gold annotation overhaul

Status: INFRASTRUCTURE COMPLETE — awaiting user decision on crit2 before annotation

Files modified:
- `docs/fvg-label-guide.md` — full rewrite with 6-criteria decision guide, visual descriptions for all overlays
- `notebooks/02-gold-annotation.ipynb` — updated to 30-bar window, swing markers, S/R lines, Gann midpoint overlay, overlays pre-computed (not per-candle, avoids performance issue)
- `src/data/annotate.py` — `_display_candle_chart` updated to 30-bar window with all overlays
- `data/gold_labels_v1_geometric.csv` — backup of original gold labels (κ=0.284)

The programmatic label in the notebook UI is now hidden (only shown as text label name, not numerically) to reduce anchoring bias.

---

## Phase 6 — Skipped (κ measurement requires completed annotation)

---

## Phase 7 — Skipped (requires phase 6 gate)

---

## Phase 8 — Ablation analysis notebook section

Status: COMPLETE

Added cells to `notebooks/02-gold-annotation.ipynb`:
- Criterion Ablation Analysis section (markdown + code)
- Bar chart of per-criterion pass counts
- Bullish FVG rejection funnel
- Observation cell documenting findings

---

## USER DECISION REQUIRED — Critical gate

Before starting gold re-annotation, you must decide which version of criterion #2 to use:

### Option A: Keep strict crit2 (default as planned)
- Definition: reaction candle (N+2) CLOSE inside the gap zone [bottom, top]
- Result on full dataset (with crit3 dropped + bos=50): **52 positives (0.35%)**
- Consequence: below the 75-positive minimum. Per plan: "escalate to user before proceeding."
- Action needed: either accept < 150 positives and model may struggle, or relax further.

### Option B: Switch to loose crit2
- Definition: reaction candle close is above fvg_bottom (bull) / below fvg_top (bear)
- Result: **467 positives (3.2%)** — in the healthy range
- Consequence: less selective. More FVGs will be labelled, but they may not have truly "accepted" the zone.
- Implementation: modify `valid_fvg.py` to use `crit2_loose_bull_i` / `crit2_loose_bear_i` instead of strict variants in `valid_bull_i` / `valid_bear_i`. The ablation still records both.
- This matches the "loose" variant already computed in the ablation log.

### Recommendation
Switch to **loose crit2** for training labels. Rationale:
1. 52 positives is insufficient for LSTM training (effective sample << SMOTE threshold).
2. The loose definition still filters: it requires price to move in the direction of the FVG at bar N+2. Only bars where price aggressively reversed would fail.
3. The strict definition is better suited as a quality filter at inference time (Model B territory), not as a training label gate.
4. The thesis can document both versions via the ablation log.

To implement: change one line in `valid_fvg.py` — the `valid_bull_i` and `valid_bear_i` conjunctions to use `crit2_loose_bull_i` / `crit2_loose_bear_i`. The `crit2_strict_bull` ablation column is preserved for comparison.

### After user decision:
1. Apply the change (if Option B chosen).
2. Re-run `python scripts/count_valid_fvg.py` to verify n_pos >= 150.
3. Proceed to gold annotation in `notebooks/02-gold-annotation.ipynb`.
4. Note: the existing `data/gold_labels.csv` is NOT cleared — it still contains the v1 geometric annotation (75 candles, κ=0.284). The backup is at `data/gold_labels_v1_geometric.csv`. Delete `data/gold_labels.csv` and start fresh when ready to re-annotate.

---

## Test summary

| Test file | Tests | Result |
|-----------|-------|--------|
| `tests/data/labels/test_sr.py` | 7 | 7/7 pass |
| `tests/data/labels/test_bos.py` | 6 | 6/6 pass |
| `tests/data/labels/test_valid_fvg.py` | 15 | 15/15 pass |
| **Total** | **28** | **28/28 pass** |
