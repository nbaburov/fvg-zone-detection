# Build Log — notebooks/03-status-update-1.ipynb

**Date:** 2026-05-13
**Agent:** nb-notebook (sonnet) + nb-humanize

## Summary

Created `notebooks/03-status-update-1.ipynb` — presentation-style status update notebook for May 17 peer review. Education mode. 9 sections, 4 figures, ~18 cells total.

## Source material read

- `.nb-suite/changelogs/12-May-26/final-rigor-sprint-and-data-extension.md`
- `.nb-suite/changelogs/13-May-26/cnn-lstm-sprint.md`
- `.nb-suite/changelogs/13-May-26/config-refactor-phase1.md`
- `.nb-suite/research/8-May-26/smc-library-validation.md`
- `.nb-suite/research/12-May-26/mps-gpu-fix.md`
- `docs/models-status.md`
- `docs/data-model.md`
- `reports/rigor/2026-05-13/G2/multiseed_summary.json`
- `reports/rigor/2026-05-13/G3/focal_ablation.json`
- `reports/rigor/2026-05-13/G5/shap_xgb.json`
- `reports/rigor/2026-05-13/cnn_lstm_G2/multiseed_summary.json`
- `reports/inspect/2026-05-13_004532/summary.md`
- `git log --oneline`

## Structure

| Section | Type | Content |
|---------|------|---------|
| Abstract | markdown | One-para summary |
| 1. What we're building | markdown | Pitch, FVG plain English |
| Figure 1 | code | Load LSTM inspect overlay HTML (IFrame) |
| 2. Why DL | markdown | Geometry noise, 6 criteria, ValidFVG, XGB justification |
| 3. Data | markdown | Alpaca, two labellers, gold annotation, splits table |
| 4. Models | markdown | Results table, XGBoost finding, architecture notes |
| Figure 2 | code | Bar chart: model F1 comparison (from JSON) |
| 5. HP search + ablations | markdown | G1 HP tables, G3-G9 ablation table, focal loss note |
| Figure 3 | code | SHAP bar chart (from JSON) |
| 6. What went wrong | markdown | 7 honest failures |
| 7. What worked | markdown | 6 successes |
| Figure 4 | code | Timeline scatter (from git log) |
| 8. Current state | markdown | Status table, leaderboard |
| 9. Where next | markdown | Deadline table, 5 planned experiments |
| Final Summary | markdown | Concrete numbers, limitations, LO evidence, APA refs |

## Figures

- Figure 1: Plotly timeline overlay (IFrame from existing inspect HTML)
- Figure 2: Bar chart from `G2/multiseed_summary.json` + `cnn_lstm_G2/multiseed_summary.json`
- Figure 3: SHAP horizontal bar from `G5/shap_xgb.json`
- Figure 4: Timeline scatter from hardcoded git log events

All figures saved to `reports/figure0{2,3,4}_*.html`.

## Execution

Executed clean: `jupyter nbconvert --execute` completed without errors.
HTML export: `notebooks/03-status-update-1.html` (345KB).

## nb-humanize pass

9 markdown cells rewritten. Key changes:
- Removed inline bold headers (pattern 16) in sections 1, 7, Final Summary
- Removed "structured rigor process" throat-clear opener in section 5
- Removed "each model was selected for a specific reason" setup in section 4
- Tightened abstract from passive to active
- Replaced "no labeled FVG dataset exists publicly. Everything had to be built" drama with direct statement
- Removed em dash overuse (pattern 14) — replaced with comma or colon
- Focal loss paragraph: cut "worth a note" filler, moved straight to claim
- Final Summary restructured to drop bold-header bullets, use prose for limitations/next steps
