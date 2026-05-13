# Build Log — notebook-03-expand

**Date:** 2026-05-13
**Task:** Expand notebooks/03-status-update-1.ipynb from 15 cells / 3 charts to 41 cells / 15 charts.

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Total cells | 15 | 41 |
| Code cells | 4 | 15 |
| Markdown cells | 11 | 26 |
| Figures | 3 (Fig 1-3) | 15 (Fig 1-15) |

## New cells added (26 cells)

### Pedagogical visuals
- Fig 5: FVG annotated diagram — bullish + bearish 3-candle with N-1/N/N+1 labels, gap zone shaded
- Fig 6: Data pipeline flow — Alpaca 1-min → H1 RTH → ValidFVG label → split → windows
- Fig 7: Temporal split timeline — 2016-2025 with train/val/test bands
- Fig 8: Sliding window concept — 3 overlapping windows on synthetic SPY, diamond at label bar
- Fig 15: Model input transformation cartoon — XGB / LSTM / CNN-LSTM processing paths

### Results visuals
- Fig 9: Per-class F1 grouped bar — None / Bull / Bear x XGB / LSTM / CNN-LSTM
- Fig 10: Bootstrap CI forest plot — horizontal whiskers with no XGB-DL overlap
- Fig 11: Focal loss ablation — gamma {1,2,3} vs WeightedCE (LSTM G3)
- Fig 12: Window sweep line — W=30,45,60,90,120 vs val+test F1 with effective_n annotations
- Fig 13: Regularisation ablation — control/no-dropout/no-L2/both-off for LSTM + CNN-LSTM
- Fig 14: Optuna trial scatter — CNN-LSTM trials, coloured by lstm_hidden

Each figure has a preceding markdown framing cell and a following caption cell.

## Execution

- `jupyter nbconvert --execute` ran clean (0 errors) on first attempt after 3 targeted bug fixes
- Bug fixes: CNN bootstrap JSON schema mismatch, focal loss label `\n` → SyntaxError, Optuna relative path
- HTML export: `notebooks/03-status-update-1.html` (513 KB)

## Data sources used

| Figure | Source |
|--------|--------|
| 5 | Synthetic |
| 6 | Synthetic (bar counts from pipeline logs) |
| 7 | Synthetic (dates from CLAUDE.md split config) |
| 8 | Synthetic |
| 9 | reports/rigor/2026-05-13/G10/bootstrap_ci_xgb.json + lstm + cnn |
| 10 | Same as Fig 9 |
| 11 | reports/rigor/2026-05-13/G3/focal_ablation.json |
| 12 | reports/rigor/2026-05-13/G7/window_sweep_results.json |
| 13 | reports/rigor/2026-05-13/G9/reg_ablation_summary.json + cnn_lstm_G9 md files |
| 14 | checkpoints/cnn_lstm/optuna_2026-05-12.db |
| 15 | Synthetic (architecture params from CLAUDE.md) |

## Files touched

- `notebooks/03-status-update-1.ipynb` — expanded in-place
- `notebooks/03-status-update-1.html` — exported
- `scripts/expand_nb03.py` — build script (can be deleted)
- `reports/figure05_fvg_diagram.html` through `reports/figure15_model_cartoon.html`
