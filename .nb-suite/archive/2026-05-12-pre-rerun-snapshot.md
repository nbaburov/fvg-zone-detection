# Archive — Pre-Rerun Snapshot (2026-05-12)

**Why this exists:** Before GPU fix + full rigor rerun on extended 2016–2025 dataset, document everything being deleted so reasoning and observations survive. Git holds the diffs; this doc holds the human summary.

**Trigger commit:** `6a37bc0` (docs sync, rigor sprint end-state). Next commits = cleanup + rerun.

---

## What's being deleted and why

### Checkpoints

All current checkpoints invalidated. Mix of two generations:
- **May 11 21:10** — `lstm_seed42.pt` (211k), `xgb_seed42.ubj` — old 2018–2024 splits, raw FVG labels.
- **May 12 00:36–00:54** — multi-seed LSTM + focal variants + XGB seeds — trained AFTER 2016–2025 extension commit `8f02b3d`. **Partial coverage** (focal only 2/5 seeds done; LSTM HP only 8/50 trials).

Reason for full wipe: rigor rerun requires consistent config (new GPU fix, new Optuna search space on full data, full 5-seed coverage). Mixed-generation checkpoints would confuse future readers.

| Path | Size | Splits | Labels | Notes |
|------|------|--------|--------|-------|
| `checkpoints/lstm/lstm_seed42.pt` | 211k | 2018–2024 | raw FVG | Original baseline, F1=0.824 |
| `checkpoints/lstm/lstm_seed{0,17,123,2024}.pt` | 23k each | 2016–2025 | raw FVG | Multi-seed, mean ~0.836 |
| `checkpoints/lstm/lstm_focal_g{1,2,3}_seed{17,42}.pt` | 24k each | 2016–2025 | raw FVG | Partial ablation (3 seeds hung on MPS) |
| `checkpoints/lstm/optuna_2026-05-12.db` | 152k | 2016–2025 | raw FVG | 8/50 trials only — hp space invalid for rerun |
| `checkpoints/xgboost/xgb_seed{0,17,42,123,2024}.ubj` | ~1.2M each | 2016–2025 | raw FVG | Multi-seed (F1=0.6365 ± 0.004) |
| `checkpoints/xgboost/optuna_2026-05-12.db` | 164k | 2016–2025 | raw FVG | 50/50 trials. Will rerun with GPU + new test |

### Reports

| Path | Action | Reason |
|------|--------|--------|
| `reports/rigor/2026-05-11_*` | delete | 28 dirs, ~75MB, all stale (old data) |
| `reports/rigor/v2_2016_2025/2026-05-12_*` | delete | partial new-data run, will rerun comprehensively |
| `reports/inspect/2026-05-11_*` | delete | Plotly overlays on stale checkpoints |

Total reclaim: ~76 MB reports + 6 MB checkpoints.

---

## Key findings worth remembering (rigor sprint 12-May-26)

### Baselines (2016–2025 partial / 2018–2024 original)

| Model | Splits | Test Macro F1 | Notes |
|-------|--------|---------------|-------|
| LSTM seed42 | 2018–2024 (old) | 0.824 | Single seed, MPS, original baseline |
| LSTM multi-seed (5) | 2016–2025 (new) | 0.8365 ± 0.009 | seed42 best run = 0.846 |
| LSTM Focal γ=2 (2 seeds) | 2016–2025 (new) | 0.849 | Beats WeightedCE by ~1.4% — **adopt for rerun** |
| LSTM Focal γ=1 (2 seeds) | 2016–2025 (new) | 0.848 | Tied with γ=2 |
| LSTM Focal γ=3 (2 seeds) | 2016–2025 (new) | 0.828 | Degrades — over-focusing |
| XGB seed42 (Optuna) | 2016–2025 (new) | 0.632 | depth=6, lr=0.196, n_est=527 |
| XGB multi-seed (5) | 2016–2025 (new) | 0.6365 ± 0.004 | Tighter std than LSTM |
| XGB bootstrap CI | 2016–2025 (new) | 0.598 [0.541, 0.649] | Single-seed CI, wide |

### LSTM best HP (8/50 Optuna trials)
- `hidden_size=32, num_layers=1, lr=0.0006, dropout=0.2`
- **Small arch wins** — confirms small-data regime hypothesis (~117 eff windows)
- num_layers=3 configs hung on MPS — search space biased toward shallow

### XGB best HP (50/50 Optuna trials)
- `n_estimators=527, max_depth=6, lr=0.196, min_child_weight=6, subsample=0.84, colsample=0.73`
- Expanded search (v2) re-confirmed max_depth=6 optimal

### Threshold tuning (LSTM seed 2024)
- Val-tuned per-class thresholds: `none=0.351, bull=0.621, bear=0.580`
- Δ macro F1 = +0.009, Δ bull F1 = +0.018, Δ bear F1 = −0.002
- Mostly helps bull (minority-ish); negligible elsewhere

### SHAP (XGB seed42)
- Top features: `ret_5`, `pos_in_range`, `prior_trend_5`, `mid_range_norm`, `macd_signal`
- All 35 features contribute above drop threshold
- **Caveat:** ran on `xgb_seed42` — confirm whether pre- or post-Optuna in rerun

### Reg ablation (single seed)
- Control: 0.8331 | no dropout: 0.8263 (-0.007) | no L2: 0.8288 (-0.004) | both off: 0.8210 (-0.012)
- Dropout > L2 contribution. **Single-seed only — rerun with multi-seed.**

### Asymmetry
- Bull F1: 0.789 ± 0.011, Bear F1: 0.809 ± 0.008 (gap = −0.020)
- No systematic asymmetry. FP bull gaps slightly larger (0.484) than TP bull gaps (0.461)

### Bootstrap CI (LSTM, old splits)
- Macro F1: 0.846 [0.831, 0.861], effective n=57
- **Single-seed CI** — rerun should be CI of mean across seeds

### Data scaling Δ (never measured)
- Old 2018–2024 vs new 2016–2025 head-to-head Δ never computed
- Plan said "adopt if Δ>0.01 F1" — adopted blind. **Measure in rerun.**

---

## Known bugs to fix in rerun (GPU fix sprint)

1. LSTM `num_layers=3` hangs on MPS — research agent investigating root cause
2. Focal seeds 0/123/2024 hung on MPS — same root cause
3. Window-sweep hangs on larger windows — same
4. XGB `n_jobs=-1` segfault on Python 3.14 — workaround = `n_jobs=1`
5. XGB import after torch in subprocess segfaults — workaround = torch-free worker
6. 3 XGB rigor tests skipped (`tests/rigor/`) — restore after fix

---

## Methodological gaps to close in rerun

- Bootstrap CI should be CI of mean across seeds (not within one seed)
- Reg ablation needs multi-seed (single seed std ~0.009; reported deltas may be noise)
- SHAP must run on Optuna-tuned XGB (not pre-tuning seed42)
- Threshold tuning must be val-only, applied to test exactly once (verify pipeline)
- Add naive baseline numbers on new test (majority class, uniform random) for honest "beats baseline" claim
- Measure data scaling Δ (old splits vs new, same config, same seed)
- ValidFVGLabeller training never attempted — separate scientific contribution, defer or include

---

## Pointers (kept on disk, not deleted)

- Build logs: `.nb-suite/build/12-May-26/{phase0,phase1,sprint-complete,data-extension-2015-2025,scripts-reorg}.md`
- Plans: `.nb-suite/plan/12-May-26/model-rigor-sprint.md`
- Research: `.nb-suite/research/12-May-26/data-scaling.md`
- Changelogs: `.nb-suite/changelogs/12-May-26/final-rigor-sprint-and-data-extension.md`
- Git: commits `cfb56c8`, `8f02b3d`, `0f67a3c`, `fb9d18b`, `6a37bc0`
- `docs/models-status.md` — current advertised numbers (stale; will overwrite in rerun)

---

## Restore path

If rerun goes wrong, restore from git: `git checkout 6a37bc0 -- checkpoints/ reports/`. All files tracked.
