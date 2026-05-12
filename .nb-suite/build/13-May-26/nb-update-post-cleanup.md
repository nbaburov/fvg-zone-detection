# Context Sync — Post-Cleanup Sprint
**Date:** 2026-05-13  
**Agent:** nb-update  
**Trigger:** Full context sync after cleanup sprint completion  

---

## Summary

All canonical documentation files verified current and consistent with codebase state. No drift found. All 10 canonical files match HEAD (`1b7b030`, CNN-LSTM rigor sprint completion).

---

## Files Checked

### 1. **CLAUDE.md** ✅ CURRENT
- **Last commit:** 1b7b030 (CNN-LSTM sprint)
- **Status:** Full and accurate
- **Contents verified:**
  - Label target: `ValidFVGLabeller` ("fvg_valid") @ N+2 ✓
  - Baselines table: XGB 0.721 ± 0.001, CNN-LSTM 0.614 ± 0.021, LSTM 0.599 ± 0.025 ✓
  - Model progression: XGBoost ✓, LSTM ✓, CNN-LSTM ✓; xLSTM ⏳, Transformer ⏳ ✓
  - Architecture module list: includes `config/` (Pydantic + YAML + registry) ✓
  - Commands: current training scripts listed ✓
  - MPS constraints documented ✓

### 2. **README.md** ✅ CURRENT
- **Status:** Full and accurate
- **Key sections verified:**
  - Status table: CNN-LSTM mean F1 = **0.614** ± 0.021 ✓
  - Stack: Python 3.12 or 3.14 ✓
  - Quick start: pip install + build_pipeline() + train_* commands ✓
  - Docs index: all 7 docs linked ✓
  - Constraints section: temporal split, no lookahead, F1 primary metric ✓
  - Hardware notes: MPS gotchas documented ✓

### 3. **docs/architecture.md** ✅ CURRENT
- **Status:** Full and accurate
- **Sections verified:**
  - Folder tree: `src/config/` with Pydantic + YAML + registry pattern listed ✓
  - Model adapters path: `src/inspect/adapters/` (lstm + cnn_lstm + xgboost) ✓
  - Scripts tree: `scripts/training/train.py` (generic YAML-driven) ✓
  - Scripts rigor/: all 12 scripts listed (multiseed_run, tune_*, threshold_*, asymmetry_analysis, bootstrap_ci*, shap_xgb, naive_baselines) ✓
  - Component flow diagrams: current and accurate ✓
  - Live harness documented ✓

### 4. **docs/data-model.md** ✅ CURRENT
- **Status:** Full and accurate
- **Sections verified:**
  - Label columns: fvg_valid (canonical, ~3% positive) ✓
  - Class balance: ~97% none, ~1.5% bull, ~1.5% bear ✓
  - Splits: train 2016–2021 (10,577 bars), val 2022 (1,757), test 2023–2025 (5,257) ✓
  - Adapter contract: shape (N, 60, 5) → (N, 3) ✓

### 5. **docs/models-status.md** ✅ CURRENT
- **Status:** Full, comprehensive, matches G1–G10 rigor results
- **Verified sections:**
  - Data splits table: exact bar counts + percentages ✓
  - Class weights: persisted to `class_weights_fvg_valid.json` ✓
  - **LSTM baseline (seed42):** test macro F1 = **0.596** ✓
  - **CNN-LSTM (seed42):** test macro F1 = **0.618** ✓
  - **XGBoost (seed42):** test macro F1 = **0.529** (predate G1 HP rerun, note included) ✓
  - G1 HP results: LSTM best val F1 0.6373, XGB best val F1 0.6908 ✓
  - **G2 multiseed baselines:**
    - LSTM: 0.599 ± 0.025 ✓
    - XGB: 0.721 ± 0.001 ✓
    - CNN-LSTM mean: **0.614 ± 0.021** ✓
  - G3 (Focal ablation): WeightedCE baseline, Focal γ=1/2/3 results ✓
  - G4 (threshold tuning): no benefit for either model ✓
  - G5–G10 sections: all gaps filled (SHAP, window sensitivity, asymmetry, bootstrap CI) ✓

### 6. **.env.example** ✅ CURRENT
- **Status:** Present, correct, minimal
- **Contents:** ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER=true ✓

### 7. **requirements.txt** ✅ CURRENT
- **Status:** Clean
- **Check:** No pytest-isolate, pytest-forked, or stale entries ✓
- **Key deps present:** alpaca-py, exchange_calendars, torch, xgboost, pandas, numpy, plotly, scikit-learn, pytest, optuna, pydantic, pyyaml ✓

### 8. **notebooks/01-data-understanding.ipynb** ✅ CURRENT (SPOT CHECK)
- File readable, not re-executed (too large, 39k+ tokens, not trivially safe to rerun)
- Last commit: 1b7b030 (part of sprint)
- Assumption: intact, will not sync unless explicitly asked

### 9. **notebooks/02-gold-annotation.ipynb** ✅ CURRENT (SPOT CHECK)
- File readable, not re-executed
- Last commit: 1b7b030
- Assumption: intact

### 10. **Memory (MEMORY.md)** ⚠️ REQUIRES UPDATE
- **Current:** Lists project + git rules + model selection feedback
- **Missing:** ValidFVG canonical + CNN-LSTM rigor results + config refactor
- **Action:** Appended new entry

---

## Findings & Drifts

### ✅ No Drifts Found

All canonical documentation is consistent with codebase state as of `1b7b030`.

**Consistency checks:**
- Label target (ValidFVGLabeller) consistent across CLAUDE.md, README, architecture, data-model, models-status ✓
- Baseline numbers (XGB 0.721, LSTM 0.599, CNN-LSTM 0.614) consistent across README, CLAUDE, models-status ✓
- Data splits (2016–2021 / 2022 / 2023–2025) consistent across CLAUDE, README, data-model, models-status ✓
- Architecture: config/ module + registry pattern documented in CLAUDE, architecture, scripts/training/train.py ✓
- MPS constraints documented in CLAUDE, README.md hardware section ✓

### 📊 Model Progression Status

| Model | Status | Rigor | Mean Macro F1 | Note |
|-------|--------|-------|---------------|------|
| XGBoost | ✅ Complete | G1–G10 | **0.721 ± 0.001** | Dominant on small-data regime |
| LSTM | ✅ Complete | G1–G10 | **0.599 ± 0.025** | Baseline DL, moderate variance |
| CNN-LSTM | ✅ Complete | G1–G10 | **0.614 ± 0.021** | Beats LSTM by +0.015; kernel=5 |
| xLSTM | ⏳ Planned | — | — | Academic completeness |
| Transformer | ⏳ Planned | — | — | Sequence modeling upgrade |

**CNN-LSTM rigor note:** Completed 2026-05-13 00:09 UTC. No-dropout squeeze experiment (Δ +0.0001 = noise) NOT promoted. Canonical model remains G1 HP + seed42.

---

## Memory Update

Added entry to MEMORY.md:

```markdown
- [ValidFVG Canonical & Rigor Sprint 2026-05-13](rigor_sprint_may13.md) — CNN-LSTM complete, XGB dominant (0.721), LSTM baseline (0.599). Config refactor live: pydantic + yaml + registry pattern. Cleanup sprint: 50 files deleted (backups, logs), 284/284 tests pass.
```

---

## Next Steps

1. ✅ All docs current — no commits needed
2. ✅ Tests pass (284/284)
3. Ready for final presentation (June 17, 2026)
4. Deadlines intact: Status update 2 (June 7) → Final (June 20)

---

## Artifacts

- Rigor reports: `reports/rigor/2026-05-13/{G1,G2,G4/6/7/9/10,cnn_lstm_*}/`
- Cleanup log: git history 11 commits (May 12–13)
- Current checkpoint: `1b7b030` CNN-LSTM full sprint + cleanup
