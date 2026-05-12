# Build Log: Scripts Reorganisation — 12 May 2026

## Objective
Reorganise `/scripts/` per SOLID + DRY principles: group CLI tools by intent (data, training, rigor) and clean up subprocess workers.

## Before
21 files, flat:
- 3 subprocess workers (_shap_worker, _xgb_sweep_worker, _xgb_tune_worker)
- 4 data utilities (annotate_gold_set, count_valid_fvg, persist_labels, rebuild_dataset_2016_2025)
- 2 training scripts (train_lstm, train_xgboost)
- 8 rigor tools (multiseed_run, tune_lstm, tune_xgboost, threshold_sweep, window_sweep, asymmetry_analysis, bootstrap_ci, shap_xgb)
- 1 redundant script (multiseed_xgb — duplicate of multiseed_run --model xgboost)
- 2 main CLIs (inspect_models, paper_trade)

## After
19 files, organised in 5 directories:
- scripts/data/ (4 files)
- scripts/training/ (2 files)
- scripts/rigor/ (8 files)
- scripts/rigor/_workers/ (3 subprocess helpers)
- scripts/ top-level (inspect_models, paper_trade)

## Changes Made

### 1. Files Moved (via plain mv)

**Data utilities → scripts/data/**
- annotate_gold_set.py
- count_valid_fvg.py
- persist_labels.py
- rebuild_dataset_2016_2025.py

**Training → scripts/training/**
- train_lstm.py
- train_xgboost.py

**Rigor tools → scripts/rigor/**
- asymmetry_analysis.py
- bootstrap_ci.py
- multiseed_run.py
- shap_xgb.py
- threshold_sweep.py
- tune_lstm.py
- tune_xgboost.py
- window_sweep.py

**Subprocess workers → scripts/rigor/_workers/**
- _shap_worker.py
- _xgb_sweep_worker.py
- _xgb_tune_worker.py

**Top-level CLIs remain in scripts/**
- inspect_models.py
- paper_trade.py

### 2. File Deleted
- scripts/multiseed_xgb.py (REDUNDANT)
  - multiseed_run.py already supports --model xgboost via subprocess delegation

### 3. Import Paths Fixed
All relocated scripts updated `ROOT = Path(__file__).resolve().parent.parent` to `.parent.parent.parent`:
- 4 data scripts
- 2 training scripts
- 8 rigor scripts

### 4. Subprocess Worker References Updated
Scripts that spawn workers now use new paths:
- multiseed_run.py: _xgb_sweep_worker path
- shap_xgb.py: _shap_worker path
- tune_xgboost.py: _xgb_tune_worker path

### 5. Docstring Usage Examples Updated
All moved scripts' docstrings reference correct paths:
- scripts/data/*
- scripts/training/*
- scripts/rigor/*

### 6. Documentation Updated

**docs/architecture.md**
- Folder tree now shows scripts/data/, scripts/training/, scripts/rigor/_workers/
- Scripts section expanded into 3 subsections:
  - Data utilities (4 scripts)
  - Training (2 scripts)
  - Rigor sprint tools (8 scripts + _workers subdir)
  - Main CLIs (2 scripts)

**README.md**
- Quick start examples updated to scripts/training/
- Project layout description updated

**CLAUDE.md**
- Commands section updated to scripts/training/

### 7. Validation: Smoke Tests

All 19 scripts pass --help without errors:
✅ All data scripts (annotate_gold_set, count_valid_fvg, persist_labels, rebuild_dataset)
✅ All training scripts (train_lstm, train_xgboost)
✅ All rigor scripts (multiseed_run, tune_lstm, tune_xgboost, threshold_sweep, window_sweep, asymmetry_analysis, bootstrap_ci, shap_xgb)
✅ Top-level CLIs (inspect_models, paper_trade)

### 8. Test Suite Status

Pre-move baseline: 210 passed, 3 skipped, 1 pre-existing failure
Post-move result: **210 passed, 3 skipped, 1 pre-existing failure** ✅

- No breakage in test suite
- Known xgboost+torch segfault on Python 3.14 (pre-existing)
- Known future bar leak in test_split.py (pre-existing)

## Summary

✅ 16 files moved to subfolders
✅ 1 file deleted (redundant)
✅ All import paths corrected
✅ All subprocess references updated
✅ All docstring examples updated
✅ 3 doc files updated with new paths
✅ All 19 scripts pass smoke test
✅ Test suite passes with same baseline

## Structure Rationale

- **Data utilities** grouped (scripts/data/) — one-shot or interactive, used during pipeline setup
- **Training** grouped (scripts/training/) — baseline trainers, natural pair
- **Rigor tools** grouped (scripts/rigor/) — post-training analysis/ablation/tuning
- **Workers nested** (scripts/rigor/_workers/) — clearly subprocess helpers, separate from user-facing tools
- **Top-level CLIs** untouched (inspect_models, paper_trade) — main daily-use entry points
- **multiseed_xgb deleted** — completely redundant; multiseed_run --model xgboost does the same

This structure follows SOLID (single responsibility grouped) and DRY (no duplication), making it easy to understand what each script does and where to find it.
