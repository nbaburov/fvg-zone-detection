# Build Log: Post-Sprint Polish

**Date:** 2026-05-13
**Branch:** master
**Scope:** CNN-LSTM no-dropout squeeze (Task 1) + context sync (Task 2) + cleanup (Task 3)

---

## Task 1 — CNN-LSTM No-Dropout Squeeze

**Hypothesis:** G9 reg ablation showed no-dropout mean F1 = +0.009 lift (0.623 vs 0.614 canonical). May suggest dropout=0.2217 is over-regularizing for a 29k-param model.

**Config created:** `experiments/cnn_lstm_g1_nodrop.yaml` — identical to `cnn_lstm_g1.yaml` except `dropout: 0.0`, `head_dropout: 0.0`.

**Run:** 5-seed multiseed_run, output `reports/rigor/2026-05-13/cnn_lstm_G1_nodrop_squeeze/`

**Results:**

| seed | macro_f1 |
|------|----------|
| 0 | 0.5829 |
| 17 | 0.6198 |
| 42 | 0.6177 |
| 123 | 0.6466 |
| 2024 | 0.6033 |
| **mean±std** | **0.6141 ± 0.0234** |

**Canonical baseline:** 0.614 ± 0.021

**Decision:** No promotion. Delta = +0.0001 — below 0.005 marginal threshold. Dropout does not appear to be over-regularizing at this scale. G9 lift (+0.009) was not replicated under identical full-run conditions — seed distribution variance absorbed it. Canonical `experiments/cnn_lstm_g1.yaml` and checkpoints unchanged.

**Files created:** `experiments/cnn_lstm_g1_nodrop.yaml`

---

## Task 2 — Context Sync (nb-update)

**Depth:** Deep — post-CNN-LSTM sprint, config refactor, multiple new files.

**Files updated:**

| File | Changes |
|------|---------|
| `CLAUDE.md` | Added CNN-LSTM to baselines line; added `src/config/` to architecture block; added `cnn_lstm.py` to models; expanded scripts section (train_cnn_lstm.py, train.py, new rigor scripts); updated CNN-LSTM model progression line to show complete result; fixed MPS note — all training CPU-only including CNN-LSTM; fixed XGB worker path |
| `README.md` | Added CNN-LSTM row to status table; updated "planned" line; added cnn_lstm to src/models layout |
| `docs/architecture.md` | Added `src/config/` subtree; added `cnn_lstm.py` + `cnn_lstm_adapter.py`; added `train_cnn_lstm.py`, `train.py`; added 5 new rigor scripts to folder tree and scripts table; updated mermaid component map (models + adapters); updated torch tools row |
| `docs/models-status.md` | Fixed CNN-LSTM device from "MPS" to "CPU" |
| `memory/project_smc_challenge.md` | Updated model F1 table; updated model progression; updated recent changes to 13-May-26 |

**Already current:** `.env.example`, `requirements.txt` (no stale deps), notebooks (01 synced in prior session, 02 unchanged)

---

## Task 3 — Cleanup

**Deleted:**

| Category | Files | Notes |
|----------|-------|-------|
| Backup checkpoints | 2 | `cnn_lstm_seed42_base_backup.{pt,meta.json}` — canonical committed, no further value |
| Stale Optuna DBs | 2 | `checkpoints/{xgboost,lstm}/optuna_2026-05-12.db` — superseded by 2026-05-13 DBs |
| Raw run logs in `logs/` | 25 | All `.log` files — superseded by JSON results in `reports/rigor/` |
| Raw run logs in `reports/rigor/` | 10 | `g2_run.log`, `g4_run.log`, `seed42_retrain.log`, `g9_*.log`, etc. |
| Empty report dirs | 11 | Zero-file dirs from interrupted/smoke runs |
| **Total** | **50 files + 11 dirs** | |

**Confirmed gitignored (no action needed):** `reports/`, `logs/`, `__pycache__/`, `.DS_Store`, `.ipynb_checkpoints/` — all in `.gitignore`.

**Experiments YAMLs:** All 9 YAMLs purposeful — no duplicates. `cnn_lstm_g1_nodrop.yaml` created this session (Task 1 squeeze experiment, documented).

---

## Open flags

- No-dropout YAML kept as documented experiment in `experiments/cnn_lstm_g1_nodrop.yaml`
- xLSTM and Transformer still planned — no timeline set
