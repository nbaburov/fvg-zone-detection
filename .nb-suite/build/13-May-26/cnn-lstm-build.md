# Build Log: CNN-LSTM Sprint

**Date:** 2026-05-13
**Branch:** master
**Plan:** `.nb-suite/plan/13-May-26/cnn-lstm-sprint.md`
**Research:** `.nb-suite/research/13-May-26/cnn-lstm-architecture.md`

---

## Phase 1 — Model + config plumbing

### Done

- `src/models/cnn_lstm.py` — `FVGCNNLSTMClassifier`: Conv1d(5→F, k) → BN → ReLU [× n_conv_layers] → LSTM(F→H) → Dropout → Linear(H→3)
- `src/config/schema.py` — Added `CNNLSTMModelConfig`; updated `ModelConfig` union with discriminator
- `src/config/_model_registrations.py` — Registered `"cnn_lstm"` → `FVGCNNLSTMClassifier`
- `src/config/__init__.py` — Exported `CNNLSTMModelConfig`
- `scripts/training/train.py` — Added `_train_cnn_lstm()` dispatch + `"cnn_lstm"` branch in `main()`
- `scripts/training/train_cnn_lstm.py` — Thin wrapper, mirrors `train_lstm.py` pattern
- `scripts/rigor/multiseed_run.py` — Added `"cnn_lstm"` to `--model choices`
- `src/rigor/seed_sweep.py` — Added `_train_cnn_lstm()`, updated `Literal` + dispatch + ckpt path logic
- `experiments/cnn_lstm_base.yaml` — Baseline config for smoke + G1 starting point
- `experiments/cnn_lstm_g1.yaml` — Placeholder; populated after G1 Optuna completes
- `tests/models/test_cnn_lstm.py` — 4 tests: forward shape, gradient flow, param count, single-sample

### Good

- 29,027 params at default HP — matches research doc estimate of ~28,771
- No NaN in 5-epoch debug smoke
- Class weights loaded correctly: `[0.0242, 1.2320, 1.7438]` (fvg_valid)
- BN stable at B=16 debug subset

### Bad / Fixed

- `class_weights_fvg_valid.json` referenced in plan does not exist — only `class_weights.json` present. This is the correct file (it contains fvg_valid weights). LSTM training uses the same filename. No action needed.
- Pool padding: plan references `MaxPool1d(stride=1, padding=1)` but research doc notes this produces output length L+1 for even kernel. Used `padding=0` instead — sequence length reduced by 1 per pool layer (59 from 60 with 1 pool). This matches the research doc's safer recommendation.

### Observations

- `--set "train.max_epochs=2"` override with `--debug` runs 5 epochs (debug mode caps at 5, ignores max_epochs override). Smoke still valid — no error is the gate.
- `seed_sweep.py` has its own `_train_lstm` that bypasses `train.py` entirely. CNN-LSTM uses same pattern — self-contained `_train_cnn_lstm` in `seed_sweep.py`.

### Tests

- `tests/models/test_cnn_lstm.py`: **4/4 pass**
- Full suite: **284/284 pass**

---

## Phase 2 — G1 Optuna (sealed)

G1 Optuna sealed at 6/12 trials (60min cap). Best val F1 = 0.6440, trial 0. Config written to `experiments/cnn_lstm_g1.yaml`.

---

## Phase 3 — G2–G10 Rigor Sprint

### Step 1 — cnn_lstm_g1.yaml populated
Overwrote placeholder with G1 best HP. Verified load: kernel_size=5, lr=0.000731.

### Step 2 — G2 Multi-seed
Issue: seed42 had stale base-config checkpoint (skip logic fired). Retrained standalone after backing up old checkpoint. Corrected G2: **0.614 ± 0.021** — Gate PASS.

### Step 3 — Phase 3 gates
- **G4** threshold: +0.006 Δ (negligible). `threshold_multiseed.py` extended for CNN-LSTM (meta-driven HP).
- **G6** asymmetry: bull 0.451±0.013 vs bear 0.417±0.059, gap +0.034, not systematic.
- **G7** window sweep: best val W=90, W=60 retained. `window_sweep.py` extended for CNN-LSTM.
- **G9** reg ablation: no-dropout **improves** (+0.009, lower variance); no-L2 hurts (−0.034). Full reg retained.
- **G10** bootstrap CI: 95% [0.576, 0.649]. `bootstrap_ci_multiseed.py` extended.

### Step 4 — Phase 4 docs + inspect
- `src/inspect/adapters/cnn_lstm_adapter.py` — written, auto-discovered
- Inspect: `reports/inspect/2026-05-13_004532/` (F1_macro=0.439 on seed42)
- `docs/models-status.md` — CNN-LSTM canonical entry + full sprint section

### Tests
Full suite: **284/284 pass** throughout.

---

---

## Post-sprint Polish — Task 1: No-dropout Squeeze (2026-05-13)

**Experiment:** `experiments/cnn_lstm_g1_nodrop.yaml` — G1 HPs with `dropout=0.0`, `head_dropout=0.0`.

**Command:**
```
.venv/bin/python scripts/rigor/multiseed_run.py \
  --model cnn_lstm \
  --config experiments/cnn_lstm_g1_nodrop.yaml \
  --seeds 0 17 42 123 2024 \
  --output-dir reports/rigor/2026-05-13/cnn_lstm_G1_nodrop_squeeze
```

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

**Decision:** No promotion. Delta = +0.0001 — negligible, below 0.005 marginal threshold. Dropout does not appear to be over-regularizing at this model scale. Canonical `experiments/cnn_lstm_g1.yaml` and checkpoints unchanged.

**Files touched:** `experiments/cnn_lstm_g1_nodrop.yaml` (created), `reports/rigor/2026-05-13/cnn_lstm_G1_nodrop_squeeze/` (results).

---

## Open flags

- Pool output length: with `use_pool=True, padding=0`, L=60 → 59 per pool layer. Two conv layers with pool → 58. LSTM handles variable T — not a correctness issue.
- seed42 backup checkpoints in `checkpoints/cnn_lstm/cnn_lstm_seed42_base_backup.*` — keep until next full sprint.
- G9 no-dropout finding: if pursuing further, consider retune without dropout constraint.
- W=90 sensitivity: CNN-LSTM may benefit from W=90 sweep, but not explored this sprint.
