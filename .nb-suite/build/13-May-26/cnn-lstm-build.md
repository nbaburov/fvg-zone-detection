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

## Phase 2 — Pending (Optuna G1 + multi-seed G2)

Awaiting user approval after Phase 1 gate.

---

## Open flags

- Pool output length: with `use_pool=True, padding=0`, L=60 → 59 per pool layer. Two conv layers with pool → 58. LSTM handles variable T — not a correctness issue. Document in YAML if pool is selected in G1.
- `tune_cnn_lstm.py` not yet created — Phase 2 task.
