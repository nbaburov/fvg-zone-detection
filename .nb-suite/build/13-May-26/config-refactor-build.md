# Build Log — Config-Architecture Refactor

**Date:** 2026-05-13
**Phase:** 1 of 3
**Status:** PHASE 1 COMPLETE — awaiting user approval

---

## Phase 1 — Config module + training scripts + experiments + tests

### Done

**New files created:**

- `src/config/__init__.py` — public API exports
- `src/config/schema.py` — Pydantic v2 models: ExperimentConfig, DataConfig, LSTMModelConfig, XGBModelConfig, TrainConfig, EvalConfig, RuntimeConfig with discriminated union on `arch`
- `src/config/loader.py` — `load_experiment(path, overrides)` + `experiment_from_json(path)` for legacy JSON compat
- `src/config/registry.py` — MODELS / LOSSES dicts + `@register_model` / `@register_loss` decorators; force-imports `_model_registrations` and `_loss_registrations`
- `src/config/_model_registrations.py` — registers FVGLSTMClassifier as "lstm", XGBoostFVGClassifier as "xgb"
- `src/config/_loss_registrations.py` — registers WeightedCE as "weighted_ce", FocalLoss as "focal"
- `scripts/training/train.py` — unified dispatcher; reads YAML, dispatches by arch; contains `_train_lstm()`, `_lstm_train_loop()`, `_train_xgb()`; `main(default_config=)` for wrapper reuse
- `experiments/_base.yaml` — canonical defaults
- `experiments/lstm_g1.yaml` — G1 HP from best_hp_lstm.json (lr=5.30e-4, Adam, patience=15)
- `experiments/xgb_g1.yaml` — G1 HP from best_hp_xgb.json (n_estimators=513)
- `experiments/lstm_g1_rawfvg.yaml` — same HP, labeller=fvg
- `experiments/xgb_g1_rawfvg.yaml` — same HP, labeller=fvg
- `experiments/_template.yaml` — every field documented with inline comments
- `tests/config/test_schema.py` — 13 tests
- `tests/config/test_loader.py` — 13 tests
- `tests/config/__init__.py`

**Modified files:**

- `scripts/training/train_lstm.py` — added `--config` and `--set` args; when `--config` provided, delegates to `scripts.training.train._train_lstm(cfg, seed)`. Legacy path (no `--config`) unchanged.
- `scripts/training/train_xgboost.py` — added `--config` and `--set` args; when `--config` provided, reads `XGBModelConfig` HP from YAML and passes to `XGBoostFVGClassifier(params=_xgb_hp)`. Legacy path (no `--config`) unchanged.

### Test result

```
277 passed, 1 warning in 14.41s
(251 original + 26 new config tests)
```

### Gate checks

- `pytest` → 277/277 pass
- `python -c "from src.config import load_experiment; cfg = load_experiment('experiments/lstm_g1.yaml'); print(cfg.model.hidden_size)"` → `128`
- `python scripts/training/train.py --config experiments/lstm_g1.yaml --set "train.seeds=[42]" --debug` → completes, writes checkpoint + meta.json
- `python scripts/training/train_lstm.py --seed 42 --debug` (no `--config`) → legacy path runs training, fails only in `write_eval_log` due to pre-existing 2-class debug issue (not a regression)

### Deviations from plan

1. **`_model_registrations.py` uses function-call pattern, not decorator on class** — plan said `@register_model("lstm")` decorator on class definition in models/. That approach would create a circular import: `registry.py → _model_registrations.py → lstm.py` and if `lstm.py` ever imports config, it would cycle. The leaf-file pattern (registrations are plain function calls in the separate module) avoids this entirely and is equivalent. The `MODELS` / `LOSSES` dicts are still populated correctly at import time.

2. **`train.py` uses inline `_lstm_train_loop()` rather than calling `train_lstm.train()`** — plan suggested reusing the existing `train()` function. However that function reads `WINDOW_SIZE`, `LR`, `WEIGHT_DECAY`, `BATCH_TRAIN` from module-level constants which are stale pre-tuning defaults. Patching module globals is fragile. A clean loop function that takes explicit parameters is safer and more maintainable.

3. **`drop_cross_session_windows=False` explicit** — train.py passes this explicitly from `cfg.data.drop_cross_session` (default false). Original train_lstm.py hardcoded `False`. Config-driven path matches.

4. **Checkpoint naming preserved** — `lstm_seed{N}.pt` / `lstm_seed{N}_rawfvg.pt` unchanged. Risk item from plan was correctly handled.

5. **Pre-existing `write_eval_log` debug bug** — when `--debug` produces only 2 classes, `classification_report` crashes with target_names mismatch. This predates Phase 1 and is not a regression. Only affects `--debug` in the legacy path.

### Open flags

- Phase 2 still needed: rigor scripts (multiseed_run.py etc.) not yet config-driven
- `experiments/` directory not yet in `.gitignore` exclusions — should be tracked (no action needed, it's new)
- `train_xgboost.py` config path reads model HP but XGBoost feature extraction still hardcodes `window_size=60, stride=1` — acceptable for Phase 1 since config HP are passed to the model constructor, and `window_size` in XGB is only used for feature extraction which is a separate concern (Phase 2 will wire `cfg.data.window_size` to `extract_window_features`)

---

## Phase 2 — Rigor scripts (pending)

Not started. Awaiting Phase 1 approval.

## Phase 3 — Inspect + paper_trade verification (pending)

Not started.
