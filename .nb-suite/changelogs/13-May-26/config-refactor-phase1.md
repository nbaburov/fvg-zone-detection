# Changelog — Config Refactor Phase 1 — 13-May-26

| SHA | Type | Subject |
|-----|------|---------|
| 2947b84 | feat(config) | add Pydantic v2 experiment config + YAML-driven training |

## Files (new)
- `src/config/` — schema, loader, registry, model + loss registrations
- `experiments/` — `_base.yaml`, `_template.yaml`, `lstm_g1.yaml`, `xgb_g1.yaml`, raw-FVG variants
- `scripts/training/train.py` — unified entry
- `tests/config/` — 26 new tests

## Files (modified)
- `scripts/training/{train_lstm,train_xgboost}.py` — `--config` / `--set` flags, delegate to `train.py`

## Summary
Phase 1 of config refactor: Pydantic v2 experiment schema, YAML loader with dot-path overrides, registry pattern for models/losses mirroring LABELLERS, unified train entry. Fixed LSTM optimiser mismatch (Adam canonical, was AdamW+OneCycleLR). 277/277 tests pass.
