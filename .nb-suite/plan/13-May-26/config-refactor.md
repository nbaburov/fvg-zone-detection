# Config-Driven Architecture Refactor — Plan

**Date:** 2026-05-13
**Feeds:** @nb-build
**Research source:** `.nb-suite/research/13-May-26/config-architecture.md`
**Outcome:** Every run reads `experiments/<name>.yaml`. New model = 1 model file + 1 YAML, zero script forks.

---

## Decision log (resolved before build starts)

### D1: Canonical LSTM optimiser

**Conflict confirmed:** `train_lstm.py` uses **AdamW + OneCycleLR**. `seed_sweep._train_lstm` and `optuna_utils.LSTMObjective` both use plain **Adam**. G1 HP search ran via `tune_lstm.py → LSTMObjective` → plain Adam. The `best_hp_lstm.json` values (lr=5.30e-4, weight_decay=3.92e-5) were tuned under **Adam**, not AdamW.

**Decision: Adam is canonical for LSTM.** Rationale: the HP were tuned under Adam; using AdamW+OneCycleLR with those HP is a mismatched pairing. `train_lstm.py` constants (LR=3e-3, WEIGHT_DECAY=1e-2) are the old pre-tuning defaults — they are overridden when `--config` is passed. The standalone `train_lstm.py` run currently uses stale defaults, not G1 HP. Config refactor fixes this by making G1 HP the source of truth.

**Action:** `TrainConfig.optimizer = "adam"`, `TrainConfig.scheduler = "none"` as defaults. Expose `"adamw"` and `"onecycle"` as valid options. `train.py` dispatches accordingly. `train_lstm.py` (if kept) updated to use `cfg.train.optimizer`.

### D2: Single unified `train.py` vs keep separate scripts

**Decision: Single `scripts/training/train.py` + keep `train_lstm.py` and `train_xgboost.py` as thin wrappers.** Rationale: existing test suite and CLAUDE.md commands reference the individual scripts. Rewriting them as thin wrappers that call `train.py` with a baked-in model type preserves backwards compat without duplication.

Wrapper pattern:
```python
# scripts/training/train_lstm.py (new version)
from scripts.training.train import main
main(default_config="experiments/lstm_g1.yaml")
```

### D3: `experiments/` directory location and git tracking

`experiments/` at repo root, checked in. Contents: YAML configs that are the canonical experiment identity. `reports/`, `checkpoints/` remain gitignored. When `tune_*.py` finds a new best, it writes to `reports/rigor/<ts>/best_hp_*.json` (existing behavior) **and** optionally updates `experiments/<name>.yaml` via `cfg.to_yaml()` — caller decides.

### D4: `SeedSweepConfig` retirement timeline

Phase 1: `SeedSweepConfig` coexists with `ExperimentConfig`. No migration.
Phase 2: `multiseed_run.py` switches to `ExperimentConfig`. `SeedSweepConfig` is kept in `src/rigor/seed_sweep.py` (it is used internally by `run_seed_sweep`) but its construction is driven from `ExperimentConfig` fields — a thin adapter factory is added. Full removal deferred to post-CNN-LSTM when the internal `_train_lstm` in `seed_sweep.py` is retired in favor of the unified trainer.

### D5: Backwards compat for `best_hp_*.json`

Old JSON format (flat dict with metadata keys `val_macro_f1`, `trial_number`, `study_name`, `storage`) is already handled by the strip pattern in `multiseed_run.py` (line: `hp_only = {k: v for k, v in ... if k not in (...)}`). The loader will use the same strip logic when loading a JSON-format config. **No migration script needed.** Old JSONs remain readable indefinitely.

---

## Canonical YAML schema

### `experiments/_base.yaml`
```yaml
# Defaults inherited by all experiments via loader merge.
# Explicit experiment YAMLs override only what they change.
name: base
data:
  data_dir: data/processed
  labeller: fvg_valid           # "fvg" | "fvg_valid"
  splits: default               # "default" | "legacy" | path/to/dir
  window_size: 60
  stride: 1
  drop_cross_session: false
model:
  arch: lstm                    # discriminator field — sets model class
train:
  seeds: [42]
  batch_size: 32
  batch_eval: 256
  lr: 1.0e-3
  weight_decay: 1.0e-4
  optimizer: adam               # "adam" | "adamw"
  scheduler: none               # "none" | "onecycle"
  max_epochs: 100
  patience: 15
  ema_alpha: 0.3
  max_grad_norm: 1.0
  loss: weighted_ce             # "weighted_ce" | "focal"
  focal_gamma: 2.0
  device: cpu
  ablation_no_dropout: false
  ablation_no_l2: false
eval:
  lookahead_bars: 20
  bootstrap_n_iter: 1000
  bootstrap_block_size: 60
runtime:
  output_dir: reports/rigor
  checkpoint_dir: checkpoints
  n_jobs: -1
```

### `experiments/lstm_g1.yaml`
```yaml
name: lstm_g1_validfvg
data:
  labeller: fvg_valid
  window_size: 60
model:
  arch: lstm
  hidden_size: 128
  num_layers: 1
  dropout: 0.31753
  head_dropout: 0.52621
train:
  seeds: [0, 17, 42, 123, 2024]
  batch_size: 16
  lr: 5.301506e-04
  weight_decay: 3.9156e-05
  optimizer: adam
  scheduler: none
  max_epochs: 100
  patience: 15
  loss: weighted_ce
  device: cpu
```

### `experiments/xgb_g1.yaml`
```yaml
name: xgb_g1_validfvg
data:
  labeller: fvg_valid
  window_size: 60
model:
  arch: xgb
  n_estimators: 513
  max_depth: 4
  learning_rate: 0.13114
  min_child_weight: 1
  subsample: 0.82644
  colsample_bytree: 0.72495
train:
  seeds: [0, 17, 42, 123, 2024]
  device: cpu
```

### `experiments/lstm_g1_rawfvg.yaml`
```yaml
name: lstm_g1_rawfvg
data:
  labeller: fvg
  window_size: 60
model:
  arch: lstm
  hidden_size: 128
  num_layers: 1
  dropout: 0.31753
  head_dropout: 0.52621
train:
  seeds: [0, 17, 42, 123, 2024]
  batch_size: 16
  lr: 5.301506e-04
  weight_decay: 3.9156e-05
  optimizer: adam
  scheduler: none
  device: cpu
```

### `experiments/xgb_g1_rawfvg.yaml`
```yaml
name: xgb_g1_rawfvg
data:
  labeller: fvg
  window_size: 60
model:
  arch: xgb
  n_estimators: 513
  max_depth: 4
  learning_rate: 0.13114
  min_child_weight: 1
  subsample: 0.82644
  colsample_bytree: 0.72495
train:
  seeds: [0, 17, 42, 123, 2024]
  device: cpu
```

### `experiments/_template.yaml`
Documents every field with inline comments — written last in Phase 1.

---

## Pydantic v2 schema (exact)

### `src/config/schema.py`

```python
from __future__ import annotations
from pathlib import Path
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    data_dir: Path = Path("data/processed")
    labeller: Literal["fvg", "fvg_valid"] = "fvg_valid"
    splits: str = "default"
    window_size: int = 60
    stride: int = 1
    drop_cross_session: bool = False

class LSTMModelConfig(BaseModel):
    arch: Literal["lstm"] = "lstm"
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.318
    head_dropout: float = 0.526

class XGBModelConfig(BaseModel):
    arch: Literal["xgb"] = "xgb"
    n_estimators: int = 513
    max_depth: int = 4
    learning_rate: float = 0.1311
    min_child_weight: int = 1
    subsample: float = 0.826
    colsample_bytree: float = 0.725

ModelConfig = Annotated[
    LSTMModelConfig | XGBModelConfig,
    Field(discriminator="arch")
]

class TrainConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [42])
    batch_size: int = 32
    batch_eval: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw"] = "adam"
    scheduler: Literal["none", "onecycle"] = "none"
    max_epochs: int = 100
    patience: int = 15
    ema_alpha: float = 0.3
    max_grad_norm: float = 1.0
    loss: Literal["weighted_ce", "focal"] = "weighted_ce"
    focal_gamma: float = 2.0
    device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"
    ablation_no_dropout: bool = False
    ablation_no_l2: bool = False

class EvalConfig(BaseModel):
    lookahead_bars: int = 20
    bootstrap_n_iter: int = 1000
    bootstrap_block_size: int = 60

class RuntimeConfig(BaseModel):
    output_dir: Path = Path("reports/rigor")
    checkpoint_dir: Path = Path("checkpoints")
    n_jobs: int = -1

class ExperimentConfig(BaseModel):
    name: str = "unnamed"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=LSTMModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
```

### `src/config/loader.py`

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from src.config.schema import ExperimentConfig

_BASE_YAML = Path(__file__).resolve().parents[2] / "experiments" / "_base.yaml"

def load_experiment(path: Path | str, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load YAML, merge with _base.yaml defaults, apply overrides dict.

    overrides: flat dotted keys, e.g. {"train.seed": 17, "model.hidden_size": 256}
    """
    base = _load_yaml(_BASE_YAML) if _BASE_YAML.exists() else {}
    experiment = _load_yaml(Path(path))
    merged = _deep_merge(base, experiment)
    if overrides:
        for key, val in overrides.items():
            _set_nested(merged, key.split("."), val)
    return ExperimentConfig.model_validate(merged)

def experiment_from_json(path: Path | str) -> ExperimentConfig:
    """Load legacy best_hp_*.json. Strips metadata keys, infers arch from filename."""
    import json, re
    data = json.loads(Path(path).read_text())
    strip = {"val_macro_f1", "best_value", "trial_number", "study_name", "storage",
             "n_complete", "n_pruned", "n_trials_completed"}
    hp = {k: v for k, v in data.items() if k not in strip}
    arch = "xgb" if "xgb" in Path(path).name else "lstm"
    return ExperimentConfig.model_validate({"model": {"arch": arch, **hp}})

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def _set_nested(d: dict, keys: list[str], val: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val
```

### `src/config/registry.py`

```python
from __future__ import annotations
from typing import Callable, TypeVar

T = TypeVar("T")

MODELS: dict[str, type] = {}
LOSSES: dict[str, type] = {}
OPTIMIZERS: dict[str, Callable] = {}

def register_model(name: str):
    def dec(cls):
        MODELS[name] = cls
        return cls
    return dec

def register_loss(name: str):
    def dec(cls):
        LOSSES[name] = cls
        return cls
    return dec

# Import concrete registrations so decorators fire
from src.config import _model_registrations   # noqa: E402, F401
from src.config import _loss_registrations    # noqa: E402, F401
```

`_model_registrations.py` and `_loss_registrations.py` are thin files that import model/loss classes and apply `@register_model`/`@register_loss`. This avoids circular imports (registry.py → models/ → training/ cycles).

### `src/config/__init__.py`

```python
from src.config.schema import (
    ExperimentConfig, DataConfig, LSTMModelConfig, XGBModelConfig,
    TrainConfig, EvalConfig, RuntimeConfig,
)
from src.config.loader import load_experiment, experiment_from_json
from src.config.registry import MODELS, LOSSES, register_model, register_loss

__all__ = [
    "ExperimentConfig", "DataConfig", "LSTMModelConfig", "XGBModelConfig",
    "TrainConfig", "EvalConfig", "RuntimeConfig",
    "load_experiment", "experiment_from_json",
    "MODELS", "LOSSES", "register_model", "register_loss",
]
```

---

## File-by-file change list

### Phase 1 — Config module + training scripts + base YAMLs + tests
**Estimate: 2 hours**

#### New files
| File | Action | Notes |
|------|--------|-------|
| `src/config/__init__.py` | CREATE | exports above |
| `src/config/schema.py` | CREATE | Pydantic v2 models |
| `src/config/loader.py` | CREATE | YAML load + override + JSON compat |
| `src/config/registry.py` | CREATE | MODELS, LOSSES dicts + decorators |
| `src/config/_model_registrations.py` | CREATE | imports LSTM, XGB classes + applies @register_model |
| `src/config/_loss_registrations.py` | CREATE | imports WeightedCE, FocalLoss + applies @register_loss |
| `scripts/training/train.py` | CREATE | unified dispatcher; reads `--config` + `--set key=val` overrides |
| `experiments/_base.yaml` | CREATE | canonical defaults |
| `experiments/lstm_g1.yaml` | CREATE | from best_hp_lstm.json |
| `experiments/xgb_g1.yaml` | CREATE | from best_hp_xgb.json |
| `experiments/lstm_g1_rawfvg.yaml` | CREATE | rawfvg comparison |
| `experiments/xgb_g1_rawfvg.yaml` | CREATE | rawfvg comparison |
| `experiments/_template.yaml` | CREATE | every field documented |
| `tests/config/test_schema.py` | CREATE | see test strategy |
| `tests/config/test_loader.py` | CREATE | see test strategy |

#### Modified files
| File | Change |
|------|--------|
| `scripts/training/train_lstm.py` | Thin wrapper: adds `--config` arg, calls `load_experiment()`, passes cfg into `make_loaders()` and `train()`. Module-level constants remain as fallback when `--config` not provided. Adds `--set` for overrides. Optimiser changed to Adam (matches G1 tuning) when config-driven. |
| `scripts/training/train_xgboost.py` | Adds `--config` arg. When provided, reads `cfg.model.*` for XGB HP instead of hardcoded defaults. `FALLBACK_N_ESTIMATORS` stays as guard. |
| `src/models/lstm.py` | Add `@register_model("lstm")` decorator to `FVGLSTMClassifier`. |
| `src/models/xgboost_baseline.py` | Add `@register_model("xgb")` decorator to `XGBoostFVGClassifier`. |
| `src/training/loss.py` | Add `@register_loss("weighted_ce")` to `WeightedCE`, `@register_loss("focal")` to `FocalLoss`. |

#### NOT touched in Phase 1
`src/rigor/seed_sweep.py`, all rigor scripts, `paper_trade.py`, `inspect/`.

---

### Phase 2 — Rigor scripts + remaining YAMLs
**Estimate: 1.5 hours**

#### Modified files
| File | Change |
|------|--------|
| `scripts/rigor/multiseed_run.py` | Replace `--config best_hp.json` → `--config experiments/lstm_g1.yaml`. Load via `load_experiment()`. Build `SeedSweepConfig` from cfg fields via adapter factory `_cfg_to_seed_sweep(cfg)`. `--seeds` CLI arg overrides `cfg.train.seeds`. `--no-dropout`, `--no-l2`, `--loss`, `--gamma` become overrides: `{"train.ablation_no_dropout": True}` etc. |
| `scripts/rigor/window_sweep.py` | Replace JSON load with `load_experiment()`. Iterate window sizes by cloning config with override: `load_experiment(path, {"data.window_size": W})`. |
| `scripts/rigor/tune_lstm.py` | Add `--base-config experiments/lstm_g1.yaml` arg. Load base cfg, Optuna trial overrides HP fields. On study complete, write `cfg.to_yaml(ts_dir / "lstm_g1_tuned.yaml")` alongside existing `best_lstm_config.json`. |
| `scripts/rigor/tune_xgboost.py` | Same pattern as tune_lstm. |
| `scripts/rigor/threshold_sweep.py` | Add `--config` arg. Use `cfg.data.*` for paths, `cfg.eval.*` for bootstrap params. |
| `scripts/rigor/shap_xgb.py` | Add `--config` arg. Use `cfg.data.data_dir` for data paths. |
| `scripts/rigor/bootstrap_ci_multiseed.py` | Add `--config` arg. Use `cfg.eval.bootstrap_n_iter`, `cfg.eval.bootstrap_block_size`. |
| `scripts/rigor/threshold_multiseed.py` | Add `--config` arg. Use `cfg.data.*` for paths. |

#### New file
| File | Action |
|------|--------|
| `src/rigor/seed_sweep.py` | Add `from_experiment_config(cfg: ExperimentConfig) -> SeedSweepConfig` classmethod/factory function. Maps: `cfg.model.arch` → `model_type`, `cfg.model.*` → `hyperparams` dict, `cfg.train.seeds` → `seeds`, `cfg.train.loss` → `loss_type`, `cfg.train.focal_gamma` → `focal_gamma`, `cfg.runtime.*` → dirs, `cfg.train.ablation_*` → ablation flags. |

#### NOT touched in Phase 2
`paper_trade.py`, `inspect/`, `_workers/`. `SeedSweepConfig` dataclass stays — only its construction changes.

---

### Phase 3 — inspect + paper_trade verification
**Estimate: 0.5 hours**
**Scope:** Read-only verification pass. No code changes required unless gaps found.

- `src/inspect/adapters/` — already meta.json-driven. `cfg.model.arch` maps to adapter key. Verify adapter registry accepts `"lstm"` and `"xgb"` keys matching `MODELS` dict. No code change expected.
- `scripts/paper_trade.py` — reads checkpoint path at inference time, no HP config needed. Add `--config` arg for `cfg.data.*` paths only (optional, defaults work). Minimal change.
- Verify `scripts/inspect_models.py` still works with existing `meta.json` sidecars.

---

## Unified `train.py` design

```
scripts/training/train.py --config experiments/lstm_g1.yaml [--set train.seed=17] [--seed 42]

  1. load_experiment(args.config, overrides from --set)
  2. Dispatch on cfg.model.arch:
       "lstm"  → _train_lstm(cfg)
       "xgb"   → _train_xgb(cfg)  (subprocess worker path)
  3. _train_lstm(cfg):
       - set_seed(cfg.train.seeds[0] if --seed not provided)
       - select_device(cfg.train.device)  # always returns cpu for lstm
       - load_datasets(cfg)
       - make_loaders(cfg)
       - build model from MODELS[cfg.model.arch](**cfg.model.model_dump(exclude={"arch"}))
       - build criterion from LOSSES[cfg.train.loss](weights)
       - build optimizer: Adam or AdamW per cfg.train.optimizer
       - build scheduler: none or OneCycleLR per cfg.train.scheduler
       - train loop (same logic as current train_lstm.py train())
       - eval + write meta.json
  4. Checkpoint path: {cfg.runtime.checkpoint_dir}/{cfg.model.arch}/{cfg.name}_seed{seed}.pt
```

`train_lstm.py` becomes:
```python
#!/usr/bin/env python3
"""Thin wrapper: calls train.py with lstm_g1.yaml as default config."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.train import main
main(default_config="experiments/lstm_g1.yaml")
```

This preserves `python scripts/training/train_lstm.py --seed 42` as a valid invocation.

---

## Test strategy

### `tests/config/test_schema.py`

```
test_lstm_model_config_valid          — valid fields parse correctly
test_xgb_model_config_valid           — valid fields parse correctly
test_discriminated_union_lstm         — arch="lstm" → LSTMModelConfig
test_discriminated_union_xgb          — arch="xgb" → XGBModelConfig
test_missing_required_arch_raises     — model without arch raises ValidationError
test_wrong_type_hidden_size_raises    — hidden_size="128" (str) raises ValidationError
test_experiment_config_defaults       — ExperimentConfig() populates all defaults
test_train_config_seeds_default       — seeds defaults to [42]
test_optimizer_literal_invalid        — optimizer="sgd" raises ValidationError
test_device_literal_invalid           — device="tpu" raises ValidationError
```

### `tests/config/test_loader.py`

```
test_load_experiment_lstm_g1          — loads experiments/lstm_g1.yaml, checks HP fields
test_load_experiment_xgb_g1           — loads experiments/xgb_g1.yaml
test_override_train_seed              — override {"train.seeds": [99]} applies
test_override_model_hidden_size       — override {"model.hidden_size": 64} applies
test_deep_merge_base_yaml             — _base.yaml defaults filled for missing keys
test_nested_override_path             — dotted key "data.window_size" set correctly
test_experiment_from_json_lstm        — loads best_hp_lstm.json, produces LSTMModelConfig
test_experiment_from_json_xgb         — loads best_hp_xgb.json, produces XGBModelConfig
test_json_metadata_keys_stripped      — val_macro_f1, trial_number not in model fields
test_missing_config_file_raises       — FileNotFoundError on bad path
```

### Affected existing tests

`tests/` currently has 251 passing tests. Affected areas:
- `tests/models/` — if tests import `FVGLSTMClassifier` directly, decorator addition is invisible (non-breaking).
- `tests/training/` — if tests call `train_lstm.py` as subprocess, thin wrapper preserves interface.
- No test currently imports from `src/config/` (new module) — no regressions expected from Phase 1.

Phase 2: rigor script tests (if any test `SeedSweepConfig` construction) will need the adapter factory added.

**Gate rule:** run `pytest` after each phase. Must stay at 251/251 before proceeding.

---

## Migration order with backwards-compat checkpoints

```
Phase 1
├── Create src/config/ module (no existing code touched yet)
├── Create experiments/ YAMLs
├── Write tests/config/ tests → run → must pass
├── Modify src/models/lstm.py: add @register_model (decorator is additive, non-breaking)
├── Modify src/models/xgboost_baseline.py: add @register_model
├── Modify src/training/loss.py: add @register_loss
├── Modify train_lstm.py: --config optional, backwards compat when omitted
│   └── TEST: python scripts/training/train_lstm.py --seed 42 (no --config) → still works
│   └── TEST: python scripts/training/train_lstm.py --config experiments/lstm_g1.yaml → works
├── Modify train_xgboost.py: same pattern
├── Create scripts/training/train.py
├── CHECKPOINT: pytest → 251/251
│
Phase 2
├── Add from_experiment_config() to src/rigor/seed_sweep.py
├── Modify multiseed_run.py: --config accepts YAML, builds SeedSweepConfig via adapter
│   └── Old JSON paths still accepted (loader.experiment_from_json fallback by extension)
├── Modify window_sweep.py
├── Modify tune_lstm.py, tune_xgboost.py
├── Modify threshold_sweep.py, shap_xgb.py, bootstrap_ci_multiseed.py, threshold_multiseed.py
├── CHECKPOINT: pytest → 251/251
│
Phase 3
├── Read-only verify: inspect adapters, paper_trade
├── Minimal touch if needed
└── CHECKPOINT: pytest → 251/251
```

**Backwards compat guarantee:** all scripts accept `--config` as **optional**. When omitted, current argparse defaults apply unchanged. This means any existing shell commands in CLAUDE.md or docs continue to work without modification.

---

## Risk register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Pydantic discriminated union: `model` field must have `arch` key in YAML | Low | `_base.yaml` sets `model.arch: lstm`; missing arch in experiment YAML falls back to base. Pydantic raises clear ValidationError if arch is invalid. |
| `@register_model` decorator fires only if module is imported | Low | `src/config/_model_registrations.py` imports both model classes — force-imported by `registry.py`. Same pattern as LABELLERS. |
| Circular import: registry.py → models/ → training/ → config/ | Medium | `_model_registrations.py` is a leaf file with no imports back into config/. Schema and loader never import from models/. Registry imports models/ only through the `_registrations` leaf. |
| `train_lstm.py` constants block (LR=3e-3 etc.) still hardcoded — confusing when --config overrides them | Low | Add docstring comment in constants block: "Fallback defaults when --config not provided. Ignored when config is loaded." |
| `SeedSweepConfig.hyperparams` is a flat dict — model fields need to be extracted from `cfg.model` into that dict | Low | `from_experiment_config()` builds hyperparams dict via `cfg.model.model_dump(exclude={"arch"})`. All HP fields end up in flat dict. |
| `best_hp_lstm.json` has `best_value` key (not `val_macro_f1`) — different from `best_hp_xgb.json` | Low | Strip set in `experiment_from_json` includes both. Confirmed in data: lstm JSON uses `best_value`, xgb uses `val_macro_f1`. |
| `window_sweep.py` trains LSTM with hardcoded `patience=15`, `loss=weighted_ce` — must not change behavior | Low | Config-driven override only changes `data.window_size` per iteration. Other fields inherit from experiment YAML (which specifies patience=15, loss=weighted_ce). Behavior identical. |
| Optuna trial params in `tune_lstm.py` use `trial.suggest_*` — must still override base config HP | Low | Optuna callback creates override dict from trial params and calls `load_experiment(base_path, overrides=trial_dict)` per trial. |
| `train.py` checkpoint path changes from `lstm_seed{N}.pt` to `{cfg.name}_seed{N}.pt` — breaks inspect_models | Medium | Keep existing checkpoint naming: `{model.arch}_seed{N}{label_tag}.pt`. `cfg.name` used for reports only, not checkpoint filenames. Verify inspect_models adapter reads checkpoint by path, not by name pattern. |
| CNN-LSTM not yet implemented — `MODELS["cnn_lstm"]` will KeyError | None now | CNN-LSTM phase adds `CNNLSTMModelConfig` to union and `@register_model("cnn_lstm")`. No action in this refactor. |

---

## Wall-clock estimates

| Phase | Estimate | Notes |
|-------|----------|-------|
| Phase 1 | 2h | Schema + loader + registry + 5 YAMLs + train.py + wrappers + tests |
| Phase 2 | 1.5h | 8 rigor scripts + seed_sweep adapter |
| Phase 3 | 0.5h | Verify-only, minimal touch |
| **Total** | **~4h** | |

---

## Subagent ownership

| Agent | Scope |
|-------|-------|
| `@nb-build` (orchestrator, sonnet) | Executes all three phases in sequence. Reads this plan before starting. |
| `@nb-test` (haiku) | Spawned by nb-build after Phase 1 and Phase 2. Runs `pytest` + writes test log. |
| `@nb-review` (sonnet) | Spawned by nb-build after Phase 3. Checks schema completeness, circular import safety, backwards compat. |

---

## Decision gates

### Gate 1 (end of Phase 1)
- `pytest` → 251/251
- `python scripts/training/train_lstm.py --seed 42` (no --config) runs without error
- `python scripts/training/train.py --config experiments/lstm_g1.yaml --set train.seeds=[42]` runs without error
- `python -c "from src.config import load_experiment; cfg = load_experiment('experiments/lstm_g1.yaml'); print(cfg.model.hidden_size)"` prints `128`

### Gate 2 (end of Phase 2)
- `pytest` → 251/251
- `python scripts/rigor/multiseed_run.py --model lstm --config experiments/lstm_g1.yaml --seeds 42` runs without error
- Old invocation `multiseed_run.py --model lstm --config reports/rigor/2026-05-13/G1/best_hp_lstm.json --seeds 42` still works (JSON compat by extension detection)

### Gate 3 (end of Phase 3)
- `pytest` → 251/251
- `python scripts/inspect_models.py --models lstm xgboost --lookahead-bars 20` runs without error
- `python scripts/paper_trade.py --model lstm:checkpoints/lstm/lstm_seed42.pt --session lstm-001 --dry-run` runs without error

---

## CLAUDE.md commands update (after all phases)

The following new commands should be added to CLAUDE.md `## Commands` section:

```bash
# Run any experiment by config
python scripts/training/train.py --config experiments/lstm_g1.yaml

# Override a field without editing YAML
python scripts/training/train.py --config experiments/lstm_g1.yaml --set train.seeds=[0,17,42]

# Multi-seed sweep via experiment config
python scripts/rigor/multiseed_run.py --model lstm --config experiments/lstm_g1.yaml

# Window sweep
python scripts/rigor/window_sweep.py --config experiments/lstm_g1.yaml --windows 30 60 90 120
```

The old `train_lstm.py` and `train_xgboost.py` commands remain valid as thin wrappers.
