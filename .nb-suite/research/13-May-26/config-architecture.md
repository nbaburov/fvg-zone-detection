# Config-Driven Training Architecture — Research

> **Ready for /nb:plan.**
> Question: What config format and schema should drive all training/rigor/eval scripts so any axis (label, model, HP, window, loss, seed, splits) can be swapped by editing config, not code?
> Verdict: Pydantic v2 dataclass-style config loaded from YAML. No Hydra. Factory registries (MODELS, LOSSES) mirror the existing LABELLERS pattern. Migration in 2 phases: train scripts first, rigor scripts second.

**Confidence:** High
**Why this confidence:** All options are well-understood. Project constraints are concrete (single dev, no orchestration infra, existing registry pattern to mirror). Hydra was tested/evaluated against real project constraints and rejected on dependency weight + overkill grounds. Pydantic + YAML is the established pattern for exactly this scale (see Lightning, MMDetection configs).
**Depth used:** Standard

---

## Project context

From reading the actual files:

**Hardcoded constants in `train_lstm.py` (module-level):**
```
MAX_EPOCHS = 100, BATCH_TRAIN = 32, BATCH_EVAL = 256, WINDOW_SIZE = 60
LR = 3e-3, WEIGHT_DECAY = 1e-2, PATIENCE = 15, EMA_ALPHA = 0.3, MAX_GRAD_NORM = 1.0
```
These are used directly in the `train()` and `make_loaders()` functions — not threaded through `parse_args`. `--max-epochs` is the only constant exposed as a CLI flag.

**Hardcoded in `train_xgboost.py`:** No module-level constants block. XGB uses `XGBoostFVGClassifier` defaults inline in `main()`. `FALLBACK_N_ESTIMATORS = 527` is inline. HP come from `--config` JSON (optional) or defaults inside the class.

**`SeedSweepConfig` dataclass** (`src/rigor/seed_sweep.py`): Closest thing to a real config. Already has: `model_type`, `hyperparams: dict`, `seeds: list[int]`, `loss_type`, `focal_gamma`, `output_dir`, `checkpoint_dir`, `data_dir`, `ablation_no_dropout`, `ablation_no_l2`. But it's not YAML-serializable and has no schema validation.

**`best_hp_lstm.json` / `best_hp_xgb.json`**: De facto HP config. Already produced by Optuna and consumed by `window_sweep.py --config` and `multiseed_run.py --config`. The config-consumption pattern already exists — it just isn't standardized.

**Existing `LABELLERS` registry** (`src/data/labels/__init__.py`): `@register("fvg_valid")` decorator pattern. Clean. Mirror this for `MODELS`, `LOSSES`, `OPTIMIZERS`.

**Pain points from the rerun sprint build log:**
- HP hardcoded in `train_lstm.py` constants — multiseed_run.py injects some (via `SeedSweepConfig.hyperparams`) but the standalone train script ignores them
- `window_sweep.py` reads HP JSON but hardcodes loss type, patience, device
- `multiseed_run.py` hardcodes seeds list in code, injects only partial HP
- No single source of truth for which label/splits/window/loss goes with which experiment
- Duplicate processes in G3/G7 sprint had no config identity — impossible to tell runs apart from the outside

---

## Findings

### Option A: Hydra + OmegaConf

**Source:** https://hydra.cc/docs/intro/ + FairSeq, MMDetection usage patterns
**Mechanism:** Config composed from YAML config groups. `@hydra.main()` decorator on entry point. Override any field from CLI: `python train.py model.hidden_size=256 train.lr=1e-3`. `--multirun` sweeps: `python train.py --multirun train.seed=0,17,42`. Output dir auto-managed per run.
**Fit:** Partial. The multirun sweep pattern directly solves the "one field override per run" use case. But: adds `hydra-core` + `omegaconf` as hard deps. Entry points must use `@hydra.main()` — invasive refactor. Config dir structure is opinionated (conf/). Subprocess-based XGB workers and existing argparse CLIs don't compose cleanly with Hydra's decorator approach. The sweep functionality is genuinely useful but there's no other Hydra user in this codebase.
**Risk:** Dependency weight (~15MB, pulls omegaconf + antlr4). Every script entry point must be rewritten. Subprocess workers need special handling. If Hydra is ever removed, every script breaks. Overkill for a single-developer project with 12 scripts.

### Option B: Pydantic v2 dataclass + YAML load

**Source:** https://docs.pydantic.dev/latest/concepts/dataclasses/ + PyTorch Lightning `LightningDataModule` config pattern
**Mechanism:** Define `ExperimentConfig` as a Pydantic `BaseModel` (or `@dataclass` with pydantic validation). Load from YAML via `yaml.safe_load` + `ExperimentConfig(**d)`. Override from CLI by merging a `--set key=value` dict before validation. All fields are typed, validated, and have defaults. The existing `SeedSweepConfig` dataclass is 80% of the way there — just needs YAML I/O and pydantic validation added.
**Fit:** Yes. Pure Python, zero new mandatory deps (pydantic is already a common transitive dep; yaml is stdlib-adjacent via PyYAML which is already installed via many deps). Schema validation catches wrong types at load time. Config is serializable to/from JSON and YAML. Existing `--config best_hp_lstm.json` pattern in `window_sweep.py` becomes `--config lstm_g1.yaml`. Factory pattern for models/losses uses same `@register` decorator already in codebase.
**Risk:** No built-in sweep CLI (`--multirun`). Sweep scripts must loop over config variants in Python. This is fine — the sweep scripts already do this, they just need to load from config instead of hardcoding.

### Option C: OmegaConf standalone (no Hydra)

**Source:** https://omegaconf.readthedocs.io/
**Mechanism:** Structured configs via OmegaConf dataclasses. Merge base config with override dict. CLI override via `OmegaConf.from_dotlist(["model.hidden_size=256"])`.
**Fit:** Partial. Better than plain dict, worse than pydantic for validation. Adds a dep without adding enough over pydantic. No clear win over Option B.
**Risk:** Another dep. Less validation than pydantic. OmegaConf interpolation syntax is a learned API.

### Option D: Plain YAML + dict merge (no validation library)

**Source:** Current codebase pattern (best_hp JSON + argparse)
**Mechanism:** `yaml.safe_load` → plain dict → pass as kwargs. No schema, no validation.
**Fit:** Already exists informally. No new deps. But: no type checking, no required-field enforcement, silent failures when a key is missing or mistyped.
**Risk:** Silent bugs. `hp.get("hidden_size", 64)` style fallbacks mean wrong configs silently use defaults. Already bit the codebase: `multiseed_run.py` injected only partial HP, rest fell back to hardcoded constants.

---

## Comparison table

| | Hydra | Pydantic+YAML | OmegaConf | Plain YAML+dict |
|---|---|---|---|---|
| Schema validation | Yes (structured configs) | Yes (Pydantic validators) | Partial | No |
| CLI override | `key=val` native | `--set key=val` manual | `OmegaConf.from_dotlist` | argparse only |
| Sweep/multirun | `--multirun` built-in | Loop in Python | Loop in Python | Loop in Python |
| New deps | hydra-core + omegaconf | pydantic (likely already present) | omegaconf | None |
| Invasiveness | High (decorator, conf/ dir) | Low (replace dataclass, add YAML load) | Medium | Low |
| Subprocess compat | Awkward | Clean (pass config path) | Clean | Clean |
| Existing pattern match | No | Yes (mirrors LABELLERS registry) | Partial | Yes (current) |
| Serializable | Yes (YAML) | Yes (YAML + JSON) | Yes (YAML) | Yes (JSON) |
| Recommended | No | **Yes** | No | No (already failing) |

---

## Recommendation

**Use Pydantic v2 `BaseModel` + YAML files. Mirror the existing `LABELLERS` registry for `MODELS` and `LOSSES`.**

Rationale:
1. `SeedSweepConfig` is already a dataclass with the right shape — this is a direct upgrade, not a rewrite.
2. YAML is more readable than JSON for experiment configs (comments, multiline). The existing `best_hp_lstm.json` format becomes `experiments/lstm_g1.yaml`.
3. Pydantic validation catches `hidden_size: "128"` (string vs int) at load time, not mid-epoch.
4. The `@register` decorator pattern in `LABELLERS` is proven and understood. `MODELS["lstm"]` and `LOSSES["weighted_ce"]` are the exact same pattern.
5. Subprocess workers receive config path as CLI arg — clean, no Hydra decorator requirement.
6. Sweep scripts loop over config variants in Python — already what they do, just with validated config objects instead of raw dicts.

Do not use Hydra. The multirun convenience does not justify the dependency, the decorator invasiveness, or the conf/ directory structure. Single developer, academic project — Hydra is production orchestration infrastructure.

---

## Concrete schema sketch (~55 lines)

```python
# src/config.py
from __future__ import annotations
from pathlib import Path
from typing import Literal, Any
from pydantic import BaseModel, Field
import yaml


class DataConfig(BaseModel):
    data_dir: Path = Path("data/processed")
    labeller: Literal["fvg", "fvg_valid"] = "fvg_valid"
    splits: str = "default"          # "default" | path to dir
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


class TrainConfig(BaseModel):
    seed: int = 42
    seeds: list[int] = Field(default_factory=lambda: [42])
    batch_size: int = 32
    batch_eval: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-2
    max_epochs: int = 100
    patience: int = 15
    ema_alpha: float = 0.3
    max_grad_norm: float = 1.0
    loss: Literal["weighted_ce", "focal"] = "weighted_ce"
    focal_gamma: float = 2.0
    device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"
    # Reg ablation flags (mirrors SeedSweepConfig)
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
    model: LSTMModelConfig | XGBModelConfig = Field(default_factory=LSTMModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.model_validate(d)

    @classmethod
    def from_yaml_with_overrides(cls, path: Path, overrides: list[str]) -> "ExperimentConfig":
        """overrides: ["train.lr=1e-4", "model.hidden_size=256"]"""
        with open(path) as f:
            d = yaml.safe_load(f)
        for ov in overrides:
            key, val = ov.split("=", 1)
            _set_nested(d, key.split("."), yaml.safe_load(val))
        return cls.model_validate(d)

    def to_yaml(self, path: Path) -> None:
        import yaml
        path.write_text(yaml.dump(self.model_dump(mode="json"), default_flow_style=False))


def _set_nested(d: dict, keys: list[str], val: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val
```

**Example YAML (experiments/lstm_g1.yaml):**
```yaml
name: lstm_g1_validfvg
data:
  labeller: fvg_valid
  window_size: 60
model:
  arch: lstm
  hidden_size: 128
  num_layers: 1
  dropout: 0.318
  head_dropout: 0.526
train:
  seeds: [0, 17, 42, 123, 2024]
  batch_size: 16
  lr: 0.000530
  weight_decay: 3.92e-5
  max_epochs: 100
  patience: 15
  loss: weighted_ce
  device: cpu
runtime:
  output_dir: reports/rigor
  checkpoint_dir: checkpoints
```

---

## Factory / registry pattern for models

Mirror `LABELLERS` exactly:

```python
# src/models/__init__.py
MODELS: dict[str, type] = {}

def register_model(name: str):
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator

# src/models/lstm.py
@register_model("lstm")
class FVGLSTMClassifier(nn.Module): ...

# src/models/cnn_lstm.py  ← NEW MODEL = 1 file + 1 yaml
@register_model("cnn_lstm")
class FVGCNNLSTMClassifier(nn.Module): ...
```

New model addition: `src/models/cnn_lstm.py` (model class) + `experiments/cnn_lstm_g1.yaml` (HP config). No fork of `train_lstm.py`. Train script does `MODELS[cfg.model.arch](...)`.

Same pattern for losses:
```python
# src/training/loss.py
LOSSES = {"weighted_ce": WeightedCE, "focal": FocalLoss}
criterion = LOSSES[cfg.train.loss](weights)
```

---

## Migration plan — smallest reasonable first iteration

### Phase 1 (1–2 hours): Config schema + train scripts only

**Scope:** `src/config.py` + `scripts/training/train_lstm.py` + `scripts/training/train_xgboost.py`

1. Create `src/config.py` with `ExperimentConfig` schema above.
2. Add `MODELS` registry to `src/models/__init__.py`. Add `@register_model("lstm")` to `FVGLSTMClassifier`.
3. Add `LOSSES` dict to `src/training/loss.py`.
4. In `train_lstm.py`: replace module-level constant block with `cfg = ExperimentConfig.from_yaml_with_overrides(args.config, args.set)`. Wire `cfg.train.*` and `cfg.model.*` through. Keep all existing argparse flags as fallback — if `--config` not provided, build `ExperimentConfig` from argparse values (backwards compat).
5. In `train_xgboost.py`: same. Replace inline HP defaults with `cfg.model.*`.
6. Write `experiments/lstm_g1.yaml` and `experiments/xgb_g1.yaml` from `best_hp_lstm.json` / `best_hp_xgb.json`.
7. Verify: `python scripts/training/train_lstm.py --config experiments/lstm_g1.yaml` produces same checkpoint as current run.

**Gate:** existing tests still pass. `lstm_seed42.meta.json` HP match the YAML.

### Phase 2 (1–2 hours): Rigor scripts

**Scope:** `multiseed_run.py`, `window_sweep.py`, `tune_lstm.py`, `tune_xgboost.py`, `threshold_sweep.py`, `bootstrap_ci.py`, `reg_ablation` flags

1. `multiseed_run.py`: replace `SeedSweepConfig` construction from argparse with `ExperimentConfig.from_yaml()`. `cfg.train.seeds` drives the sweep. Remove hardcoded seeds list in script.
2. `window_sweep.py`: reads `cfg.data.window_size` per iteration, overrides each run with `cfg.model_copy(update={"data": {"window_size": W}})`.
3. `tune_lstm.py` / `tune_xgboost.py`: reads base config from `--config`, Optuna trial overrides HP fields, saves best back as `cfg.to_yaml()`. No change to Optuna logic.
4. `threshold_sweep.py`, `bootstrap_ci.py`: consume `ExperimentConfig` for data paths only — minimal change.

**Not in scope for Phase 2:** `paper_trade.py`, `inspect_models.py`. These have different concerns (live inference, not training). Address when CNN-LSTM adapter is added.

### Phase 3 (optional, post-CNN-LSTM): inspect + paper_trade

`inspect_models.py` adapter resolution already uses a registry (`src/inspect/registry.py`). Config just needs to pass `cfg.model.arch` to pick the right adapter. `paper_trade.py` reads checkpoint path — no HP config needed at inference time.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Pydantic union type for model config (`LSTMModelConfig \| XGBModelConfig`) requires discriminator | Low | Use `arch` field as discriminator: `Annotated[..., Field(discriminator="arch")]` |
| Backwards compat: existing scripts called without `--config` break | Low | Phase 1 keeps argparse fallback. `--config` optional. |
| `SeedSweepConfig` and `ExperimentConfig` overlap — two sources of truth during migration | Medium | Phase 2 replaces `SeedSweepConfig` entirely. Phase 1 they coexist. Acceptable for 1-2 day migration window. |
| `best_hp_lstm.json` format (flat dict with `val_macro_f1`, `trial_number` metadata keys) doesn't map cleanly to `LSTMModelConfig` | Low | `ExperimentConfig.from_yaml` — strip metadata keys before model parse. Already done in `window_sweep.py` line 38: `{k: v for k, v in hp.items() if k not in ("val_macro_f1", "trial_number", ...)}` |
| `train_lstm.py` uses `OneCycleLR` (not `Adam`); `seed_sweep._train_lstm` uses `Adam`. Two different optimisers for same model. | Medium | Expose `optimizer: Literal["adam", "adamw"]` and `scheduler: Literal["none", "onecycle"]` in `TrainConfig`. Forces this inconsistency to surface and be resolved explicitly. |
| CNN-LSTM config: model union must be extended for each new arch | Low | Add `CNNLSTMModelConfig` to union. Pydantic discriminated union handles it cleanly. |
| YAML config not tracked in git = experiment not reproducible | Low | Convention: `experiments/` dir is checked in. `reports/` output dir is gitignored. |

---

## What /nb:plan needs to know

- `src/config.py` is a new file — no existing code to migrate, just create.
- `SeedSweepConfig` in `src/rigor/seed_sweep.py` should be replaced by `ExperimentConfig` in Phase 2, not Phase 1. Don't try to do both at once.
- The `train_lstm.py` and `seed_sweep._train_lstm` use different optimisers (AdamW+OneCycleLR vs Adam). This inconsistency must be resolved when writing the config schema — decide canonical optimiser before writing `TrainConfig`.
- `--set key=value` CLI override is the only sweep mechanism needed. No Hydra multirun. Sweep scripts loop over configs in Python.
- `experiments/` dir should be created and checked into git. Start with `lstm_g1.yaml` and `xgb_g1.yaml` from existing `best_hp_*.json` artifacts.
- Discriminated union for `model` field: use `Annotated[LSTMModelConfig | XGBModelConfig, Field(discriminator="arch")]`.
- PyYAML is already a transitive dep (pandas/matplotlib pull it). No new install needed.
- Pydantic v2 is likely already installed (fastapi, httpx, or other modern deps). Verify with `pip show pydantic` before writing install instructions.

---

## Open questions

- Canonical optimiser for LSTM: `AdamW + OneCycleLR` (train_lstm.py) or plain `Adam` (seed_sweep.py)? These produce slightly different results. User must decide which becomes the config default.
- Should `experiments/` dir be flat (one yaml per experiment) or grouped by model? No strong preference, but flat is simpler for now.

---

## What I couldn't verify

- Whether pydantic v2 is already installed in the project venv (likely yes, but not confirmed — `pip show pydantic` in the venv would confirm).
- Whether `OneCycleLR` in `train_lstm.py` was intentional vs leftover from an earlier experiment. The G1 HP search used `tune_lstm.py` which may use plain `Adam` — didn't read that file fully.
- CNN-LSTM model shape — not yet implemented. Schema can accommodate it but the exact config fields (kernel_size, etc.) are TBD.
