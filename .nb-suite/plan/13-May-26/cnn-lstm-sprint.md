# Plan: CNN-LSTM Build Sprint

**Date:** 2026-05-13
**Feeds:** @nb-build
**Research source:** `.nb-suite/research/13-May-26/cnn-lstm-architecture.md`
**Config pattern:** `.nb-suite/plan/13-May-26/config-refactor.md`
**Rigor mirror:** `.nb-suite/plan/12-May-26/post-research-rerun-sprint.md` (Phase D pattern)
**Goal:** Deliver fully-vetted CNN-LSTM baseline on canonical ValidFVG. Same rigor as LSTM/XGB.
**Baseline to beat:** LSTM 0.599 ± 0.025. Target: 0.62–0.65. Stretch: approach XGB 0.721.

---

## Decision log (resolved before build)

### D1: CNNLSTMModelConfig in schema
Add `CNNLSTMModelConfig` to `src/config/schema.py` union. `arch: Literal["cnn_lstm"]`. Fields: `n_conv_layers`, `conv_filters`, `kernel_size`, `use_pool`, `pool_type`, `lstm_hidden`, `lstm_layers`, `dropout`, `head_dropout`. Update `ModelConfig` union to include it. `_base.yaml` keeps `arch: lstm` as default — no regression.

### D2: Registration pattern
`register_model("cnn_lstm")(FVGCNNLSTMClassifier)` added to `src/config/_model_registrations.py`. Same leaf-file pattern. No circular import.

### D3: Training script
Create `scripts/training/train_cnn_lstm.py` as thin wrapper calling `scripts/training/train.py` with `default_config="experiments/cnn_lstm_g1.yaml"` — mirrors `train_lstm.py` wrapper pattern exactly. `train.py` dispatcher adds `"cnn_lstm"` branch calling `_train_lstm`-equivalent logic (same training loop, CNN-LSTM model built from registry).

### D4: Optuna script
Create `scripts/rigor/tune_cnn_lstm.py` mirroring `tune_lstm.py`. Shares same `LSTMObjective` plumbing — only diff is `CNNLSTMObjective` class with 7-dim search space. Output dir: `reports/rigor/<ts>/cnn_lstm_G1/`. Best JSON: `best_hp_cnn_lstm.json`.

### D5: multiseed_run.py extension
`multiseed_run.py` `--model` choices need `"cnn_lstm"` added. The config-driven path already generalises via `MODELS[cfg.model.arch]` — only the argparse `choices=["lstm","xgboost"]` list needs updating.

### D6: Inspect adapter
`src/inspect/adapters/` needs `CNNLSTMAdapter` — same ABC pattern as `LSTMAdapter`. Required for `inspect_models.py` to work on CNN-LSTM checkpoints. Scope: Phase 1 (needed for smoke test output interpretation).

### D7: Window sweep (G7)
W ∈ {30, 60, 90, 120}. Use existing `window_sweep.py` — pass `--config experiments/cnn_lstm_g1.yaml`. No new script needed.

### D8: class_weights file
Use `data/processed/class_weights_fvg_valid.json` (keys `"0"`, `"1"`, `"2"`) unchanged. `tune_cnn_lstm.py` loads this, not `class_weights.json`. Confirm filename in script — LSTM rigor used the fvg_valid weights file.

### D9: Pool padding
Research doc specifies `MaxPool1d(kernel_size=2, stride=1, padding=1)` when `use_pool=True`. This preserves sequence length (does NOT halve T). stride=2 not searched in G1. Document in YAML comment.

---

## File-by-file change list

### Phase 1 — Model + config + registration + base YAML + smoke test (1 hr)

#### New files

| File | Action | Notes |
|------|--------|-------|
| `src/models/cnn_lstm.py` | CREATE | `FVGCNNLSTMClassifier` per research doc topology |
| `experiments/cnn_lstm_base.yaml` | CREATE | Starting-point HP, W=60, seeds=[42] single-seed smoke |
| `experiments/cnn_lstm_g1.yaml` | CREATE | Placeholder; populated after G1 HP search completes |
| `scripts/training/train_cnn_lstm.py` | CREATE | Thin wrapper → `train.py` with `cnn_lstm_g1.yaml` default |
| `scripts/rigor/tune_cnn_lstm.py` | CREATE | Optuna G1 script, 50 trials, 7-dim space |
| `tests/models/test_cnn_lstm.py` | CREATE | 4 smoke tests: shape, forward, gradient, num_params |

#### Modified files

| File | Change |
|------|--------|
| `src/config/schema.py` | Add `CNNLSTMModelConfig` class; update `ModelConfig` union: `LSTMModelConfig \| XGBModelConfig \| CNNLSTMModelConfig` |
| `src/config/_model_registrations.py` | Add `from src.models.cnn_lstm import FVGCNNLSTMClassifier` + `register_model("cnn_lstm")(FVGCNNLSTMClassifier)` |
| `src/config/__init__.py` | Export `CNNLSTMModelConfig` |
| `scripts/training/train.py` | Add `"cnn_lstm"` to model dispatch — calls same `_train_lstm`-style function with CNN-LSTM model built via `MODELS["cnn_lstm"](**model_kwargs)` |
| `scripts/rigor/multiseed_run.py` | Add `"cnn_lstm"` to `--model choices` list |

#### NOT touched in Phase 1
`src/inspect/`, rigor scripts (tune/multiseed/window), existing LSTM/XGB YAML, paper_trade.

**Phase 1 gate:**
```bash
.venv/bin/python -m pytest tests/models/test_cnn_lstm.py -v         # 4 new tests green
.venv/bin/python -m pytest                                           # 280+4 = 284 green
.venv/bin/python -c "from src.config import CNNLSTMModelConfig; print('ok')"
.venv/bin/python scripts/training/train_cnn_lstm.py \
    --config experiments/cnn_lstm_base.yaml --set "train.seeds=[42]" --set "train.max_epochs=2"
```
Single-seed 2-epoch smoke must complete without error. If val macro-F1 = NaN → BatchNorm instability → switch to GroupNorm (risk R3).

---

### Phase 2 — Optuna G1 (50 trials) + multi-seed G2 (3 hr)

#### New files

| File | Action | Notes |
|------|--------|-------|
| `reports/rigor/2026-05-13/cnn_lstm_G1/best_hp_cnn_lstm.json` | GENERATED | Optuna best trial output |
| `reports/rigor/2026-05-13/cnn_lstm_G1/cnn_lstm_optuna_history.html` | GENERATED | Plotly viz |
| `reports/rigor/2026-05-13/cnn_lstm_G1/cnn_lstm_optuna_importances.html` | GENERATED | param importance |
| `reports/rigor/2026-05-13/cnn_lstm_G2/multiseed_summary_cnn_lstm_weighted_ce.md` | GENERATED | G2 multi-seed report |
| `checkpoints/cnn_lstm/cnn_lstm_seed{0,17,42,123,2024}.pt` | GENERATED | 5 checkpoints |
| `checkpoints/cnn_lstm/cnn_lstm_seed{N}.meta.json` | GENERATED | sidecar per checkpoint |

#### Modified files

| File | Change |
|------|--------|
| `experiments/cnn_lstm_g1.yaml` | UPDATE with best HP from G1 JSON after study completes |

**CLI invocations — G1:**
```bash
# G1: Optuna HP search — 50 trials, 7-dim, ValidFVG, W=60
.venv/bin/python scripts/rigor/tune_cnn_lstm.py \
    --config experiments/cnn_lstm_base.yaml \
    --n-trials 50 \
    --study-name cnn_lstm_fvg_validfvg \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G1 \
    --max-epochs 50
# Expect: ~30-90 min CPU. Best val macro-F1 should land ≥ 0.40 (gate).
```

**G1 gate check:**
- Best trial val macro-F1 ≥ 0.40. If < 0.40 → STOP. Review search space, check class weights file path, confirm data pipeline matches LSTM.
- Populate `experiments/cnn_lstm_g1.yaml` with best trial HP.

**CLI invocations — G2:**
```bash
# G2: Multi-seed with G1 best HP
.venv/bin/python scripts/rigor/multiseed_run.py \
    --model cnn_lstm \
    --config experiments/cnn_lstm_g1.yaml \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G2
# Expect: 5 seeds × ~10 min = ~50 min CPU.
```

**G2 gate check:**
- std across 5 seeds ≤ 0.10. If std > 0.10 → STOP. Investigate: head_dropout too low? batch_size too small for stable BN stats?
- Mean CNN-LSTM macro-F1 reported vs LSTM 0.599. If CNN-LSTM < LSTM → flag in report (not a stop gate).

**Phase 2 gate:**
```bash
.venv/bin/python -m pytest                      # must remain ≥ 284 green
```

---

### Phase 3 — Rigor gaps G4, G6, G7, G9, G10 (2 hr)

All use `experiments/cnn_lstm_g1.yaml` (populated with G1 best HP).

#### G4 — Threshold tuning (val-only)

```bash
.venv/bin/python scripts/rigor/threshold_sweep.py \
    --config experiments/cnn_lstm_g1.yaml \
    --model-dir checkpoints/cnn_lstm \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G4
```
Output: `reports/rigor/2026-05-13/cnn_lstm_G4/threshold_summary.md`
Expected: optimal threshold t* for bull/bear classes. Apply to test evaluation in G10 CI.

#### G6 — Asymmetry analysis (from G2 seed results)

```bash
.venv/bin/python scripts/rigor/asymmetry_analysis.py \
    --seed-dir reports/rigor/2026-05-13/cnn_lstm_G2 \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G6
```
Output: per-class F1 variance across seeds. Bull vs bear asymmetry flag.

#### G7 — Window sweep CNN-LSTM-specific

```bash
.venv/bin/python scripts/rigor/window_sweep.py \
    --config experiments/cnn_lstm_g1.yaml \
    --windows 30 60 90 120 \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G7
```
Key question: does larger W improve CNN-LSTM (as it did for LSTM: W=90 → 0.649 vs 0.618)? Hypothesis: with Conv1d(k=3), CNN-LSTM needs less context → W=60 should hold or W=90 adds marginal gain. If W=120 wins, revisit G1 with W=120 base.

#### G9 — Regularisation ablation

```bash
# Dropout off only
.venv/bin/python scripts/rigor/multiseed_run.py \
    --model cnn_lstm \
    --config experiments/cnn_lstm_g1.yaml \
    --set "train.seeds=[42]" \
    --no-dropout \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G9

# L2 off only
.venv/bin/python scripts/rigor/multiseed_run.py \
    --model cnn_lstm \
    --config experiments/cnn_lstm_g1.yaml \
    --set "train.seeds=[42]" \
    --no-l2 \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G9

# Both off
.venv/bin/python scripts/rigor/multiseed_run.py \
    --model cnn_lstm \
    --config experiments/cnn_lstm_g1.yaml \
    --set "train.seeds=[42]" \
    --no-dropout --no-l2 \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G9
```
Expected: both-off drops F1 most. If dropout-off alone barely changes F1, increase head_dropout range in next model iteration.

#### G10 — Bootstrap CI of mean across seeds

```bash
.venv/bin/python scripts/rigor/bootstrap_ci_multiseed.py \
    --seed-dir reports/rigor/2026-05-13/cnn_lstm_G2 \
    --output-dir reports/rigor/2026-05-13/cnn_lstm_G10
```
Output: mean ± std + 95% CI (block bootstrap, n_iter=1000, block_size=60). Final reportable number for `docs/models-status.md`.

**Phase 3 gate:**
```bash
.venv/bin/python -m pytest                      # must remain ≥ 284 green
```

---

### Phase 4 — Inspect adapter + docs sync (0.5 hr)

#### New files

| File | Action | Notes |
|------|--------|-------|
| `src/inspect/adapters/cnn_lstm_adapter.py` | CREATE | `CNNLSTMAdapter(ModelAdapter)` — mirrors `LSTMAdapter` exactly. Loads `.pt` via `FVGCNNLSTMClassifier` from meta.json kwargs. |

#### Modified files

| File | Change |
|------|--------|
| `src/inspect/adapters/__init__.py` | Register `CNNLSTMAdapter` in adapter registry under `"cnn_lstm"` key |
| `docs/models-status.md` | Add CNN-LSTM row to baseline comparison table. Add brief HP discussion section. |

**Post-adapter smoke:**
```bash
.venv/bin/python scripts/inspect_models.py \
    --models cnn_lstm \
    --lookahead-bars 20
```

**Phase 4 gate:**
```bash
.venv/bin/python -m pytest                      # final count ≥ 284 green
```

---

## Test strategy per phase

### Phase 1: `tests/models/test_cnn_lstm.py`

```
test_forward_shape           — input (2, 60, 5) → output (2, 3); no error
test_gradient_flows          — loss.backward() without NaN; all params.grad is not None
test_num_params_default      — ~29k params at F=32, k=3, H=64, n_layers=2
test_no_data_leakage_shape   — batch_size=1 works (no BN issue with single sample check)
```

4 tests. All in `tests/models/test_cnn_lstm.py`. Spawned via `@nb-test` after Phase 1.

### Phase 2: no new test files. `pytest` regression check only.

### Phase 3: no new test files. `pytest` regression check only.

### Phase 4: no new test files unless adapter registry has a test suite (check `tests/inspect/`). If so, add `test_cnn_lstm_adapter_loads_checkpoint` there.

---

## Decision gates summary

| Gate | Condition | Pass | Fail action |
|------|-----------|------|-------------|
| Phase 1 end | pytest ≥ 284 green + 2-epoch smoke no error | Proceed to G1 | Fix before proceeding |
| G1 end | Best val macro-F1 ≥ 0.40 | Populate g1.yaml, run G2 | STOP — debug search space + class weights path |
| G2 end | std ≤ 0.10 across 5 seeds | Proceed to G4–G10 | STOP — investigate BN instability or search space collapse |
| G2 mean | CNN-LSTM mean > LSTM 0.599 | Positive note | Flag in docs — not a stop |
| Phase 4 end | pytest ≥ 284 green + inspect_models smoke | Ship | Fix before docs update |

---

## Wall-clock estimates

| Phase | Compute | Agent time | Total |
|-------|---------|-----------|-------|
| Phase 1: model + config + smoke | ~5 min (2 epochs) | 30 min build | ~35 min |
| Phase 2: G1 Optuna (50 trials) | ~60–90 min CPU | 10 min orchestration | ~100 min |
| Phase 2: G2 multi-seed (5 seeds) | ~50 min CPU | 10 min orchestration | ~60 min |
| Phase 3: G4 + G6 + G7 + G9 + G10 | ~90 min CPU | 20 min orchestration | ~110 min |
| Phase 4: adapter + docs | ~2 min | 20 min build | ~22 min |
| **Total** | **~4.5 hr CPU** | **~1.5 hr agent** | **~6 hr wall** |

G1 is the longest single-threaded bottleneck. Can run in background while agent drafts adapter skeleton.

---

## Subagent ownership

| Agent | Scope | Model |
|-------|-------|-------|
| `@nb-build` (orchestrator) | Phases 1–4 in sequence. Reads this plan. Spawns nb-test + nb-review. | sonnet |
| `@nb-test` (Phase 1) | Run `pytest tests/models/test_cnn_lstm.py -v` + `pytest` full suite. Write test log. | haiku |
| `@nb-test` (Phase 4) | Final full `pytest` run. Write test log. | haiku |
| `@nb-review` (Phase 4 end) | Check: class shape correctness, no lookahead, schema union complete, adapter registry, docs table accuracy. | sonnet |

---

## Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | Conv over-smooths 3-candle FVG pattern (n_conv_layers=2 hurts) | Medium | Medium | G1 searches `n_conv_layers ∈ {1,2}`. Compare 1-layer vs 2-layer in importances plot. If 1-layer wins, update g1.yaml. |
| R2 | Overfit on eff-n=117: val F1 >> test F1 | Medium-High | High | head_dropout ≥ 0.3, weight_decay ≥ 1e-5, patience=15, batch=16. Monitor in G2 summary. |
| R3 | BatchNorm instability (NaN loss in first 5 epochs) | Low | Medium | Monitor epoch 1–5 val loss. If NaN: switch to `GroupNorm(num_groups=4, num_channels=conv_filters)` in `cnn_lstm.py`. Add note in Phase 1 smoke check. |
| R4 | `class_weights_fvg_valid.json` path vs `class_weights.json` | Low | High | `tune_cnn_lstm.py` must load `class_weights_fvg_valid.json` (labeller=fvg_valid). LSTM rigor confirmed this filename. Double-check in script before G1 run. |
| R5 | `padding=kernel_size//2` off-by-one for even k | None | None | k ∈ {3,5,7} are all odd in search space. Padding = 1, 2, 3 — output length = input length exactly. Only a risk if even k added later. |
| R6 | `multiseed_run.py --model cnn_lstm` argparse choices not updated | Low | Low | Explicitly listed in Phase 1 modified files. Easy fix. |
| R7 | CNN-LSTM mean F1 < LSTM 0.599 | Low-Medium | Medium | Expected +3–5% gain. If fails: report honestly. Check eff-n regime, consider pure CNN ablation as Phase 5. Not a sprint-stopper. |
| R8 | `use_pool=True` with stride=1 produces odd T' via MaxPool1d padding | Low | Low | MaxPool1d(kernel_size=2, stride=1, padding=1) on L=60: output = 61. Use `padding=0` instead for stride=1 (output = 59). Safer: only search `use_pool=False` in G1; add pool ablation post-G2 if warranted. |
| R9 | Optuna < 0.40 gate triggers on class_weights misconfiguration | Medium | High | Before G1: run single trial manually with verbose logging. Confirm class_weights loaded, confirm labeller=fvg_valid produces non-trivial val split. |
| R10 | `train.py` dispatcher missing `"cnn_lstm"` branch → KeyError at runtime | Low | Low | Phase 1 smoke test at max_epochs=2 catches this immediately. |

---

## CNNLSTMModelConfig schema (exact — for schema.py)

```python
class CNNLSTMModelConfig(BaseModel):
    arch: Literal["cnn_lstm"] = "cnn_lstm"
    n_conv_layers: int = 2
    conv_filters: int = 32
    kernel_size: int = 3
    use_pool: bool = False
    pool_type: str = "max"       # "max" | "avg" — only used if use_pool=True
    lstm_hidden: int = 64
    lstm_layers: int = 1
    dropout: float = 0.318       # between LSTM layers (active only if lstm_layers > 1)
    head_dropout: float = 0.526
```

ModelConfig union update:
```python
ModelConfig = Annotated[
    LSTMModelConfig | XGBModelConfig | CNNLSTMModelConfig,
    Field(discriminator="arch"),
]
```

---

## experiments/cnn_lstm_base.yaml (exact — for Phase 1)

```yaml
# cnn_lstm_base.yaml — baseline config for CNN-LSTM smoke + G1 HP search starting point.
# Defaults from research doc. HP will be overridden by Optuna in tune_cnn_lstm.py.
name: cnn_lstm_base_validfvg
data:
  labeller: fvg_valid
  window_size: 60
model:
  arch: cnn_lstm
  n_conv_layers: 2
  conv_filters: 32
  kernel_size: 3
  use_pool: false
  pool_type: "max"
  lstm_hidden: 64
  lstm_layers: 1
  dropout: 0.318
  head_dropout: 0.526
train:
  seeds: [42]
  batch_size: 16
  lr: 5.3e-4
  weight_decay: 3.92e-5
  optimizer: adam
  scheduler: none
  max_epochs: 100
  patience: 15
  loss: weighted_ce
  device: cpu
```

---

## experiments/cnn_lstm_g1.yaml (placeholder — populated post-G1)

```yaml
# cnn_lstm_g1.yaml — G1 Optuna best HP for CNN-LSTM on ValidFVG target.
# HP source: reports/rigor/2026-05-13/cnn_lstm_G1/best_hp_cnn_lstm.json
# PLACEHOLDER — populated by tune_cnn_lstm.py after G1 completes.
name: cnn_lstm_g1_validfvg
data:
  labeller: fvg_valid
  window_size: 60
model:
  arch: cnn_lstm
  n_conv_layers: 2          # TBD from G1
  conv_filters: 32          # TBD from G1
  kernel_size: 3            # TBD from G1
  use_pool: false           # TBD from G1
  pool_type: "max"
  lstm_hidden: 64           # TBD from G1
  lstm_layers: 1            # TBD from G1
  dropout: 0.318            # TBD from G1
  head_dropout: 0.526       # TBD from G1
train:
  seeds: [0, 17, 42, 123, 2024]
  batch_size: 16            # TBD from G1
  lr: 5.3e-4                # TBD from G1
  weight_decay: 3.92e-5     # TBD from G1
  optimizer: adam
  scheduler: none
  max_epochs: 100
  patience: 15
  loss: weighted_ce
  device: cpu
```

---

## Optuna search space — tune_cnn_lstm.py (exact)

```python
# CNNLSTMObjective.suggest_params(trial):
n_conv_layers = trial.suggest_int("n_conv_layers", 1, 3)         # {1, 2, 3}
conv_filters  = trial.suggest_categorical("conv_filters", [16, 32, 64])
kernel_size   = trial.suggest_categorical("kernel_size", [3, 5, 7])
use_pool      = trial.suggest_categorical("use_pool", [False, True])
lstm_hidden   = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
lstm_layers   = trial.suggest_int("lstm_layers", 1, 2)
dropout       = trial.suggest_float("dropout", 0.1, 0.5)          # LSTM inter-layer
head_dropout  = trial.suggest_float("head_dropout", 0.3, 0.7)
lr            = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
batch_size    = trial.suggest_categorical("batch_size", [16, 32])
```

11-dim total (not 7 — batch_size, lr, weight_decay, dropout, head_dropout are continuous/categorical beyond the 7 structural dims). 50 trials with TPE is adequate coverage.

---

## Output structure

```
checkpoints/
  cnn_lstm/
    cnn_lstm_seed0.pt
    cnn_lstm_seed17.pt
    cnn_lstm_seed42.pt
    cnn_lstm_seed123.pt
    cnn_lstm_seed2024.pt
    cnn_lstm_seed{N}.meta.json    (sidecar per checkpoint)
    optuna_2026-05-13.db

reports/rigor/2026-05-13/
  cnn_lstm_G1/
    best_hp_cnn_lstm.json
    cnn_lstm_optuna_history.html
    cnn_lstm_optuna_importances.html
  cnn_lstm_G2/
    multiseed_summary_cnn_lstm_weighted_ce.md
    cnn_lstm_seed{N}_preds.npz
  cnn_lstm_G4/
    threshold_summary.md
  cnn_lstm_G6/
    asymmetry_summary.md
  cnn_lstm_G7/
    window_sweep_summary.md
  cnn_lstm_G9/
    ablation_summary.md
  cnn_lstm_G10/
    bootstrap_ci_summary.md

.nb-suite/
  build/13-May-26/cnn-lstm-build.md    (nb-build writes this)
```

---

## docs/models-status.md update target

Add CNN-LSTM row to comparison table after LSTM row. Format mirrors existing table:

| Model | Val macro-F1 | Test macro-F1 | Seeds | Notes |
|-------|-------------|--------------|-------|-------|
| XGBoost | 0.721 ± 0.001 | TBD | 5 | Hand-crafted features, gap geometry prior |
| LSTM | 0.599 ± 0.025 | TBD | 5 | Hidden=128, 1L, W=60 |
| CNN-LSTM | **TBD ± TBD** | TBD | 5 | F=?, k=?, H=?, W=60, G1 trial ? |

Add subsection: "CNN-LSTM HP Discussion" — 3–5 sentences: top-3 important params from Optuna importances, whether pool helped, n_conv_layers result.

---

## What @nb-build receives

This plan. Read it top-to-bottom. Execute Phase 1 → gate → Phase 2 → gate → Phase 3 → Phase 4. Do not skip gates. Spawn `@nb-test` after Phase 1 and Phase 4. Spawn `@nb-review` after Phase 4. Write build log to `.nb-suite/build/13-May-26/cnn-lstm-build.md`.
