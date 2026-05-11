# Phase 4 Step 1 — LSTM Baseline Plan

> **Plan type:** Feature implementation plan — Stage 1 of Phase 4 Modelling.
> **Work type:** New build (2-layer stacked LSTM on SPY H1 ValidFVG windows).
> **Recommended mode:** /nb:build standard (single branch). All phases sequential.
> **Test strategy:** TDD (pytest) for model shape + device + causality gate. ML evaluation on held-out test split only.
> **Date:** 11 May 2026
> **Deadline constraint:** Status Update 1 = 17 May (6 days). LSTM baseline = ~1 day build.

---

## Context and locked decisions

**Floor to beat (from `.nb-suite/test-logs/11-May-26/xgboost-baseline.md`):**
| Metric | XGBoost |
|--------|---------|
| Test macro-F1 | 0.6084 |
| Bull F1 | 0.5519 |
| Bear F1 | 0.5183 |
| Naive baseline macro-F1 | 0.2846 |

LSTM is not expected to beat XGBoost on this dataset size (~117 effective independent windows). If it matches XGBoost, the comparison is a valuable finding. If it underperforms, that too is a valid academic result per master plan pre-mortem item 3. No "tune until LSTM wins" — train clean, report honestly, move on.

**Dataset already on disk:**
- `data/processed/spy_h1_train.parquet` — 7,056 rows (2018-01-02 – 2021-12-31)
- `data/processed/spy_h1_val.parquet` — 1,757 rows (2022-01-03 – 2022-12-30)
- `data/processed/spy_h1_test.parquet` — 3,514 rows (2023-01-03 – 2024-12-31)
- `data/processed/class_weights.json` — `{0: 0.217, 1: 1.097, 2: 1.686}`

**Window datasets:**
- `SMCWindowDataset` from `src/data/window.py` emits `(Tensor[60, 5], int)` per `__getitem__`.
- Normalisation already applied inside `build_windows()` via `normalise_window()` per-window.
- Shape: `(B, 60, 5)` after DataLoader collation — **no adapter needed**.
- Train stride = 1 (6,997 windows). Val/test stride = window_size = 60 (non-overlapping for early-stop signal; see note below on stride decision).

**Stride decision for evaluation:**
- XGBoost used stride=1 for all splits → 3,455 test windows → 882 positives → reliable F1.
- LSTM uses the same pipeline: train stride=1, val stride=1, test stride=1. This matches XGBoost's evaluation distribution exactly, making the comparison fair.
- The XGBoost plan originally used stride=60 for val/test but resolved to stride=1 (see the "Window count calculation" section of the XGBoost plan). LSTM follows the same resolution.
- **Concrete window counts (stride=1):**
  - Train: 6,997 windows, ~1,718 positive (24.6%)
  - Val: 1,698 windows, ~458 positive (27.0%)
  - Test: 3,455 windows, ~882 positive (25.5%)

**Temporal split (locked, never touch):**
- Train: 2018-01-02 – 2021-12-31
- Val: 2022-01-03 – 2022-12-30
- Test: 2023-01-03 – 2024-12-31

**Class weights (from train split, already persisted):**
- `[0.217, 1.097, 1.686]` for classes `[none, bull, bear]`
- Must be loaded from `data/processed/class_weights.json`, not recomputed. Tensor sent to device.

**Output head:** 3 logits (ternary: 0=none, 1=bull, 2=bear). Not binary.

**Primary metric:** Macro-F1. Per-class F1 (bull, bear) reported separately. Accuracy not the headline.

**Seed:** 42. Multi-seed (5 seeds) documented as future work for final evaluation but NOT required for this baseline run.

**Device:** Apple MPS (M4 Pro 48 GB). With MPS gotchas applied per CLAUDE.md:
- Use `nn.LSTM` (sequence-batched), NOT `nn.LSTMCell` loops.
- Do NOT enable `torch.use_deterministic_algorithms(True)` on MPS — 8× slowdown.
- No `nn.MultiheadAttention` in this model, so MPS NaN workaround not needed.
- If MPS produces NaN in loss: fall back to CPU and document.

---

## Architecture specification

### Model: `FVGLSTMClassifier`

**Input:** `(B, 60, 5)` float32 tensor — batch of 60-candle windows, 5 OHLCV channels.

**Architecture (per CLAUDE.md + R6):**

```
Input: (B, 60, 5)
  │
  ▼
nn.LSTM(
    input_size=5,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,         # applied between LSTM layers (only active if num_layers > 1)
    batch_first=True,    # expects (B, T, C)
)
  │
  ├── output: (B, 60, 64)  — all hidden states (not used)
  └── h_n: (2, B, 64)      — hidden states for both layers
         │
         └── h_n[-1]: (B, 64) — last layer's final hidden state
              │
              ▼
         nn.Dropout(0.5)   — dropout on classifier head
              │
              ▼
         nn.Linear(64, 3)  — 3 logits
              │
              ▼
         Output: (B, 3) raw logits
```

**Why last hidden state:** label is at position 59 of the window (the final candle). Taking `h_n[-1]` (last layer, last time-step hidden state) is the natural readout for per-window classification. This is equivalent to taking `output[:, -1, :]` from the LSTM output but more efficient — no intermediate sequence storage required.

**Why unidirectional:** bidirectional LSTM at position 59 would "see" positions 60+ (beyond window boundary). Even within the window, bi-LSTM can leak information from the backward pass at earlier positions toward the label position. Keep unidirectional — per R6 explicit recommendation.

**Hyperparameters (locked from R6 + R7 — do NOT change during baseline build):**
| Hyperparameter | Value | Source |
|---|---|---|
| `hidden_size` | 64 | CLAUDE.md, R6 (Fischer & Krauss baseline, ~50k params stays below danger zone) |
| `num_layers` | 2 | CLAUDE.md, R6 (2-layer stacked LSTM) |
| `dropout` (LSTM inter-layer) | 0.3 | R6 |
| `dropout` (classifier head) | 0.5 | R6 |
| `batch_first` | True | Matches DataLoader collation shape (B, T, C) |
| `bidirectional` | False | R6, CLAUDE.md — unsafe at label-position-59 |

**Total parameter count (estimate):** ~66k parameters. Well below the ~200k "danger zone" for 7k overlapping windows.

---

## Training protocol (from R7 — locked)

| Parameter | Value | Source |
|---|---|---|
| Loss | `nn.CrossEntropyLoss(weight=class_weights)` | R7 |
| Optimizer | `AdamW(lr=3e-3, weight_decay=1e-2)` | R7 — note weight_decay=1e-2 not 1e-4 (small-dataset scaling) |
| LR schedule | `OneCycleLR(max_lr=3e-3)` | R7 — LSTM max_lr |
| Batch size | 32 | R7 |
| Max epochs | 100 | R7 |
| Early stopping signal | Smoothed val macro-F1, EMA alpha=0.3, patience=15 | R7 |
| Seed | 42 | Locked |

**Early stopping detail:**
- Monitor: `smoothed_val_macro_f1 = ema_alpha * val_macro_f1 + (1 - ema_alpha) * prev_smoothed`
- Patience: 15 epochs (R7 recommends 15 for noisy val sets)
- Min delta: 1e-4
- Save checkpoint at each improvement. Load best before test evaluation.
- Do NOT use val loss as primary stopping signal — F1 is the target metric (R7).

**OneCycleLR setup:**
```python
total_steps = max_epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, total_steps=total_steps
)
```
Scheduler steps once per batch (not per epoch). This is required for OneCycleLR.

---

## Data loading strategy

**Reuse existing pipeline directly — no adapter needed.**

```python
from src.data.pipeline import build_pipeline
from torch.utils.data import DataLoader

train_ds, val_ds, test_ds, class_weights = build_pipeline(
    labeller_name="valid_fvg",
    window_size=60,
)
```

**Important:** `build_pipeline()` uses `stride=1` for train and `stride=window_size=60` for val/test. To match XGBoost's stride=1 evaluation, we need to override val/test stride. Options:

A. Reload val/test directly from parquet with `SMCWindowDataset` at stride=1.
B. Modify `build_pipeline()` to accept stride arguments.

**Decision: Option A — load val/test separately with stride=1.** Does not touch the existing pipeline:

```python
import pandas as pd
from src.data.labels import LABELLERS
from src.data.window import SMCWindowDataset

labeller = LABELLERS["valid_fvg"]()

train_df = pd.read_parquet("data/processed/spy_h1_train.parquet")
val_df   = pd.read_parquet("data/processed/spy_h1_val.parquet")
test_df  = pd.read_parquet("data/processed/spy_h1_test.parquet")

train_ds = SMCWindowDataset(train_df, labeller, stride=1,  window_size=60, drop_cross_session_windows=False)
val_ds   = SMCWindowDataset(val_df,   labeller, stride=1,  window_size=60, drop_cross_session_windows=False)
test_ds  = SMCWindowDataset(test_df,  labeller, stride=1,  window_size=60, drop_cross_session_windows=False)
```

**Class weights:** Load from `data/processed/class_weights.json`, convert to tensor, send to device:
```python
import json, torch
with open("data/processed/class_weights.json") as f:
    cw = json.load(f)
class_weights = torch.tensor([cw["0"], cw["1"], cw["2"]], dtype=torch.float32).to(device)
```

**DataLoader setup (per R7 reproducibility):**
```python
def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=0, worker_init_fn=seed_worker, generator=g)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)
```

`shuffle=True` on train is required (temporal order is broken by windowing; each window is a complete independent sample from the model's perspective). DO NOT shuffle val/test.

---

## File layout

```
src/models/
  lstm.py                           — FVGLSTMClassifier model class

src/training/
  __init__.py                       — empty
  loss.py                           — WeightedCE wrapper + FocalLoss alternative
  early_stop.py                     — EarlyStop class with EMA smoothing
  train_utils.py                    — set_seed(), log_run_metadata(), eval_epoch()

scripts/
  train_lstm.py                     — end-to-end: load data → train → eval → save → log

tests/models/
  test_lstm.py                      — unit tests (see test strategy section)

checkpoints/
  lstm/
    lstm_seed42.pt                  — best checkpoint (gitignored)
    lstm_seed42.meta.json           — run metadata

.nb-suite/test-logs/11-May-26/
  lstm-baseline.md                  — evaluation log (same format as xgboost-baseline.md)

.nb-suite/build/11-May-26/
  phase4-lstm-baseline.md           — build log
```

---

## Component breakdown

### Phase 1: Model class (`src/models/lstm.py`)

**Model:** haiku

**Class: `FVGLSTMClassifier(nn.Module)`**

```python
class FVGLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        head_dropout: float = 0.5,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) = (batch, 60, 5)
        returns: (B, num_classes) raw logits — no softmax
        """
        # _, (h_n, _) = self.lstm(x)  → h_n shape (num_layers, B, hidden_size)
        # Take h_n[-1]: (B, hidden_size)
        # Apply head_dropout → linear → (B, 3)
```

**Module structure:**
- `self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)`
- `self.head_dropout = nn.Dropout(head_dropout)`
- `self.classifier = nn.Linear(hidden_size, num_classes)`

**Acceptance criteria:**
- `model(torch.randn(8, 60, 5))` returns shape `(8, 3)` — no error.
- Output is finite: `torch.isfinite(logits).all()`.
- Works on CPU, MPS (if available), and when device=cpu is forced.
- `model.parameters()` counts ≤ 100k total parameters.
- `nn.LSTM` used, NOT `nn.LSTMCell`. Confirm by checking `isinstance(model.lstm, nn.LSTM)`.

**Effort:** 1–2 hours.

---

### Phase 2: Training infrastructure (`src/training/`)

**Model:** haiku

**Files:**
- `src/training/__init__.py` — empty
- `src/training/loss.py` — WeightedCE + FocalLoss
- `src/training/early_stop.py` — EarlyStop with EMA
- `src/training/train_utils.py` — set_seed, log metadata, eval epoch

**`src/training/loss.py`:**

```python
class WeightedCE:
    """Wrapper: nn.CrossEntropyLoss(weight=class_weights). Default for all architectures."""
    def __init__(self, class_weights: torch.Tensor) -> None:
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class FocalLoss:
    """Alternative for LSTM ablation only if minority-F1 is stuck near zero."""
    def __init__(self, class_weights: torch.Tensor, gamma: float = 1.0) -> None: ...
    def __call__(self, logits, targets) -> torch.Tensor: ...
```

**`src/training/early_stop.py`:**

```python
class EarlyStop:
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        ema_alpha: float = 0.3,
        mode: str = "max",
    ) -> None: ...

    def update(self, metric: float) -> bool:
        """
        Returns True if training should stop.
        Updates smoothed metric via EMA. Tracks best smoothed value.
        """
        ...

    @property
    def smoothed(self) -> float: ...

    @property
    def best(self) -> float: ...
```

**`src/training/train_utils.py`:**

```python
def set_seed(seed: int) -> None:
    """Set random, numpy, torch seeds. Does NOT enable deterministic mode on MPS."""
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True) — NOT called. MPS 8x slowdown per R7.


def log_run_metadata(seed: int, device: torch.device) -> dict:
    """Returns dict with git_sha, torch_version, python_version, device, timestamp."""
    ...


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: WeightedCE,
    device: torch.device,
) -> dict:
    """
    Runs one evaluation pass.
    Returns: {
        "loss": float,
        "macro_f1": float,
        "per_class_f1": list[float],   # [none_f1, bull_f1, bear_f1]
        "y_true": np.ndarray,
        "y_pred": np.ndarray,
    }
    Uses sklearn.metrics.f1_score with average="macro" and per-class.
    Handles undefined per-class F1 (zero_division=0.0).
    """
    ...
```

**Acceptance criteria:**
- `EarlyStop(patience=2).update(0.5)` returns False. After 3 calls with same value, returns True.
- `set_seed(42)` runs without error on CPU. On MPS-available machine, does not crash.
- `eval_epoch` returns dict with all expected keys. `macro_f1` in [0.0, 1.0].
- `WeightedCE(class_weights)(logits, targets)` returns scalar tensor, finite.

**Effort:** 2–3 hours.

---

### Phase 3: Training script (`scripts/train_lstm.py`)

**Model:** sonnet

**Usage:**
```bash
python scripts/train_lstm.py [--seed 42] [--device auto] [--output-dir checkpoints/lstm]
```

**Steps (sequential):**

```
1.  set_seed(42)
2.  log_run_metadata → metadata dict
3.  device = "mps" if mps available else "cpu"
4.  Load train/val/test parquets → SMCWindowDataset (stride=1 for all)
5.  Load class_weights from data/processed/class_weights.json → Tensor.to(device)
6.  Build DataLoaders (batch=32 train, batch=256 val/test, num_workers=0)
7.  Instantiate FVGLSTMClassifier → .to(device)
8.  criterion = WeightedCE(class_weights)
9.  optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
10. total_steps = max_epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=total_steps)
11. early_stop = EarlyStop(patience=15, ema_alpha=0.3)
12. TRAINING LOOP (max_epochs=100):
      a. train one epoch (model.train(), iterate batches, loss.backward(), clip_grad_norm_(1.0))
      b. scheduler.step() after each batch
      c. eval_epoch on val → val metrics
      d. if early_stop.update(val_macro_f1): save checkpoint, break
      e. if val_macro_f1 improved (smoothed): save checkpoint
      f. log epoch, train_loss, val_loss, val_macro_f1, smoothed_val_f1, per_class_f1
      g. write log row to logs/lstm_seed42.csv
13. Load best checkpoint
14. eval_epoch on TEST set → test metrics
15. Print classification_report + confusion_matrix
16. Save checkpoint: checkpoints/lstm/lstm_seed42.pt (model state_dict + metadata)
17. Write evaluation log: .nb-suite/test-logs/11-May-26/lstm-baseline.md
```

**Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — prevents gradient explosion which is a known risk with LSTM + small data.

**MPS NaN guard:**
```python
if torch.isnan(loss):
    print("NaN loss detected on MPS — falling back to CPU")
    device = torch.device("cpu")
    model = model.to(device)
    class_weights = class_weights.to(device)
    # re-run from checkpoint or restart from scratch on CPU
    break  # exit training loop, flag in log
```

**Epoch log format (CSV):**
```
epoch,train_loss,val_loss,val_macro_f1,smoothed_val_macro_f1,val_bull_f1,val_bear_f1,lr
```

**Acceptance criteria:**
- Script runs end-to-end without error: `python scripts/train_lstm.py`
- Training completes (early stopping or max 100 epochs).
- Best checkpoint saved at `checkpoints/lstm/lstm_seed42.pt`.
- Evaluation log written to `.nb-suite/test-logs/11-May-26/lstm-baseline.md`.
- Test macro-F1 reported and compared against XGBoost floor (0.6084).
- No NaN in final test metrics (NaN = training failure → document and escalate).

**Effort:** 2–3 hours (script) + training run (~5–15 min on MPS) + eval writeup (~30 min).

---

### Phase 4: Unit tests (`tests/models/test_lstm.py`)

**Model:** haiku

**Test file:** `tests/models/test_lstm.py`

**Tests:**

```python
def test_forward_shape_cpu():
    """Forward pass on CPU returns (B, 3)."""
    model = FVGLSTMClassifier()
    x = torch.randn(8, 60, 5)
    logits = model(x)
    assert logits.shape == (8, 3)


def test_forward_finite():
    """All logits are finite."""
    model = FVGLSTMClassifier()
    x = torch.randn(8, 60, 5)
    logits = model(x)
    assert torch.isfinite(logits).all()


def test_forward_single_sample():
    """Works with batch size 1."""
    model = FVGLSTMClassifier()
    x = torch.randn(1, 60, 5)
    logits = model(x)
    assert logits.shape == (1, 3)


def test_uses_nn_lstm_not_cell():
    """Confirms nn.LSTM is used, not nn.LSTMCell (slower on MPS per CLAUDE.md)."""
    import torch.nn as nn
    model = FVGLSTMClassifier()
    assert isinstance(model.lstm, nn.LSTM), "Must use nn.LSTM not nn.LSTMCell"


def test_parameter_count_under_limit():
    """Total params < 100k (safety check against over-engineering)."""
    model = FVGLSTMClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, f"Parameter count {n_params} exceeds 100k"


def test_output_not_softmax():
    """Output is raw logits, not softmax probabilities."""
    model = FVGLSTMClassifier()
    x = torch.randn(4, 60, 5)
    logits = model(x)
    row_sums = logits.softmax(dim=-1).sum(dim=-1)
    # raw logits: row sums of softmax should be ~1.0 (that's expected)
    # but raw logit row sums should NOT be ~1.0
    raw_sums = logits.sum(dim=-1)
    # This is a soft check — logits can accidentally sum to 1 but it's unlikely for all rows
    # Instead check that not all outputs are in [0, 1] range (probability range)
    assert not (logits >= 0).all() or not (logits <= 1).all(), \
        "Outputs look like probabilities — use raw logits"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
def test_forward_mps():
    """Forward pass on MPS device returns finite output."""
    device = torch.device("mps")
    model = FVGLSTMClassifier().to(device)
    x = torch.randn(8, 60, 5, device=device)
    logits = model(x)
    assert logits.shape == (8, 3)
    assert torch.isfinite(logits).all()


def test_early_stop_triggers():
    """EarlyStop triggers after patience epochs with no improvement."""
    es = EarlyStop(patience=3, min_delta=1e-4, ema_alpha=0.3)
    for _ in range(4):
        stopped = es.update(0.5)
    assert stopped, "EarlyStop should trigger after patience exhausted"


def test_early_stop_resets_on_improvement():
    """EarlyStop does not trigger when metric keeps improving."""
    es = EarlyStop(patience=3, min_delta=1e-4, ema_alpha=0.3)
    for i in range(10):
        stopped = es.update(0.5 + i * 0.01)
    assert not stopped
```

**Effort:** 1–2 hours.

---

## Evaluation output format

`.nb-suite/test-logs/11-May-26/lstm-baseline.md` must contain:

```markdown
# LSTM Baseline Evaluation — Phase 4 Step 1

**Date:** 2026-05-11
**Git SHA:** <hash>
**Python:** <version>
**PyTorch:** <version>
**Device:** mps / cpu
**Seed:** 42

## Model architecture
- hidden_size: 64, num_layers: 2, dropout: 0.3, head_dropout: 0.5
- Total parameters: <N>

## Dataset summary
| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|--------------|
| Train | 6997 | ~1718 | ~24.6% |
| Val   | 1698 | ~458  | ~27.0% |
| Test  | 3455 | ~882  | ~25.5% |

## Hyperparameters
<table>

## Training
- Stopped at epoch: <N> (early stopping or max_epochs)
- Best smoothed val macro-F1: <value> at epoch <N>
- Best checkpoint: checkpoints/lstm/lstm_seed42.pt

## Training curve (first 5 / last 5 epochs)
<table: epoch, train_loss, val_macro_f1, smoothed_f1>

## Validation results (best checkpoint)
<classification_report>
<3×3 confusion matrix>

## Test results (primary)
<classification_report>
<3×3 confusion matrix>

## Comparison to XGBoost floor and naive baseline
| Metric | Naive | XGBoost | LSTM | LSTM vs XGBoost |
|--------|-------|---------|------|-----------------|
| Macro-F1 | 0.2846 | 0.6084 | <val> | <delta> |
| Bull F1  | 0.0000 | 0.5519 | <val> | <delta> |
| Bear F1  | 0.0000 | 0.5183 | <val> | <delta> |

## Interpretation
<honest analysis: did LSTM beat XGBoost? if not, why (expected for this data scale)?>

## Phase 4 handoff to CNN-LSTM
- LSTM test macro-F1: <value>
- Model saved at: checkpoints/lstm/lstm_seed42.pt
```

---

## Sequential build order

### Step 1: Model class
**Deliverables:** `src/models/lstm.py`
**Acceptance gate:** `tests/models/test_lstm.py` passes (forward shape, finite, nn.LSTM check, MPS check)
**Model:** haiku
**Effort:** 1–2 hours

### Step 2: Training infrastructure
**Deliverables:** `src/training/__init__.py`, `src/training/loss.py`, `src/training/early_stop.py`, `src/training/train_utils.py`
**Acceptance gate:** Unit tests for EarlyStop, WeightedCE, set_seed pass
**Model:** haiku
**Effort:** 2–3 hours

### Step 3: Training script + full run
**Deliverables:** `scripts/train_lstm.py`, `checkpoints/lstm/lstm_seed42.pt`, `logs/lstm_seed42.csv`
**Acceptance gate:** Script runs end-to-end, checkpoint saved, training curve logged
**Model:** sonnet
**Effort:** 2–3 hours (write) + ~15 min (training run)

### Step 4: Evaluation log
**Deliverables:** `.nb-suite/test-logs/11-May-26/lstm-baseline.md`
**Acceptance gate:** Log contains all fields from format above. XGBoost comparison table populated.
**Model:** sonnet
**Effort:** 30 min

---

## Test strategy

### Unit tests (TDD — write before script)

| Test file | Tests |
|-----------|-------|
| `tests/models/test_lstm.py` | Forward shape (B, 3), finite logits, batch=1, uses nn.LSTM not LSTMCell, param count <100k, MPS device (if available), EarlyStop triggers/resets |

### ML evaluation (run from script, not pytest)

| Check | Where |
|-------|-------|
| Test macro-F1 vs XGBoost floor (0.6084) | `lstm-baseline.md` comparison table |
| Per-class F1 (bull, bear) | Same |
| Confusion matrix 3×3 | Same |
| Training curve: loss decreasing monotonically? | `logs/lstm_seed42.csv` |
| Val macro-F1 > naive baseline (0.2846) | Assert in script; log warning if not |
| NaN loss check | Script: fall back to CPU if NaN on MPS |

### Mandatory sanity checks (in training script)

1. **Pre-training:** assert `X.shape == (batch, 60, 5)` for first batch. If shape wrong, data loading is broken.
2. **Pre-training:** assert class_weights tensor is on correct device and finite.
3. **Overfit check (optional but valuable):** train for 5 epochs on a 200-sample subset of train. If loss does not decrease, the model or data loading has a bug. Run as a `--debug` flag.
4. **Post-training:** assert test set positive count >= 200. With stride=1 and 3,514 rows, this is ~882 — no concern.

---

## Risk register

| Risk | Likelihood | Blast radius | Mitigation |
|------|------------|--------------|------------|
| LSTM macro-F1 < XGBoost (0.6084) | High — expected per R6 | Academic story only — not a build failure | Train clean, report honestly. Pre-mortem item 3 already anticipates this. DO NOT tune. |
| NaN loss on MPS | Low-Medium | Training blocked | Guard in training loop: if `torch.isnan(loss): fall back to CPU`. Document in log. |
| Val macro-F1 = 0.0 for all epochs | Low | Training is broken (upstream or code bug) | Overfit check on 200-sample subset before full run. If overfit check also fails → debug window loading. |
| OneCycleLR `total_steps` mismatch | Low | Scheduler error at step > total_steps | Set `total_steps = max_epochs * len(train_loader)` explicitly at init. Log value. |
| Stride=1 val/test overlap inflates F1 | Acknowledged | Reported F1 looks better than generalisation | Document in eval log. Methodology is consistent with XGBoost baseline (same stride=1). |
| Class weights wrong device | Low | CUDA/MPS error on loss computation | `class_weights.to(device)` before passing to CrossEntropyLoss. Check in training init. |
| `nn.LSTM` dropout=0.3 ignored for num_layers=1 | N/A (num_layers=2) | None | Not applicable — we use num_layers=2. Explicitly documented. |
| `build_pipeline()` labeller name mismatch | Low | Dataset load fails | Use `"valid_fvg"` not `"fvg"`. Confirm with `list(LABELLERS.keys())` before script. |

---

## Pre-mortem

1. **LSTM val F1 stuck at 0.0 for 20+ epochs.** Cause: DataLoader not shuffling train (all none-class in first N batches), or class weights not on device. Fix: confirm `shuffle=True` on train_loader, confirm `class_weights.to(device)` called before loss init. Overfit check on 200-sample subset catches this in <2 min.

2. **NaN loss on epoch 1 on MPS.** Cause: gradient explosion through LSTM gates before gradient clipping. Fix: `clip_grad_norm_(model.parameters(), max_norm=1.0)` is in the plan — ensure it runs BEFORE `optimizer.step()`. If NaN persists, fall back to CPU.

3. **OneCycleLR crashes with `Step size must be less than or equal to total steps`.**  Cause: scheduler was created with `total_steps` based on wrong batch count. Fix: print `total_steps = max_epochs * len(train_loader)` before training and verify it is > 0.

4. **Test F1 reported as 0.0 for all classes.** Cause: best checkpoint not loaded before test eval (model evaluated in its post-training-stop state, not best state). Fix: confirm `model.load_state_dict(torch.load(best_ckpt_path))` is called before `eval_epoch(model, test_loader, ...)`.

5. **Script fails on `LABELLERS["valid_fvg"]` KeyError.** The labeller registry name might be `"fvg"` not `"valid_fvg"`. Check `list(LABELLERS.keys())` and update accordingly. Do not assume the name — verify from source.

---

## Out of scope (this plan)

- Hyperparameter tuning (one set of locked defaults only)
- Multi-seed runs (seed=42 only for baseline; 5-seed promoted to final evaluation)
- Focal loss (implemented in `loss.py` but not used as default)
- CNN-LSTM, xLSTM, Transformer (separate plans)
- SHAP feature importance (Phase 5 territory)
- Block-bootstrap CI (Phase 5 territory — use point estimate F1 for this baseline)
- Bidirectional LSTM (rejected per R6)
- Sequence-length ablation (not in scope for baseline)
- `build_pipeline()` modification (use direct parquet loading instead)

---

## Phase 4 CNN-LSTM handoff

After this plan is built and the eval log is written:

**The LSTM test macro-F1 becomes the DL floor.** CNN-LSTM must beat it, or the CNN architecture adds no value. The evaluation log's per-class F1 become the comparison baseline for all subsequent DL architectures.

**Files produced by this plan:**
```
src/models/lstm.py
src/training/__init__.py
src/training/loss.py
src/training/early_stop.py
src/training/train_utils.py
scripts/train_lstm.py
tests/models/test_lstm.py
checkpoints/lstm/lstm_seed42.pt           (gitignored)
checkpoints/lstm/lstm_seed42.meta.json    (gitignored)
logs/lstm_seed42.csv                      (gitignored or committed per preference)
.nb-suite/test-logs/11-May-26/lstm-baseline.md
.nb-suite/build/11-May-26/phase4-lstm-baseline.md
```

**Commands to run in order:**
```bash
# 1. Run unit tests (TDD — before training script)
pytest tests/models/test_lstm.py -v

# 2. Optional: overfit check on small subset
python scripts/train_lstm.py --debug --max-epochs 5

# 3. Full training run
python scripts/train_lstm.py --seed 42

# 4. Verify eval log
cat .nb-suite/test-logs/11-May-26/lstm-baseline.md
```

**Infrastructure reuse for CNN-LSTM, xLSTM, Transformer:**
- `src/training/loss.py` — reuse WeightedCE unchanged
- `src/training/early_stop.py` — reuse unchanged
- `src/training/train_utils.py` — reuse set_seed, log_metadata, eval_epoch unchanged
- `scripts/train_lstm.py` — copy and adapt (only model class + hyperparams change)
- `tests/models/test_lstm.py` — copy test patterns for shape/finite/device/param-count

The training infrastructure built here is the template for all subsequent Phase 4 architectures. Getting it right here pays forward.

---

## Confidence

**High** that the model class is correct — `nn.LSTM` with `batch_first=True`, 2 layers, final hidden state readout is the standard LSTM classification pattern, directly supported by Fischer & Krauss (2018) and R6.

**High** that the training infrastructure (WeightedCE, EarlyStop EMA, AdamW, OneCycleLR, gradient clipping) is correct — all components are specified by R7 with literature backing.

**Medium** on expected F1 — LSTM may not beat XGBoost on ~117 effective independent samples. This is expected and documented. The plan does not require LSTM to win.

**Medium** on MPS stability — NaN loss is possible with LSTM + MPS. The fallback to CPU is in the plan. Training on CPU for this dataset (6,997 windows, batch 32) is ~2–5 min/epoch — acceptable.

## What I couldn't verify

- Whether the `valid_fvg` labeller name is the correct registry key (vs `"fvg"` or another variant). Builder must check `list(LABELLERS.keys())` before using the name.
- Whether `data/processed/spy_h1_labeled.parquet` (referenced in the task brief) is the same as `data/processed/spy_h1.parquet` (what `build_pipeline()` produces). The XGBoost plan resolved this as equivalent — load from `spy_h1.parquet` splits, not `spy_h1_labeled.parquet`. Confirm on disk.
- Exact epoch count to convergence — estimated 50–100 for LSTM on this dataset per R7. Actual will be known after the first training run.
- Whether `SMCWindowDataset` with `drop_cross_session_windows=False` and stride=1 on val/test produces exactly 1,698 / 3,455 windows respectively. Verify window counts match XGBoost log before proceeding to training.
