# Training Protocol — Class-Imbalanced Sequence Labelling (R7)

> **Ready for /nb:plan (Phase 4).**
> Question: What is the concrete training recipe for each of the four architectures given 6,997 train windows, 75/15/10 class distribution, class weights [0.22, 1.10, 1.69], Apple M4 Pro (MPS/CPU), primary metric = minority macro-F1?
> Verdict: Weighted CrossEntropyLoss is correct for this imbalance level. Use AdamW + OneCycleLR. Early stop on smoothed val macro-F1, patience 15, with fallback to fixed epoch cap. Multi-seed (5 seeds) mandatory; block-bootstrap CI on test set.

**Confidence:** High on loss function and optimiser choice. Medium on exact LR values and epoch counts (architecture-dependent, confirm empirically). Medium on MPS reproducibility (known open issue).
**Why this confidence:** Loss function finding grounded in literature consensus for moderate imbalance (75/15/10 is not extreme). Optimiser recommendation grounded in AdamW vs Adam comparative analysis. MPS reproducibility acknowledged as a known unresolved issue with documented severity.
**Depth used:** Standard

---

## Project context

- Dataset: 6,997 train windows, 29 val windows, 58 test windows. Input shape (B, 60, 5).
- Class distribution train: 75% none (0), 15% bull (1), 10% bear (2).
- Class weights from Phase 3: `[0.22, 1.10, 1.69]` (inverse-frequency, sum ≈ 3.0).
- Hardware: Apple M4 Pro, 48 GB RAM. No CUDA. PyTorch MPS or CPU.
- Primary metric: macro-F1 across minority classes (bull + bear). Not accuracy.
- Val set is 29 windows — statistically marginal. Every val metric is noisy.
- Test set is 58 windows — CI estimation required, not point estimates.

---

## Findings

### 1. Loss function: Weighted CE vs Focal Loss

**Source:** Lin et al. 2017 (RetinaNet, focal loss original paper); PyTorch Forums empirical reports; ResearchGate practitioner consensus.

**Mechanism:**
- **Weighted CrossEntropyLoss:** Scales loss per sample by the class weight of the true label. Simple, stable, well-understood.
- **Focal Loss:** Multiplies loss by `(1 - p_t)^gamma`, down-weighting easy examples dynamically. Designed for severe imbalance (e.g., 1:1000 object detection).

**Fit for this project:** Weighted CE. The 75/15/10 split is moderate, not extreme. Focal loss adds a second hyperparameter (gamma, typically 0.5–2.0) with unclear benefit at this imbalance level. The Phase 3 class weights tensor is already computed and validated — use it directly. PyTorch practitioner reports confirm focal loss can *underperform* weighted CE on moderate imbalance because it aggressively down-weights the majority even when the majority is genuinely hard to learn.

**Risk:** If after LSTM baseline the minority F1 is stuck near zero despite weighted CE, re-run with focal loss gamma=1.0 as ablation. Keep it in `src/training/loss.py` as an importable alternative, not the default.

**Decision: `torch.nn.CrossEntropyLoss(weight=class_weights)` as default for all architectures.**

---

### 2. SMOTE / Oversampling for time series

**Source:** Standard literature on temporal data augmentation; R3 labeling-strategy.md.

SMOTE and ADASYN both operate on i.i.d. feature vectors and break temporal order when applied to sequential windows. Creating synthetic interpolated windows between two real windows produces sequences that never occurred — the OHLCV relationships within a window would be structurally invalid (e.g., interpolated volume patterns that cannot occur in real market microstructure).

**Decision: Do not apply any sequence-level oversampling. Use class-weighted loss exclusively.** If additional augmentation is needed, consider window stride variation (already in Phase 3 design) or jitter on volume only (low risk to OHLC structure).

---

### 3. Optimiser: AdamW

**Source:** AdamW paper (Loshchilov & Hutter 2017); comparative analysis at arxiv.org/pdf/2405.13698 (optimal weight decay scaling); ICLR 2023 blogpost on AdamW.

**Mechanism:** AdamW decouples weight decay from the adaptive gradient update. Adam+L2 implicitly scales weight decay by the adaptive learning rate, reducing its regularisation effect on parameters with large gradients. AdamW applies weight decay directly to parameters, consistently.

**Recommendation for ~7k samples + 3-class classification:**
- Overfitting is the primary risk at this dataset size relative to model capacity.
- AdamW's consistent regularisation is more important than Adam's marginal convergence speed advantage.
- Weight decay: `1e-2` (not `1e-4` — `1e-4` is LLM tuning scale; for small datasets 0.01–0.02 is documented as better regularisation).
- The arxiv scaling paper confirms weight decay should be larger for smaller datasets.

**Lion (Zoph et al. 2023):** Sign-based update, designed for very large models. Not recommended here — adds training instability risk and is untested on sequence labelling at this scale. Skip.

**Decision: AdamW, weight_decay=1e-2, for all architectures.**

---

### 4. Learning rate schedule: OneCycleLR

**Source:** Smith & Topin 2018 (Super-Convergence); fastai documentation; PyTorch training guides; empirical comparison on CIFAR-style tasks.

**Mechanism:** OneCycleLR linearly increases LR from `max_lr/10` to `max_lr` over 30% of steps (warmup), then cosine-decays to `max_lr/1000` over the remaining 70%. Includes simultaneous momentum cycling. Enables "superconvergence" — faster convergence with fewer epochs, especially on small datasets where CosineAnnealingLR with full-length decay can waste epochs in the early flat region.

**OneCycleLR vs CosineAnnealingLR:**
- OneCycleLR: better for small datasets with limited epoch budget. Built-in warmup. One-shot (single cycle — not restarts). Less hyperparameter tuning.
- CosineAnnealingLR: requires manual warmup setup, and warm restarts (SGDR) are overkill here — restarts benefit very long training runs.
- ReduceLROnPlateau: reactive, useful if loss plateaus unpredictably. Keep as fallback if OneCycleLR doesn't converge.

**Recommended max_lr by architecture:**
| Architecture | max_lr | Rationale |
|---|---|---|
| LSTM | 3e-3 | LSTMs tolerate moderate LR; common default |
| CNN-LSTM | 1e-3 | CNN layers more sensitive to LR; lower base |
| xLSTM | 5e-4 | Exponential gates require conservative LR to avoid gate saturation early in training |
| Transformer | 1e-3 | Standard for small Transformers; scale to 3e-4 if loss is unstable |

**Decision: OneCycleLR as primary schedule for all architectures, with architecture-specific max_lr.**

---

### 5. Batch size

**Hardware:** 48 GB unified memory. No external VRAM constraint — all model sizes feasible.

**Practical guidance:**
- At 6,997 windows × (60, 5) float32 = 6,997 × 300 × 4 bytes ≈ 8.4 MB for entire train set. Fits trivially in memory.
- Batch 32 produces ~219 batches/epoch. Reasonable gradient noise — not too stochastic (batch 4), not too smooth (batch 256 would overfit early on 6,997 samples).
- All architectures: **batch_size = 32** as default. Can increase to 64 for Transformer if training is slow, but gradient noise at 32 is beneficial for generalisation on small data.

**Gradient accumulation:** Not needed. 48 GB unified memory is not a constraint for these model sizes on a 60×5 input. Skip accumulation — adds complexity with no benefit here.

---

### 6. Epochs and training duration

With batch 32 and 6,997 windows: **~219 batches/epoch**.

| Architecture | Estimated epochs to convergence | Estimated wall time (M4 Pro MPS) | Notes |
|---|---|---|---|
| LSTM | 50–100 | 5–15 min | Fast per-epoch; convergence typically < 60 epochs |
| CNN-LSTM | 50–100 | 8–20 min | CNN adds minimal overhead on (60, 5) input |
| xLSTM | 100–200 | 20–60 min | Slower convergence documented in literature; MPS support uncertain — may fall back to CPU (2–3x slower) |
| Transformer | 80–150 | 15–40 min | Attention over length-60 is cheap; warmup important |

These are rough estimates. With early stopping (patience 15), actual runs may terminate much earlier. Budget 2 hours maximum per architecture per seed.

---

### 7. Early stopping

**Problem:** 29 val windows produces highly noisy val metrics. A single val macro-F1 swing of ±0.15 is plausible purely from noise on such a small set. Standard early stopping on raw val F1 will trigger spuriously.

**Recommendation:**
- **Signal:** Exponential moving average (EMA) of val macro-F1, alpha=0.3 (smooths over ~3 epochs).
- **Patience:** 15 epochs (not the standard 5–10, because the signal is noisy — need more evidence before stopping).
- **Fallback cap:** `max_epochs` hard limit (see table above) — early stopping may never trigger on 29 val windows.
- **Checkpoint:** Save best-of-run by smoothed val macro-F1, not by raw. On final evaluation, load the best checkpoint and evaluate on test set.
- **Do NOT** use each-class F1 separately as stopping signal — with 29 windows, any single class may have 0 val positives in a given epoch, producing undefined F1. Macro-F1 (treating undefined class F1 as 0) is the least unstable single metric.
- **Val loss vs val F1:** Val loss converges more smoothly than F1 but does not directly optimise the primary metric. Use val macro-F1 as primary signal; monitor val loss as a diagnostic. If val F1 is completely uninformative (all zeros across all epochs), something is upstream-broken — do not attribute it to the training protocol.

**Concrete early stopping config:**
```python
class EarlyStop:
    patience: int = 15
    min_delta: float = 1e-4      # minimum improvement to count as improvement
    ema_alpha: float = 0.3        # smoothing factor
    monitor: str = "val_macro_f1"  # smoothed
    mode: str = "max"
```

---

### 8. Checkpointing

- Save best checkpoint by smoothed val macro-F1 at each epoch.
- Keep: best checkpoint + last 3 checkpoints. Delete older ones.
- Checkpoint filename includes: `arch`, `seed`, `epoch`, `val_f1` (to 3 decimal places).
- Example: `checkpoints/lstm_seed42_ep047_f1_0.312.pt`
- Store: model `state_dict`, optimiser `state_dict`, scheduler `state_dict`, epoch, val_f1, seed, git_sha.
- Never commit checkpoint files (`.gitignore` `checkpoints/` and `*.pt`).

---

### 9. Reproducibility checklist

**Seed setup (run at the start of every training script):**
```python
import random, os
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS does not support full determinism via use_deterministic_algorithms.
        # torch.manual_seed() seeds the MPS generator but does NOT guarantee
        # reproducible results across runs due to MPS kernel non-determinism.
        # Documented open issue: pytorch/pytorch#97236
        pass  # MPS seed is set by torch.manual_seed() above
    os.environ["PYTHONHASHSEED"] = str(seed)
```

**`torch.use_deterministic_algorithms(True)` — DO NOT enable on MPS.**
- Confirmed via pytorch/pytorch#122394: causes 8x slowdown on Apple Silicon (macOS 14.4+), caps GPU utilisation at ~50%, makes training impractical.
- `warn_only=True` allows training to complete but does not restore reproducibility.
- **MPS is inherently non-deterministic** even with seeding. This is a known limitation of the MPS backend as of PyTorch 2.x.

**Practical MPS reproducibility approach:**
1. Set all seeds (`torch.manual_seed`, `np.random.seed`, `random.seed`, `PYTHONHASHSEED`).
2. Run each experiment with `num_workers=0` in DataLoaders (eliminates worker non-determinism).
3. Accept that MPS results will vary slightly run-to-run. Use multi-seed protocol to bound variance.
4. If strict reproducibility is required: fall back to CPU (`device = "cpu"`). Training is fast enough on CPU for this dataset size (6,997 windows, batch 32).

**DataLoader seeding:**
```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

DataLoader(..., worker_init_fn=seed_worker, generator=g, num_workers=0)
```

**Log per run (mandatory):**
```python
{
    "seed": seed,
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "device": str(device),
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "timestamp": datetime.utcnow().isoformat(),
}
```

**Multi-seed runs:**
- 5 seeds minimum: `[42, 1337, 2024, 7, 99]`. Enough to estimate mean ± std of minority F1.
- 10 seeds for final model selection and reporting. 10 provides tighter CI and allows detection of high-variance seeds.
- On 5-seed runs: report mean ± std of val minority F1. On final best architecture: promote to 10-seed evaluation on test set.

---

### 10. Bootstrap CI protocol (test set)

**Problem:** 58 test windows is small. Point F1 estimates have wide CI. Cannot report `F1 = 0.42` without a CI.

**Protocol — stationary block bootstrap:**
- Block size: 10 windows (rationale: window length = 60 candles = ~10 trading days on H1; temporal autocorrelation in label occurrence likely decays within this range). Use `arch` package (`statsmodels`) or manual implementation.
- Number of bootstrap samples: **2,000**. Standard in ML evaluation literature; diminishing returns beyond 2,000 for CI width estimation.
- Metric computed per sample: macro-F1, per-class F1 separately.
- CI: 95% percentile interval (2.5th and 97.5th percentiles of the bootstrap distribution). Not studentised — too few samples for reliable variance estimation.
- Report as: `F1 = 0.42 [95% CI: 0.31–0.53]` (example).
- **Block bootstrap not standard bootstrap:** standard bootstrap draws i.i.d. — invalid for temporal data. Block bootstrap preserves local temporal structure. With 58 test windows and block size 10, we get ~5–6 blocks, which is minimal. CI will be wide — expected. Report honestly.
- Implementation note: if 58 windows yields < 3 blocks at block_size=10, reduce block size to 5.

```python
def block_bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 10,
    n_samples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Returns {class_name: (lower_ci, upper_ci)} and {"macro": (lower_ci, upper_ci)}.
    Draws n_samples block-bootstrap resamples from y_true/y_pred.
    Preserves temporal order within blocks.
    """
    ...
```

---

## Recommendation: Training recipe table

| Architecture | Loss | Optimiser | max_lr | Schedule | Batch | Max Epochs | Early stop signal | Notes |
|---|---|---|---|---|---|---|---|---|
| LSTM | Weighted CE weight=[0.22,1.10,1.69] | AdamW wd=1e-2 | 3e-3 | OneCycleLR | 32 | 100 | Smoothed val macro-F1, patience 15 | Baseline. Must beat naive. If F1=0 at epoch 20, debug upstream not here. |
| CNN-LSTM | Weighted CE weight=[0.22,1.10,1.69] | AdamW wd=1e-2 | 1e-3 | OneCycleLR | 32 | 100 | Smoothed val macro-F1, patience 15 | Lower LR — CNN early layers sensitive. |
| xLSTM | Weighted CE weight=[0.22,1.10,1.69] | AdamW wd=1e-2 | 5e-4 | OneCycleLR | 32 | 200 | Smoothed val macro-F1, patience 20 | Conservative LR for exponential gates. Extra patience — known slower convergence. If loss NaN: drop to mLSTM-only config (no sLSTM), LR 1e-4. |
| Transformer | Weighted CE weight=[0.22,1.10,1.69] | AdamW wd=1e-2 | 1e-3 | OneCycleLR | 32 | 150 | Smoothed val macro-F1, patience 15 | Warmup critical — OneCycleLR provides this. If unstable: reduce max_lr to 3e-4. |

---

## Apple Silicon notes

1. Device selection:
```python
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
```

2. **Do not** enable `torch.use_deterministic_algorithms(True)` on MPS — performance drops 8x, GPU hits 50% cap. Document this in training logs.

3. MPS is non-deterministic by design even with seeds set. For comparative experiments requiring reproducibility, use `device = "cpu"`. Training on CPU for 6,997 windows at batch 32 is estimated at 2–5 min/epoch for LSTM/CNN-LSTM (fast enough).

4. xLSTM official package (`xlstm`) may require CUDA. Smoke-test on day 1 of Phase 4 xLSTM workstream:
```python
import xlstm; m = xlstm.xLSTMBlockStack(...); m.to(device)
# if RuntimeError mentioning CUDA: fall to CPU or community port
```

5. All tensors must be `.to(device)` including the class_weights tensor:
```python
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

6. MPS memory is unified with system RAM — 48 GB shared. No OOM risk for this model/batch scale. Monitor with `torch.mps.current_allocated_memory()`.

---

## Reproducibility checklist (condensed, for Phase 4 build)

- [ ] `set_seed(seed)` called before model init, dataloader init, and train loop
- [ ] `num_workers=0` in all DataLoaders
- [ ] `seed_worker` and generator set in DataLoader
- [ ] `git_sha` logged per run
- [ ] `torch.__version__`, `python_version` logged per run
- [ ] `device` logged per run
- [ ] `torch.use_deterministic_algorithms(True)` explicitly NOT called on MPS
- [ ] Checkpoint filename includes seed and val_f1
- [ ] 5 seeds minimum; 10 seeds for final model comparison
- [ ] class_weights tensor sent to device before passing to loss function

---

## What /nb:plan needs to know

- `src/training/loss.py`: implement `WeightedCE` (wraps `nn.CrossEntropyLoss(weight=...)`) and `FocalLoss` (alternative, not default). Both accept class_weights tensor.
- `src/training/scheduler.py`: implement `OneCycleLR` wrapper with architecture-specific defaults. Hard-code `total_steps = max_epochs * batches_per_epoch` at scheduler init time.
- `src/training/early_stop.py`: implement `EarlyStop` class with EMA smoothing (alpha=0.3), patience, and min_delta.
- `src/training/train.py`: training loop must log: epoch, train_loss, val_loss, val_macro_f1 (raw and smoothed), per-class val_f1. Write to CSV at `logs/<arch>_seed<N>.csv`.
- `src/eval/bootstrap.py`: block bootstrap F1 CI, block_size=10, n_samples=2000.
- Phase 4 build sequence: LSTM → CNN-LSTM → xLSTM → Transformer. Each architecture is a separate `/nb:build` cycle. Do not batch them.
- xLSTM day-1 smoke test: install `xlstm` package, verify it initialises on MPS/CPU before any training code is written.
- All checkpoint files excluded from git via `.gitignore`.
- Decision on focal loss is deferred: implement it but don't use it as default. If LSTM minority F1 < 0.1 at epoch 50, try focal loss gamma=1.0 as diagnostic.

---

## Open questions

- **xLSTM MPS compatibility:** Unknown until smoke-test. If incompatible: CPU fallback (acceptable, see timing estimates) or community PyTorch port. This blocks the xLSTM workstream start.
- **Exact val window count:** Prompt states 29 val windows. With 29 windows and 3 classes, the probability of having 0 bull or 0 bear windows in val is non-trivial. Compute val class distribution before training begins; if a class has 0 val examples, early stopping on per-class F1 is undefined — fall back to val loss + macro-F1 treating undefined class as 0.
- **OneCycleLR total_steps:** Must be set at scheduler init with `total_steps = max_epochs * batches_per_epoch`. If early stopping fires before max_epochs, the scheduler will be ahead of actual training — acceptable; it just means LR decays faster than scheduled. Not a correctness issue.

---

## What I couldn't verify

- **Exact MPS performance drop on M4 Pro (vs M1 Max):** The pytorch/pytorch#122394 issue reports 8x degradation on M1 Max. M4 Pro may differ — better or worse. The recommendation to avoid `use_deterministic_algorithms(True)` on MPS is conservative but safe.
- **xLSTM official package MPS/CPU support:** The `xlstm` package (Beck et al. 2024) targets CUDA by default. CPU fallback may work but is not officially documented. Community ports exist but vary in stability. Cannot confirm without running the package.
- **Exact epoch counts to convergence:** No published benchmarks for LSTM/CNN-LSTM on a 7k-window, 3-class, 60-step time series task. The estimates (50–200 epochs) are based on analogous small-dataset benchmarks and should be confirmed empirically in the LSTM baseline run.
- **Block bootstrap with 5–6 blocks:** With 58 test windows and block_size=10, only ~5–6 blocks are available. Block bootstrap theory assumes block count >> 1. This is technically valid but CI width will be substantial (±0.15–0.20 on F1). This limitation must be stated in the evaluation notebook.
- **Lion optimiser on sequence labelling:** No verified benchmarks found for Lion on LSTM/Transformer sequence classification at this scale. Exclusion is conservative and correct.
