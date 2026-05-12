# CNN-LSTM Architecture — SMC FVG Detection on SPY H1 — Research

> **Ready for /nb:plan.**
> Question: What CNN-LSTM topology, Optuna search space, and implementation pattern should be used for the next model stage in the SMC FVG detection pipeline?
> Verdict: Single-branch Conv1d(k=3) → BatchNorm → ReLU → (optional pool) → LSTM(64) → head. Multi-kernel parallel is a Optuna option, not the default. MLSTM-FCN-style 128-256-128 filters is for large UCR datasets — scale down to 32-64 for our eff-n=117 regime.

**Confidence:** High for topology; Medium for expected performance delta vs XGB.
**Why this confidence:** Topology grounded in MLSTM-FCN paper (arXiv 1801.04503), prior architecture-comparison research (8-May-26), and confirmed PyTorch 2.11 API (context7). Performance prediction is extrapolated from analogous candlestick literature — not directly measured on ValidFVG.
**Depth used:** Deep

---

## Project context

- **Current best:** XGB 0.721 ± 0.001 mean macro-F1. LSTM 0.599 ± 0.025.
- **Input:** `(B, 60, 5)` float32, per-window normalised OHLCV.
- **Label:** ValidFVG 3-class (none/bull/bear), ~3% positive rate. WeightedCE confirmed optimal (G3).
- **Eff-n:** ~117 independent samples (stride=1 → 60× overlap on train set of 10,577 bars).
- **Device:** CPU only. MPS disabled (torch 2.11 LSTM backprop bug). Conv1d on CPU is trivially fast.
- **Registration pattern:** `register_model("cnn_lstm")(FVGCNNLSTMClassifier)` in `src/config/_model_registrations.py`.
- **Class weights:** `data/processed/class_weights_fvg_valid.json` — reuse unchanged.
- **Prior LSTM HP:** hidden=128, num_layers=1, dropout=0.318, head_dropout=0.526, lr=5.30e-4, weight_decay=3.92e-5, batch_size=16.

---

## Findings

### Topology options evaluated

#### Option A: Single-branch CNN→LSTM (recommended)

**Source:** arXiv 1801.04503 (MLSTM-FCN); Mijanur Rahman Medium; architecture-comparison.md (8-May-26)
**Mechanism:** Conv1d layers extract local pattern features (k=3 encodes 3-candle FVG geometry directly), output is a transformed sequence of shape `(B, C_out, T)`. Transpose to `(B, T, C_out)`, feed to LSTM. LSTM aggregates temporal context over the conv-feature sequence. Final hidden state → dropout → linear → 3 logits.
**Fit:** Yes — this is the architecturally-motivated choice. k=3 is the exact receptive field of the FVG 3-candle rule. LSTM job is reduced from "detect AND aggregate" to "aggregate over pre-extracted features" — easier learning problem.
**Risk:** Two conv layers may over-smooth. Include 1-conv ablation in Optuna via `n_conv_layers ∈ {1, 2}`.

#### Option B: Multi-kernel parallel (k=3 + k=5 + k=7 concat)

**Source:** MLSTM-FCN inspiration; Rahman Medium parallel variant
**Mechanism:** Three Conv1d branches with k=3, k=5, k=7 run in parallel on the same input. Outputs concatenated along channel dim. Total params ~3× single-branch.
**Fit:** Partial. Captures multi-scale patterns (3-candle FVG, 5-candle BOS). But at eff-n=117, tripling params risks overfit. Only sensible if `conv_filters` is kept very small (16 per branch = 48 total).
**Risk:** Overfitting dominant at our scale. Keep as Optuna option via `multi_kernel: bool` flag, not the default.

#### Option C: MLSTM-FCN full (128-256-128, k=8-5-3, squeeze-excite)

**Source:** arXiv 1801.04503 — outperforms LSTM on 28/35 UCR datasets (p<0.0028)
**Mechanism:** Three conv blocks + SE attention + LSTM branch with dimension shuffle.
**Fit:** No for our scale. 128-256-128 filters designed for large UCR datasets (thousands of independent samples). At eff-n=117, this is ~10× over-parameterised. SE blocks add meaningful overhead for inter-variable correlation — our 5 OHLCV channels are already highly correlated (OHLC by definition). Dimension shuffle trick (transpose M×Q → Q×M) only helps when M (variables) > Q (timesteps), inverted here.

#### Option D: Parallel CNN + LSTM (independent paths concat)

**Source:** Rahman Medium architecture 3
**Mechanism:** CNN path and LSTM path process input independently, outputs concat'd.
**Fit:** No. LSTM-only path would essentially replicate the baseline LSTM. The gain from the conv path would be diluted by the LSTM path's tendency to overfit directly to raw OHLCV. Adds parameters without architectural justification.

---

## Recommended topology

```
Input: (B, 60, 5)
  ↓ permute(0, 2, 1)              # Conv1d expects (B, C_in, L) = (B, 5, 60)
  ↓ Conv1d(5, F, k, padding=k//2) # padding='same' via manual: pad=(k-1)//2
  ↓ BatchNorm1d(F)
  ↓ ReLU
  [↓ Conv1d(F, F, k, padding=k//2)  # optional 2nd conv layer
   ↓ BatchNorm1d(F)
   ↓ ReLU]
  ↓ permute(0, 2, 1)              # back to (B, 60, F) for LSTM
  ↓ LSTM(F, H, num_layers=1, batch_first=True)
  ↓ h_n[-1]                       # (B, H) final hidden state
  ↓ Dropout(head_dropout)
  ↓ Linear(H, 3)
Output: (B, 3) logits
```

**Concrete shapes (default HP: F=32, k=3, H=64):**

| Layer | Output shape | Params |
|-------|-------------|--------|
| Input | (B, 60, 5) | — |
| permute | (B, 5, 60) | — |
| Conv1d(5→32, k=3, pad=1) | (B, 32, 60) | 5×3×32+32 = 512 |
| BatchNorm1d(32) | (B, 32, 60) | 64 |
| Conv1d(32→32, k=3, pad=1) | (B, 32, 60) | 32×3×32+32 = 3,104 |
| BatchNorm1d(32) | (B, 32, 60) | 64 |
| permute → LSTM(32→64, 1L) | (B, 60, 64) + h_n | 4×(32+64+1)×64 = 24,832 |
| Dropout + Linear(64→3) | (B, 3) | 195 |
| **Total** | | **~28,771** |

LSTM baseline was ~120k params. CNN-LSTM at default HP is ~29k — significantly smaller, appropriate for eff-n=117. Optuna can search up to 64/128 filters + 128 hidden for ~100k total.

**No pooling in default config.** At 60 bars, MaxPool1d(2) halves to 30 — acceptable but unnecessary since LSTM handles variable-length aggregation. Pool only if Optuna finds it beneficial.

---

## Optuna search space

```yaml
# experiments/cnn_lstm_g1.yaml
name: cnn_lstm_g1_validfvg
data:
  labeller: fvg_valid
  window_size: 60
model:
  arch: cnn_lstm
  # Optuna will override these:
  n_conv_layers: 2          # {1, 2}
  conv_filters: 32          # {16, 32, 64}
  kernel_size: 3            # {3, 5, 7}
  use_pool: false           # {true, false}
  pool_type: "max"          # {"max", "avg"} — only used if use_pool=true
  lstm_hidden: 64           # {32, 64, 128}
  lstm_layers: 1            # {1, 2}
  dropout: 0.318            # uniform(0.1, 0.5)
  head_dropout: 0.526       # uniform(0.3, 0.7)
train:
  seeds: [0, 17, 42, 123, 2024]
  batch_size: 16            # {16, 32} — keep small for eff-n=117
  lr: 5.3e-4                # log-uniform(1e-4, 1e-3)
  weight_decay: 3.92e-5     # log-uniform(1e-5, 1e-4)
  optimizer: adam
  scheduler: none
  max_epochs: 100
  patience: 15
  loss: weighted_ce
  device: cpu
```

**Optuna trial budget:** 50 trials. Search space is ~7-dimensional (n_conv_layers × conv_filters × kernel_size × use_pool × lstm_hidden × lstm_layers × lr/wd continuous). 50 trials with TPE sampler gives reasonable coverage — mirrors LSTM G1 (38+38 complete+pruned → settled trial 42).

**Pruning:** EMA-smoothed val macro-F1, same as LSTM. Prune if median of completed trials at same epoch step is better.

---

## Window size decision

**Use W=60 (canonical).** Rationale:
- G7 found W=90 best for plain LSTM (val F1 0.649 vs 0.618). The gain came from giving LSTM more context to detect the FVG pattern.
- With Conv1d(k=3), the conv layer explicitly encodes 3-candle locality at every position in the 60-bar sequence. The LSTM no longer needs extra bars to "find" the pattern — it receives pre-extracted FVG-signal features.
- Switching to W=90 reduces eff-n from 86 to 57 (−34%). At our scale, this is a meaningful cost. The receptive-field benefit is already provided by the conv layer.
- **Option:** After G1 HP is established, run a targeted window sweep (W=60 vs W=90, best HP, 3 seeds) as a post-G1 ablation. Do not make W=90 the default for CNN-LSTM from the start.

---

## Loss + class weights

**Reuse unchanged.** G3 confirmed WeightedCE strictly dominates Focal loss for ValidFVG at all γ (worst Δ = −0.109 at γ=3). `data/processed/class_weights_fvg_valid.json` weights [0.0242, 1.2320, 1.7438] were computed on train split only — no leakage. CNN-LSTM uses same `WeightedCE` from `src/training/loss.py`.

---

## Implementation pattern

New file: `src/models/cnn_lstm.py`

```python
"""cnn_lstm.py — CNN-LSTM classifier for FVG ternary prediction.

Architecture: Conv1d(5→F, k) → BN → ReLU [× n_conv_layers] → LSTM(F→H) → FC(H→3)
Input: (B, 60, 5) float32 — batch of 60-candle OHLCV windows (per-window normalised)
Output: (B, 3) raw logits
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FVGCNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        conv_filters: int = 32,
        kernel_size: int = 3,
        n_conv_layers: int = 2,
        use_pool: bool = False,
        pool_type: str = "max",      # "max" | "avg"
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.318,
        head_dropout: float = 0.526,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2   # 'same' padding: preserves L dimension

        conv_blocks: list[nn.Module] = []
        in_ch = input_size
        for _ in range(n_conv_layers):
            conv_blocks += [
                nn.Conv1d(in_ch, conv_filters, kernel_size, padding=padding),
                nn.BatchNorm1d(conv_filters),
                nn.ReLU(),
            ]
            in_ch = conv_filters
            if use_pool:
                Pool = nn.MaxPool1d if pool_type == "max" else nn.AvgPool1d
                conv_blocks.append(Pool(kernel_size=2, stride=1, padding=1))
        self.conv = nn.Sequential(*conv_blocks)

        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → conv expects (B, C, T)
        x = x.permute(0, 2, 1)          # (B, 5, 60)
        x = self.conv(x)                 # (B, conv_filters, T')
        x = x.permute(0, 2, 1)          # (B, T', conv_filters)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]            # (B, lstm_hidden)
        out = self.head_dropout(last_hidden)
        return self.classifier(out)      # (B, 3)
```

Registration in `src/config/_model_registrations.py`:
```python
from src.models.cnn_lstm import FVGCNNLSTMClassifier
register_model("cnn_lstm")(FVGCNNLSTMClassifier)
```

The training script mirrors `scripts/training/train_lstm.py` — same `LSTMObjective` pattern adapted for CNN-LSTM kwargs. Create `scripts/training/train_cnn_lstm.py`.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Conv over-smooths 3-candle pattern | Medium | Medium | Include `n_conv_layers=1` in Optuna; compare 1-layer vs 2-layer |
| Overfit on eff-n=117 | Medium-High | High | Head dropout ≥ 0.3, weight_decay ≥ 1e-5, patience=15, small batch=16 |
| Pool reduces temporal resolution too aggressively | Low | Medium | Default `use_pool=False`; only searched if Optuna explores it |
| BatchNorm unstable at batch_size=16 | Low | Low | BN with B=16 is fine (L=60 gives 60×16=960 values per BN stat). Monitor val loss for instability. |
| `padding=kernel_size//2` drops 1 bar for even k | Low | Low | k ∈ {3,5,7} are all odd — padding = 1, 2, 3 — output length exactly preserved |
| LSTM dropout ignored when lstm_layers=1 | Low | None | Code guards with `dropout=0.0 if lstm_layers==1` — correct per PyTorch docs |
| CPU compute budget | Low | Low | At F=32, k=3, H=64, B=16: ~3 MFLOPS/batch. 100 epochs × 700 batches ≈ minutes. +20-30% vs LSTM baseline. |

---

## Expected performance

**Honest prediction:** CNN-LSTM will beat LSTM (0.599 → 0.62-0.65 macro-F1, ~+3-5%). Beating XGB (0.721) is uncertain.

Evidence basis:
- MLSTM-FCN outperforms LSTM-only on 28/35 UCR datasets. Arithmetic rank 3.29 vs LSTM 4.63. But UCR datasets have far more independent samples than our eff-n=117.
- Financial OHLCV CNN-LSTM literature (candlestick papers, MSE-focused) claims ~20% error reduction vs LSTM. MSE improvement does not translate directly to F1 on 3% positive rate.
- SHAP G5 found `gap_norm_bull`, `gap_bear` dominate XGB. These are hand-engineered features summarising the geometric gap between bars — exactly what k=3 conv should learn to encode. If conv learns this geometry, CNN-LSTM may approach XGB.
- XGB advantage: operates on 35 hand-crafted features encoding gap geometry + momentum explicitly. CNN-LSTM must discover these from raw OHLCV. At eff-n=117, XGB's feature engineering is a strong prior.

**Conservative estimate:** CNN-LSTM macro-F1 ∈ [0.62, 0.68]. Beating XGB 0.721 requires the conv layer to match or exceed the gap-geometry inductive bias built into XGB features — possible but not guaranteed.

**Failure mode:** If CNN-LSTM macro-F1 < 0.60 (worse than LSTM), likely cause is overfit. Check: val F1 >> test F1 = overfit signal. Fix: reduce conv_filters to 16, add more head_dropout.

---

## What /nb:plan needs to know

- **New file:** `src/models/cnn_lstm.py` with class `FVGCNNLSTMClassifier`.
- **New experiment config:** `experiments/cnn_lstm_g1.yaml` mirroring `lstm_g1.yaml`.
- **New training script:** `scripts/training/train_cnn_lstm.py` mirroring `train_lstm.py`.
- **Registration:** Add to `src/config/_model_registrations.py` — one import + one `register_model` call.
- **`padding=k//2`** (integer division) preserves sequence length for odd k={3,5,7}. Do not use `padding='same'` — PyTorch Conv1d `'same'` disallows stride>1 and has torch version caveats.
- **`use_pool=False` default.** If pool is enabled in Optuna trial, use `kernel_size=2, stride=1, padding=1` for MaxPool1d — this preserves length (does NOT halve T). If stride=2 is wanted, Optuna must explicitly set it — do not default to stride-2.
- **No multi-kernel parallel branch as default.** Keep Optuna search space simple (7 dims, 50 trials). Add parallel branch only if G1 results suggest plateau and ablation time permits.
- **WeightedCE unchanged.** No new class weights needed.
- **Window W=60 canonical.** Post-G1 ablation vs W=90 is optional, not a blocker.
- **No MPS.** CPU only — same `device: cpu` in YAML.
- **Inspect adapter needed.** After training, `src/inspect/` will need a `CNNLSTMAdapter` — same pattern as `LSTMAdapter`. Plan this as part of CNN-LSTM workstream day 2.

---

## Open questions

1. **Should Optuna for CNN-LSTM include `multi_kernel: bool` as a dimension?** Would increase search space to ~9-dim. Recommend against for G1 — run clean single-branch first. Add multi-kernel only if single-branch plateaus.
2. **`use_pool` with `stride=1` vs `stride=2`.** Pool with stride=1 adds context averaging without length reduction. Pool with stride=2 halves T (60→30) — reduces LSTM computation but destroys per-bar resolution. Recommend stride=1 if pool is enabled; document this decision in the YAML.
3. **Should `dropout` in `nn.LSTM` be searched independently from `head_dropout`?** Current plan: yes, same ranges as LSTM G1. Could simplify to one shared dropout param.

---

## What I couldn't verify

- **Conv1d on CPU with deterministic mode (`torch.use_deterministic_algorithms(True)`):** Not tested. LSTM G2 runs used CPU deterministic. Conv1d deterministic on CPU is standard (no scatter-add ops) — should not be an issue, but confirm in smoke test.
- **Exact F1 delta CNN-LSTM vs LSTM on ValidFVG-equivalent tasks:** No published paper benchmarks this exact task (3-class, ~3% positive, eff-n~117, 5-channel OHLCV). The performance prediction is extrapolated.
- **BatchNorm1d behaviour with B=16 and heavy class imbalance (97/1.9/1.3%):** BN computes stats over (B × L) = 960 values per channel per batch. With batch sampling weighted by class, some batches may be all-None class. Monitor for BN instability (NaN loss in first 5 epochs = signal to switch to GroupNorm or remove BN).

---

## Falsification attempts

1. **"A plain CNN classifier (no LSTM) would be simpler and equally good."** Plausible — Bai et al. (TCN) shows conv-only ≥ LSTM on many short-sequence tasks. But LSTM adds global context over the 60-bar window, which is relevant for whether the 3-candle FVG occurs in a trending context. Keep CNN-LSTM; flag pure CNN as a Phase 5 ablation.
2. **"Larger filters (64-128) always help."** Contradicted by eff-n=117 regime. MLSTM-FCN's 128-256-128 was designed for UCR datasets with thousands of independent samples. At eff-n=117, large filter counts increase overfitting risk faster than they increase capacity benefit. Starting at F=32 is correct.
3. **"k=3 is too small — financial patterns span 10-20 bars."** G5 SHAP shows `ret_60` (60-bar return) dominates. But the *specific FVG geometry* is 3-bar local. Conv(k=3) captures the local pattern; LSTM captures the global context. No need for conv k>7.

---

## Sources

- [Multivariate LSTM-FCNs for Time Series Classification — arXiv 1801.04503](https://ar5iv.labs.arxiv.org/html/1801.04503)
- [Different ways to combine CNN and LSTM for time series classification — Medium](https://medium.com/@mijanr/different-ways-to-combine-cnn-and-lstm-networks-for-time-series-classification-tasks-b03fc37e91b6)
- [PyTorch 2.11 Conv1d + MaxPool1d + LSTM API — context7](https://docs.pytorch.org/docs/2.11/generated/torch.nn.modules.conv.Conv1d.html)
- [Investigating Market Strength Prediction with CNNs on Candlestick Chart Images — ACM 2024](https://dl.acm.org/doi/full/10.1145/3690771.3690776)
- [CNN Long Short-Term Memory Networks — MachineLearningMastery](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)
- [Architecture Comparison Research 8-May-26](../.nb-suite/research/8-May-26/architecture-comparison.md) — internal
- [Empirical Evaluation of Generic Convolutional and Recurrent Networks — Bai et al. 2018 (TCN)](https://arxiv.org/abs/1803.01271)
