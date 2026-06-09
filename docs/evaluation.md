# Evaluation Methodology

How we tested fairly and what the results actually mean.

---

## 1. The cardinal rule: no peeking into the future

A model trained on data that overlaps with its test period will look better than it really is
because it has already "seen" the market conditions it is being evaluated on. To prevent this,
every data split is strictly temporal.

**Boundaries (hardcoded in `src/data/split.py`, constant `SPLIT_BOUNDARIES`):**

| Split | Period |
|-------|--------|
| Train | 2016-01-01 -- 2021-12-31 |
| Validate | 2022-01-01 -- 2022-12-31 |
| Test | 2023-01-01 -- 2025-12-31 |

The `temporal_split` function slices by timestamp inequality; it never shuffles rows. The function
also raises `ValueError` if any split returns zero bars, enforcing the boundary contract at
runtime.

**Label integrity (the N+2 rule).** Each label is attached to the candle at index N+2, the
reaction bar -- the first bar at which all six FVG criteria are knowable from market history alone.
The labeller (`ValidFVGLabeller` in `src/data/labels/valid_fvg.py`) uses only
`shift(1)`-lagged rolling values so that no bar at position i can see bar i+1 during labelling.
A mandatory pytest fixture enforces this anti-lookahead invariant on every CI run.

---

## 2. The right metric for a rare event

Roughly 97% of SPY H1 candles are labelled "no FVG". A model that predicts "no FVG" for every
bar achieves ~97% accuracy while being completely useless. We therefore report **macro-averaged
F1 on the three-class output** (none / bullish FVG / bearish FVG). Macro averaging weights each
class equally, which means the rare bull and bear classes count as much as the majority class.
We also report per-class F1 for bull and bear separately to see which direction the model
struggles with.

To counteract the imbalance during training, we use inverse-frequency class weights in the
loss function (`WeightedCrossEntropyLoss`). The weights are derived from the training split only
and persisted to `data/processed/class_weights.json` so they cannot accidentally include
validation or test label counts.

---

## 3. Not trusting one lucky run

A single training run with a single random seed can produce an unusually good or unusually
bad result by chance. We run every model with **five fixed seeds** (0, 17, 42, 123, 2024)
and report mean F1 plus sample standard deviation. If a model's mean looks good but its
standard deviation is large, the headline number is not reliable.

On top of the seed spread we compute **block bootstrap confidence intervals**. The procedure
is: take the test-set predictions, resample blocks of 60 consecutive bars (rather than
individual bars, to preserve the autocorrelation structure of a time series), compute the
mean-across-seeds F1 for each resample, repeat 1 000 times, and read off the 2.5th and 97.5th
percentiles. This gives a 95% CI that reflects both seed variance and the limited test-set
size (5 198 bars, effective_n = 86 non-overlapping blocks). Implementation:
`scripts/rigor/bootstrap_ci_multiseed.py`.

The effective_n of 86 blocks is small. This means CIs are wide enough that some model pairs
overlap -- for those pairs we call the result "statistically tied" rather than asserting a
winner (see Section 5).

---

## 4. Deciding the rule before seeing the results

To prevent unconsciously choosing the metric or comparison that makes results look best, the
decision rule was written down before running the Phase 4 multi-symbol experiment.

The pre-registered criterion (from `.nb/plan/09-Jun-26/multi-symbol-expansion.md`, recorded
before Phase 4 training):

- CNN-LSTM macro-F1 on the fixed SPY holdout **above 0.674** (the upper bound of the baseline
  95% CI) --> data-bound hypothesis confirmed.
- CNN-LSTM F1 in **0.639--0.674** --> grey zone, defer to bear_f1 direction.
- CNN-LSTM F1 **at or below 0.639** --> hypothesis rejected.

The Phase 4 result (0.675) cleared the threshold. This pre-registration prevents the
post-hoc reasoning "it improved a little, so let's call it confirmed."

Similarly, the DL-vs-capacity diagnostic was run against a pre-committed decision matrix
(documented in `reports/rigor/07-Jun-26/dl_diagnosis_decision.md`): the matrix maps
(train F1 level, learning-curve plateau yes/no) to a regime cell (data+reg-bound vs
capacity-bound vs unstable). Each model was placed in a cell by objective criteria before
the improvement lever was named.

---

## 5. Honest reporting of what did not work

Results are reported with their failure modes, not just the headline numbers.

**Transformer instability.** The Transformer reaches 0.58--0.61 on four of five seeds but
collapses to majority-only prediction on seed 42 (macro-F1 0.379, bear_f1 0.000). The
collapse recurred in independent learning-curve reruns with the same seeds, so the true
collapse rate is higher than 1-in-5. The mean F1 (0.548 +/- 0.095) is therefore
run-dependent. We report the mean and std honestly and flag that the Transformer is
instability-bound, not data-bound. Multi-symbol data cannot fix an optimisation stability
problem; the indicated lever is architecture-level stabilisation (warmup schedule, lower
learning rate, pre-norm).

**xLSTM underfit.** xLSTM train F1 is 0.33--0.40 (below 0.85, the threshold for "model can
fit the data"). A model that cannot fit its training set will not benefit from more data. The
result (macro-F1 0.369 +/- 0.007) is the ladder floor. We report it as an untuned result and
note that the 2025 literature suggests xLSTM underperforms on short financial sequences; the
gap to CNN-LSTM (0.27) is unlikely to close with tuning alone.

**XGBoost input mismatch.** XGBoost consumes 35 hand-engineered features; the four DL models
consume raw (60, 5) OHLCV windows. The XGB lead over CNN-LSTM (0.721 vs 0.639) is partly a
representation advantage, not a pure architecture comparison. The DL-vs-DL ranking (identical
input) is fair; the XGB row is context only.

**CI overlap for CNN-LSTM vs LSTM.** Their 95% CIs overlap (CNN-LSTM [0.603, 0.674], LSTM
[0.560, 0.628], overlap region 0.603--0.628). Strictly these two are statistically tied on
F1 alone. The carrier designation for CNN-LSTM is therefore made on the tie-break: higher
point estimate (0.639 vs 0.595), tighter standard deviation (0.013 vs 0.015), and the larger
final-segment learning-curve gain (+0.041 vs +0.021 from 80% to 100% of training data).

**The "supported, not proven" caveat.** Phase 4 confirmed the data-bound hypothesis by the
pre-registered threshold, but the CI lower bound (0.637) falls 0.002 below the baseline
point estimate (0.639). Strict interpretation: the improvement is not unambiguously confirmed
at 95% confidence given the 86-block effective test size. We describe the result as "supported"
rather than "proven."

---

## 6. Reproducibility

Every training run is fully reproducible from its YAML config plus the fixed seed list.

- **Seeds.** `src/training/train_utils.py::set_seed(seed)` sets `random.seed`, `numpy.random.seed`,
  and `torch.manual_seed` before model init and data loading.
- **Configs.** All experiment hyperparameters live in `experiments/*.yaml`. Configs are version-
  controlled alongside code. The `--set key=value` CLI override mechanism lets individual
  parameters be changed without modifying the YAML file.
- **Prediction artefacts.** Each seed run saves a `<model>_seed<N>_preds.npz` file containing
  `y_true` and `y_pred` arrays for the test set. These are the inputs to the bootstrap script,
  so CIs can be recomputed independently from the model checkpoints.
- **Reports.** All rigor outputs are written to timestamped directories under `reports/rigor/`.
  The canonical results referenced in this document are in
  `reports/rigor/07-Jun-26/` (DL ladder) and `reports/rigor/09-Jun-26/` (Phase 4 multi-symbol).

**Exact commands to reproduce the ladder results:**

```bash
# Five-seed ladder for CNN-LSTM
python scripts/rigor/multiseed_run.py \
    --model cnn_lstm \
    --config experiments/cnn_lstm_g1.yaml \
    --set "train.seeds=[0,17,42,123,2024]"

# Bootstrap CI from saved prediction files
python scripts/rigor/bootstrap_ci_multiseed.py \
    --pred-dir reports/rigor/07-Jun-26 \
    --model cnn_lstm \
    --block-size 60 --n-iter 1000 \
    --output-dir reports/rigor/07-Jun-26/bootstrap
```

---

## Validation sprint (G1--G10)

In May 2026 we ran a ten-gate validation sprint on the LSTM and XGBoost models (2016--2025 data,
ValidFVG labels). Each gate answered one methodological question before advancing. The table below
summarises what each gate tested and what it found; full per-gate numbers are in
`reports/rigor/2026-05-13/` and the per-model summaries in `reports/rigor/`.

| Gate | What it tested | What we found |
|------|----------------|---------------|
| G1 -- HP tuning | Used Optuna to search hyperparameters (38+38 trials for LSTM, 50 for XGB) | LSTM best val F1 0.637; XGB best val F1 0.691 -- both tuned before seed sweeps |
| G2 -- Seed variance | Ran 5 fixed seeds to check result stability | LSTM 0.599 +/- 0.025 (moderate variance); XGB 0.721 +/- 0.001 (very stable) |
| G3 -- Loss function | Compared focal loss (focuses training on hard examples) against weighted cross-entropy | All focal gamma values hurt F1 (worst: -0.109 at gamma=3); weighted cross-entropy kept |
| G4 -- Decision threshold | Searched for a confidence cutoff better than argmax on the val PR curve | Threshold tuning hurt both models (LSTM -0.006, XGB -0.043); argmax kept |
| G5 -- Feature importance | Used SHAP to rank which hand-engineered features drive XGBoost predictions | 60-bar return and gap geometry dominate; vol_spike near-zero and prunable |
| G6 -- Bull/bear asymmetry | Measured whether bull-FVG and bear-FVG F1 differ systematically | Bull slightly ahead (+0.035 gap) but not consistent across all seeds; no architectural change needed |
| G7 -- Window size | Swept lookback window W over {30, 45, 60, 90, 120} bars | W=60 balances F1 and test-set coverage (86 independent blocks); W=90 slightly better F1 but only 57 blocks |
| G8 -- Data scaling | Compared extended (2016--2025) vs shorter (2018--2024) training set | Extended data marginally hurt both models (max delta -0.013); 2016--2025 retained |
| G9 -- Regularisation | Ablated dropout and L2 weight decay individually and together | Head dropout is the primary regulariser; removing both hurts most (-0.020); full regularisation kept |
| G10 -- Bootstrap CI | Estimated 95% confidence intervals via block bootstrap (1000 resamples, block=60) | LSTM CI [0.563, 0.635]; XGB CI [0.671, 0.761] -- wide enough that model ranking is real but margins are not |

The CNN-LSTM model was subsequently run through the same sprint (G1, G2, G4, G6, G7, G9, G10)
and became the carrier model (0.639 +/- 0.013, 95% CI [0.603, 0.674]).

---

## Summary table

| Rigor measure | What it guards against | Where it is implemented |
|---|---|---|
| Temporal-only split | Training on future data | `src/data/split.py::temporal_split` |
| N+2 label index, shift(1) features | Lookahead in labels | `src/data/labels/valid_fvg.py` |
| Macro-F1 + per-class F1 | Accuracy masking class imbalance | `scripts/rigor/multiseed_run.py` |
| Weighted cross-entropy loss | Gradient signal dominated by majority class | `src/training/` WeightedCE |
| 5-seed sweep + mean +/- std | Single lucky run | `scripts/rigor/multiseed_run.py` |
| Block bootstrap CI (block=60, 1000 iter) | Point estimate with no uncertainty range | `scripts/rigor/bootstrap_ci_multiseed.py` |
| Pre-registered decision rule | Post-hoc metric cherry-picking | `.nb/plan/09-Jun-26/`, `reports/rigor/07-Jun-26/dl_diagnosis_decision.md` |
| Honest negative reporting | Hiding failures | `reports/rigor/07-Jun-26/dl_diagnosis_decision.md`, this doc Section 5 |
| Reproducible seeds + versioned YAML configs | Results that cannot be recreated | `src/training/train_utils.py`, `experiments/` |
