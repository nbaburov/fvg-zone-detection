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
| Train | 2016-01-01 to 2021-12-31 |
| Validate | 2022-01-01 to 2022-12-31 |
| Test | 2023-01-01 to 2025-12-31 |

The `temporal_split` function slices by timestamp inequality; it never shuffles rows. The function
also raises `ValueError` if any split returns zero bars, enforcing the boundary contract at
runtime.

**Label integrity (the N+2 rule).** Each label is attached to the candle at index N+2, the
reaction bar, the first bar at which all six FVG criteria are knowable from market history alone.
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
and persisted to `data/processed/class_weights_spy_h1.json` so they cannot accidentally include
validation or test label counts.

---

## 3. Not trusting one lucky run

A single training run with a single random seed can produce an unusually good or unusually
bad result by chance. We run every model with **five fixed seeds** (0, 17, 42, 123, 2024)
and report mean F1 plus sample standard deviation. If a model's mean looks good but its
standard deviation is large, the headline number is not reliable.

On top of the seed spread we compute **block bootstrap confidence intervals**. The procedure
is: take the test-set predictions, resample blocks of 60 consecutive bars (rather than
individual bars, to preserve the autocorrelation structure of a time series, meaning nearby time points are related, not independent, so you cannot treat each bar as a fresh coin flip), compute the
mean-across-seeds F1 for each resample, repeat 1 000 times, and read off the 2.5th and 97.5th
percentiles. This gives a 95% CI that reflects both seed variance and the limited test-set
size (5 198 bars, effective_n = 86 non-overlapping blocks). Implementation:
`scripts/rigor/stats/bootstrap_ci_multiseed.py`.

The effective_n of 86 blocks is small. This means CIs are wide enough that some model pairs
overlap, for those pairs we call the result "statistically tied" rather than asserting a
winner (see Section 5).

---

## 4. Deciding the rule before seeing the results

To prevent unconsciously choosing the metric or comparison that makes results look best, the
decision rule was written down before running the Phase 4 multi-symbol experiment.

The pre-registered criterion (recorded before Phase 4 training):

- CNN-LSTM macro-F1 on the fixed SPY holdout **above 0.674** (the upper bound of the baseline
  95% CI) → data-bound hypothesis confirmed.
- CNN-LSTM F1 in **0.639 to 0.674** → grey zone, defer to bear_f1 direction.
- CNN-LSTM F1 **at or below 0.639** → hypothesis rejected.

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

**Transformer instability.** The Transformer reaches 0.58-0.61 on four of five seeds but
collapses to majority-only prediction on seed 42 (macro-F1 0.379, bear_f1 0.000). The
collapse recurred in independent learning-curve reruns with the same seeds, so the true
collapse rate is higher than 1-in-5. The mean F1 (0.548 ± 0.095) is therefore
run-dependent. We report the mean and std honestly and flag that the Transformer is
instability-bound, not data-bound. Multi-symbol data cannot fix an optimisation stability
problem; the indicated lever is architecture-level stabilisation (warmup schedule, lower
learning rate, pre-norm).

**xLSTM underfit.** xLSTM train F1 is 0.33 to 0.40 (below 0.85, the threshold for "model can
fit the data"). A model that cannot fit its training set will not benefit from more data. The
result (macro-F1 0.369 ± 0.007) is the ladder floor. We report it as an untuned result and
note that the 2025 literature suggests xLSTM underperforms on short financial sequences; the
gap to CNN-LSTM (0.27) is unlikely to close with tuning alone.

**XGBoost input mismatch.** XGBoost consumes 35 hand-engineered features; the four DL models
consume raw (60, 5) OHLCV windows. The XGB lead over CNN-LSTM (0.721 vs 0.639) is partly a
representation advantage, not a pure architecture comparison. The DL-vs-DL ranking (identical
input) is fair; the XGB row is context only.

**CI overlap for CNN-LSTM vs LSTM.** Their 95% CIs overlap (CNN-LSTM [0.603, 0.674], LSTM
[0.560, 0.628], overlap region 0.603-0.628). Strictly these two are statistically tied on
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
python scripts/rigor/eval/multiseed_run.py \
    --model cnn_lstm \
    --config experiments/cnn_lstm_g1.yaml \
    --set "train.seeds=[0,17,42,123,2024]"

# Bootstrap CI from saved prediction files
python scripts/rigor/stats/bootstrap_ci_multiseed.py \
    --pred-dir reports/rigor/07-Jun-26 \
    --model cnn_lstm \
    --block-size 60 --n-iter 1000 \
    --output-dir reports/rigor/07-Jun-26/bootstrap
```

---

## Validation sprint (G1-G10)

In May 2026 we ran a ten-gate validation sprint on the LSTM and XGBoost models (2016-2025 data,
ValidFVG labels). Each gate answered one methodological question before advancing. The table below
summarises what each gate tested and what it found; full per-gate numbers are in
`reports/rigor/2026-05-13/` and the per-model summaries in `reports/rigor/`.

| Gate | What it tested | What we found |
|------|----------------|---------------|
| G1: HP tuning | Used Optuna (a tool that automatically tries many settings to find the best) to search hyperparameters (the model settings chosen before training, e.g. number of layers; 38+38 trials for LSTM, 50 for XGB) | LSTM best val F1 0.637; XGB best val F1 0.691, both tuned before seed sweeps (a sweep = running the same thing across a range of settings to see what changes) |
| G2: Seed variance | Ran 5 fixed seeds to check result stability | LSTM 0.599 ± 0.025 (moderate variance); XGB 0.721 ± 0.001 (very stable) |
| G3: Loss function | Compared focal loss (an alternative training objective that focuses on hard-to-classify and rare cases, rather than treating all examples equally) against weighted cross-entropy | All focal gamma values hurt F1 (worst: -0.109 at gamma=3); weighted cross-entropy kept |
| G4: Decision threshold | Searched for a confidence cutoff better than argmax (simply pick the highest-scoring class) on the val PR curve (precision-recall trade-off curve: higher recall means more FVGs found but more false alarms, and vice versa) | Threshold tuning hurt both models (LSTM -0.006, XGB -0.043); argmax kept |
| G5: Feature importance | Used SHAP (a method that scores how much each input feature influenced each prediction) to rank which hand-engineered features drive XGBoost predictions | 60-bar return and gap geometry dominate; vol_spike near-zero and prunable |
| G6: Bull/bear asymmetry | Measured whether bull-FVG and bear-FVG F1 differ systematically | Bull slightly ahead (+0.035 gap) but not consistent across all seeds; no architectural change needed |
| G7: Window size | Swept lookback window W over {30, 45, 60, 90, 120} bars | W=60 balances F1 and test-set coverage (86 independent blocks); W=90 slightly better F1 but only 57 blocks |
| G8: Data scaling | Compared extended (2016-2025) vs shorter (2018-2024) training set | Extended data marginally hurt both models (max delta -0.013); 2016-2025 retained |
| G9: Regularisation | Ablated dropout and L2 weight decay individually and together | Head dropout is the primary regulariser; removing both hurts most (-0.020); full regularisation kept |
| G10: Bootstrap CI | Estimated 95% confidence intervals via block bootstrap (1000 resamples, block=60) | LSTM CI [0.563, 0.635]; XGB CI [0.671, 0.761], wide enough that model ranking is real but margins are not |

The CNN-LSTM model was subsequently run through the same sprint (G1, G2, G4, G6, G7, G9, G10)
and became the carrier model (0.639 ± 0.013, 95% CI [0.603, 0.674]).

---

## 7. Lower-timeframe (5m / 15m) evaluation

**Date:** 11-15 Jun 2026. **Status: COMPLETE.** Un-tuned baseline ladder + carrier (CNN-LSTM) Optuna tune + 5-seed tuned retrain with bootstrap CIs, both timeframes. Tuned test results in §7.1 below.

### Why run lower-timeframe experiments?

The H1 multi-symbol experiment confirmed the task is data-bound: the two recurrent models (LSTM, CNN-LSTM) both improved when training data was multiplied ~4x, while the Transformer (instability-bound) did not. Resampling the existing 1-minute raw bars to 5m and 15m candles multiplies the labelled bar count further: roughly 4x (15m) and 12x (5m) relative to H1, without any new data collection. This is the cheapest remaining lever for the data-bound models.

### Same methodology, applied per timeframe

Every methodological safeguard from Sections 1-6 applies identically to the lower-TF experiments:

- **Temporal split unchanged.** Train 2016-2021, validate 2022, test 2023-2025: boundaries applied in the resampled timeframe's timestamp space. No future data can leak across the boundary.
- **Label integrity.** `ValidFVGLabeller` uses only `shift(1)`-lagged values; the N+2 indexing rule holds at any timeframe because it is defined in bar counts, not clock time. The pytest anti-lookahead fixture was re-verified against 5m and 15m label outputs.
- **Five fixed seeds,** same set [0, 17, 42, 123, 2024].
- **Primary metric: macro-F1** on the three-class output (none / bull-FVG / bear-FVG).
- **Pooled multi-symbol training** (SPY + QQQ + IWM + DIA), same as Phase 4.
- **Test set:** SPY-only fixed holdout, resampled to the same TF as the model under test.

Bootstrap CI (block-size 60, 1,000 resamples) will be added after the tune completes and the final 5-seed ladder runs.

### Un-tuned baseline results

| Architecture | 15m F1 | 5m F1 | H1 (reference) | Direction |
|---|---|---|---|---|
| CNN-LSTM | 0.641 ± 0.066 | **0.693 ± 0.006** | 0.675 | lifts at 5m |
| LSTM | 0.652 ± 0.003 | 0.670 ± 0.003 | 0.640 | monotone lift H1 < 15m < 5m |
| Transformer | 0.649 ± 0.022 | 0.615 ± 0.145 * | 0.577 | lifts on stable seeds; instability persists |
| XGBoost | 0.696 ± 0.001 | 0.654 ± 0.001 | **0.738** | degrades at lower TF |

\* Transformer 5m: one seed collapsed (F1 0.328). Four converged seeds reach ≈ 0.69.

CNN-LSTM 15m: seed 42 collapsed (F1 0.512); four stable seeds ≈ 0.674.

### Key findings

**The leaderboard inverts.** At H1, XGBoost (0.738) leads all DL models by a clear margin. At 5m, CNN-LSTM (0.693, un-tuned) beats XGBoost (0.654). This inversion is the headline result of the lower-TF experiments and is the clearest evidence yet that the task was data-starved (not capacity-starved) at hourly resolution.

**DL lifts with more data; XGBoost degrades.** LSTM improves at every step: H1 0.640, 15m 0.652, 5m 0.670. XGBoost moves in the opposite direction: 0.738, 0.696, 0.654. The feature-engineered GBT has 35 inputs designed around hourly price structure; at finer timeframes those features lose their intended meaning. The raw-OHLCV neural nets are not subject to the same mismatch.

**Transformer instability is independent of data volume.** At 5m (most data), the collapse pattern persists unchanged. The four stable seeds reach ≈ 0.69, showing the model can fit the task at this scale, but the optimisation failure mode is not resolved by more training examples. This is consistent with the H1 and multi-symbol findings.

**These are un-tuned lower bounds.** See §7.1 for the tuned carrier numbers; they confirm the lift and remove the seed collapse.

### 7.1 Tuned carrier results (CNN-LSTM): 15-Jun-26

The carrier (CNN-LSTM) was Optuna-tuned per timeframe (validation Macro-F1 best: 15m 0.741, 5m 0.770), then retrained 5 seeds at the tuned HP (window 60) and evaluated on the fixed SPY test set. 95% CI = moving-block bootstrap (block 60, 1000 resamples, detailed in section 3).

| Timeframe | Tuned TEST Macro-F1 | 95% CI | per-seed range | std | vs un-tuned |
|---|---|---|---|---|---|
| 15m | 0.691 ± 0.003 | [0.681, 0.702] | 0.686-0.696 | 0.003 | +0.050 (0.641 to 0.691) |
| **5m** | **0.713 ± 0.005** | **[0.708, 0.719]** | 0.705-0.720 | 0.005 | +0.020 (0.693 to 0.713) |

**Tuning removed the 15m seed-42 collapse.** Un-tuned 15m std was 0.066 (one seed at 0.512); tuned, all five seeds land in 0.686-0.696 (std 0.003). This is the cleanest demonstration that the collapse was an optimisation artifact, not a data limit. The right HP made it reproducible across seeds.

**Tuned 5m (0.713) is the project's best DL test result.** It exceeds tuned 15m (0.691) and every H1 DL model, reinforcing the data-bound reading. It still trails H1 XGBoost (0.738): the lower-timeframe + tuning levers close most of the DL-vs-GBM gap but do not overturn XGBoost as the single best detector on this task. Methodology is identical to the H1 rigor (same fixed SPY test, same bootstrap protocol), so the numbers are directly comparable across timeframes. Window sweep, learning curve, and SHAP were de-scoped for this pass (trimmed rigor); the tuned HP carried directly from each Optuna study's best trial.

### Failure modes (honest reporting)

**XGBoost input mismatch at lower TF.** The 35 engineered features include H1-oriented metrics (60-bar momentum windows, gap geometry ratios calibrated to hourly price swings). At 5m, these features are computed over 60 five-minute bars (5 hours of market time), which changes their interpretation and compresses many signals that were meaningful at H1. This is a representation mismatch, not a model architecture failure. A proper XGBoost 5m experiment would re-engineer features for the new timeframe. This experiment reuses the H1 feature set unchanged and is therefore a floor for what XGBoost could achieve at 5m.

**Seed collapse carries through.** The CNN-LSTM 15m and Transformer 5m stds (0.066 and 0.145) are dominated by single seed collapses, not broad instability. The four or five stable seeds in each case are tightly clustered. This means the mean ± std presentation overstates the risk for a deployment that does per-seed selection or ensemble averaging. It also means the un-tuned mean F1 is artificially depressed compared to a collapse-free run.

---

## Summary table

| Rigor measure | What it guards against | Where it is implemented |
|---|---|---|
| Temporal-only split | Training on future data | `src/data/split.py::temporal_split` |
| N+2 label index, shift(1) features | Lookahead in labels | `src/data/labels/valid_fvg.py` |
| Macro-F1 + per-class F1 | Accuracy masking class imbalance | `scripts/rigor/eval/multiseed_run.py` |
| Weighted cross-entropy loss | Gradient signal dominated by majority class | `src/training/` WeightedCE |
| 5-seed sweep + mean +/- std | Single lucky run | `scripts/rigor/eval/multiseed_run.py` |
| Block bootstrap CI (block=60, 1000 iter) | Point estimate with no uncertainty range | `scripts/rigor/stats/bootstrap_ci_multiseed.py` |
| Pre-registered decision rule | Post-hoc metric cherry-picking | `reports/rigor/07-Jun-26/dl_diagnosis_decision.md` |
| Honest negative reporting | Hiding failures | `reports/rigor/07-Jun-26/dl_diagnosis_decision.md`, this doc Section 5 |
| Reproducible seeds + versioned YAML configs | Results that cannot be recreated | `src/training/train_utils.py`, `experiments/` |
