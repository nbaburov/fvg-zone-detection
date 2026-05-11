# Phase 4 Step 0 — XGBoost Baseline Plan

> **Plan type:** Feature implementation plan — Stage 0 of Phase 4 Modelling.
> **Work type:** New build (XGBoost classifier on SPY H1 ValidFVG labels).
> **Recommended mode:** /nb:build standard (single branch). All phases sequential.
> **Test strategy:** TDD (pytest) for feature engineering + causality. ML evaluation on held-out test split only.
> **Date:** 11 May 2026
> **Deadline constraint:** Status Update 1 = 17 May (6 days). This plan must be buildable in ≤2 days.

---

## Context and locked decisions

**Labeller in use:** `ValidFVGLabeller` with defaults:
- `bos_lookback=50`, `require_crit3=False` (crit3 OFF — loose)
- Criterion 2 uses loose variant (`crit2_loose_bull/bear`)
- Label index: N+2

**Dataset already on disk:**
- `data/processed/spy_h1.parquet` — full 14,686 candles (2018–2026-05-07), OHLCV + session_type + raw_label + label
- Splits already written: `spy_h1_train.parquet`, `spy_h1_val.parquet`, `spy_h1_test.parquet`
- Class weights: `data/processed/class_weights.json` = `{0: 0.217, 1: 1.097, 2: 1.686}`

**Empirical label counts (from persisted parquet):**
| Split | Rows | Positives | Positive rate |
|-------|------|-----------|--------------|
| Train | 7,056 | 1,738 | 24.6% |
| Val | 1,757 | 470 | 26.8% |
| Test | 3,514 | 897 | 25.5% |

Note: The 3.2% positive rate referenced in the brief was a projection for all-6-criteria strict mode. Actual deployed defaults (crit3 OFF, crit2 loose) produce 25% positive rate. This is a **moderate imbalance** regime, not extreme. Class weights [0.217, 1.097, 1.686] reflect this.

**Temporal split (locked):**
- Train: 2018-01-02 – 2021-12-31
- Val: 2022-01-03 – 2022-12-30
- Test: 2023-01-03 – 2024-12-31

**Window:** 60 candles, stride=1 for train. Stride=60 for val/test.

**Primary metric:** F1 on minority class (bull FVG + bear FVG). Macro-F1 as headline. Accuracy not reported.

**Stop point:** After XGBoost baseline run and evaluation. No DL code in this plan. No hyperparameter search — one default-config run only.

**Seed:** 42 everywhere.

---

## What XGBoost requires vs the DL pipeline

The existing pipeline emits `SMCWindowDataset` objects — 3D tensors `(B, 60, 5)` for sequential models. XGBoost needs 2D tabular input `(n_samples, n_features)`. We do NOT flatten 60×5=300 features (high-dim, overfits on 7k samples). Instead we engineer ~40 summary features per window.

**Feature engineering lives in `src/features/window_features.py`** — a new module distinct from the windowing code. It receives a DataFrame split and emits a 2D feature matrix + label vector. No tensor conversion needed.

---

## Feature specification

All features are computed from bars `[t-59 .. t]` where t is the label bar index. Label is at t. This is always causal — no bar past t is used.

### Group A: Price return features (8 features)

| Feature | Formula | Window |
|---------|---------|--------|
| `ret_1` | `(close[t] - close[t-1]) / close[t-1]` | 1 bar |
| `ret_5` | `(close[t] - close[t-5]) / close[t-5]` | 5 bars |
| `ret_10` | `(close[t] - close[t-10]) / close[t-10]` | 10 bars |
| `ret_20` | `(close[t] - close[t-20]) / close[t-20]` | 20 bars |
| `ret_60` | `(close[t] - close[t-59]) / close[t-59]` | Full window |
| `high_low_range` | `(high[t] - low[t]) / close[t]` | Bar t only |
| `body_ratio` | `abs(close[t] - open[t]) / (high[t] - low[t] + 1e-9)` | Bar t only |
| `upper_wick` | `(high[t] - max(open[t], close[t])) / (high[t] - low[t] + 1e-9)` | Bar t only |

### Group B: Volatility features (5 features)

| Feature | Formula | Window |
|---------|---------|--------|
| `atr_14` | `mean(|close[i] - close[i-1]|, i in [t-13..t])` | 14 bars |
| `atr_28` | `mean(|close[i] - close[i-1]|, i in [t-27..t])` | 28 bars |
| `vol_ratio_14_28` | `atr_14 / (atr_28 + 1e-9)` | Ratio |
| `range_60` | `(max(high[t-59..t]) - min(low[t-59..t])) / close[t]` | Full window |
| `range_20` | `(max(high[t-19..t]) - min(low[t-19..t])) / close[t]` | 20 bars |

### Group C: Momentum / trend features (7 features)

| Feature | Formula | Window |
|---------|---------|--------|
| `rsi_14` | Wilder RSI over 14 bars: `100 - 100 / (1 + avg_gain / avg_loss)` | 14 bars |
| `macd_signal` | `ema_12 - ema_26` (simplified: use pandas ewm with span 12/26) | 60 bars |
| `macd_norm` | `macd_signal / (close[t] + 1e-9)` | Normalised |
| `pos_in_range` | `(close[t] - min(low[t-59..t])) / (range_60 * close[t] + 1e-9)` | Position 0–1 |
| `trend_slope` | `polyfit(range(60), close[t-59..t], 1)[0] / close[t]` | Linear slope |
| `above_ma20` | `1 if close[t] > mean(close[t-19..t]) else 0` | Binary |
| `above_ma50` | `1 if close[t] > mean(close[t-49..t]) else 0` | Binary |

### Group D: Volume features (5 features)

| Feature | Formula | Window |
|---------|---------|--------|
| `vol_zscore_5` | `(volume[t] - mean(volume[t-4..t])) / (std(volume[t-4..t]) + 1)` | 5 bars |
| `vol_zscore_20` | `(volume[t] - mean(volume[t-19..t])) / (std(volume[t-19..t]) + 1)` | 20 bars |
| `vol_trend` | `mean(volume[t-4..t]) / (mean(volume[t-9..t-5]) + 1)` | 5-bar vs prev-5 ratio |
| `vol_spike` | `1 if volume[t] > 2 * mean(volume[t-19..t]) else 0` | Binary |
| `vol_body_corr` | `(volume[t] * abs(close[t] - open[t])) / (volume[t] + 1)` | Size × volume |

### Group E: FVG-locality features (10 features — structural priors)

These features directly encode the 3-candle FVG geometry. Since label is at t=N+2, the 3-candle pattern sits at bars t-3, t-2, t-1 within the window (N-1=t-3, N=t-2, N+1=t-1, N+2=t).

| Feature | Formula | Bars |
|---------|---------|------|
| `gap_bull` | `max(0, low[t-1] - high[t-3])` — bullish gap size | t-3, t-1 |
| `gap_bear` | `max(0, low[t-3] - high[t-1])` — bearish gap size | t-3, t-1 |
| `gap_norm_bull` | `gap_bull / (close[t] + 1e-9)` | Normalised |
| `gap_norm_bear` | `gap_bear / (close[t] + 1e-9)` | Normalised |
| `mid_body_bull` | `1 if close[t-2] > open[t-2] else 0` — middle candle bullish body | t-2 |
| `mid_body_bear` | `1 if close[t-2] < open[t-2] else 0` — middle candle bearish body | t-2 |
| `mid_range_norm` | `(high[t-2] - low[t-2]) / (close[t] + 1e-9)` — middle candle range | t-2 |
| `react_in_gap_bull` | `1 if gap_bull > 0 and low[t-1] <= close[t] <= high[t-3] else 0` | t, reaction candle position |
| `react_in_gap_bear` | `1 if gap_bear > 0 and low[t-3] <= close[t] <= high[t-1] else 0` | t |
| `prior_trend_5` | `mean(close[t-7..t-3]) - mean(close[t-12..t-8])` / `close[t]` — trend before pattern | pre-pattern |

**Total features: 8 + 5 + 7 + 5 + 10 = 35 features.** Well within the "< 50 for 14k samples" guideline from the risk register.

**Feature causality invariant:** All features reference only bars `[t-59..t]`. No bar `t+1` is used. Enforced by a mandatory pytest fixture.

---

## Component breakdown

### New files to create

```
src/features/
  __init__.py           — empty
  window_features.py    — feature engineering: DataFrame → (X: ndarray, y: ndarray)

src/models/
  __init__.py           — empty
  xgboost_baseline.py   — XGBClassifier wrapper: train, predict, save/load

scripts/
  persist_labels.py     — idempotent: check if spy_h1_labeled.parquet exists + is current, run pipeline if not
  train_xgboost.py      — end-to-end: load splits → features → train → eval → save

tests/models/
  __init__.py
  test_xgboost_baseline.py  — unit tests for model wrapper
tests/features/
  __init__.py
  test_window_features.py   — causality test + feature value tests
```

### `src/features/window_features.py`

**Inputs:** `df: pd.DataFrame` (must have columns `open, high, low, close, volume, label`), `window_size: int = 60`.

**Output:** `(X: np.ndarray shape (n_samples, 35), y: np.ndarray shape (n_samples,) int)`

**Window construction:** Slide over the DataFrame with stride=1 for train (stride=`window_size` for val/test — configurable). For each window ending at index t (inclusive), extract the 60-bar slice `df.iloc[t-59:t+1]` and compute all 35 features. Label = `df["label"].iloc[t]`.

**Edge handling:** First 59 rows produce no windows (no full lookback). Implemented via Python slice — if `df.iloc[t-59:t+1]` has < 60 rows, skip (should not happen after warmup).

**API:**
```python
def extract_window_features(
    df: pd.DataFrame,
    window_size: int = 60,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y).
    X shape: (n_windows, 35) float32.
    y shape: (n_windows,) int (0/1/2 — encoded from raw label via encoded_map).
    No NaN in output. All features causal.
    """
```

**RSI implementation (no pandas_ta):**
```python
def _rsi_14(close_window: np.ndarray) -> float:
    """Wilder RSI on last 14 bars of close_window."""
    delta = np.diff(close_window[-15:])   # 14 diffs from 15 bars
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
```

**MACD implementation:**
```python
def _macd(close_window: np.ndarray) -> float:
    """EMA(12) - EMA(26) on close_window (length 60)."""
    # pandas ewm on numpy array via pd.Series
    s = pd.Series(close_window)
    ema12 = s.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = s.ewm(span=26, adjust=False).mean().iloc[-1]
    return float(ema12 - ema26)
```

**Trend slope:**
```python
def _trend_slope(close_window: np.ndarray) -> float:
    x = np.arange(len(close_window), dtype=float)
    slope = np.polyfit(x, close_window, 1)[0]
    return float(slope / (close_window[-1] + 1e-9))
```

**NaN guard:** Any window producing a NaN feature (e.g., zero-volume bar yields 0/0) fills with 0.0. Assert no NaN in final X matrix.

### `src/models/xgboost_baseline.py`

```python
class XGBoostFVGClassifier:
    """Thin wrapper around XGBClassifier for FVG 3-class prediction."""

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": ["mlogloss", "merror"],
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",   # fastest on CPU; no GPU needed
        "early_stopping_rounds": 30,
    }

    def __init__(self, params: dict | None = None): ...
    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None): ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def save(self, path: Path): ...
    @classmethod
    def load(cls, path: Path) -> "XGBoostFVGClassifier": ...
```

**Class weight application:** `sample_weight` array passed to `fit()`. Computed from class weights tensor: `sample_weight[i] = class_weights[y_train[i]]`. This is the XGBoost equivalent of `scale_pos_weight` for multiclass.

**Early stopping:** `early_stopping_rounds=30` with val mlogloss. XGBoost built-in via `eval_set`.

**Save format:** `model.save_model(path)` (XGBoost native format, `.ubj` extension). Also save a sidecar `<path>.meta.json` with all hyperparams, git_sha, python/xgboost version, timestamp, final n_estimators used.

### `scripts/persist_labels.py`

**Purpose:** Ensure `data/processed/spy_h1_labeled.parquet` exists and is up-to-date. This is the canonical labeled file for Phase 4. XGBoost reads from here, not from `spy_h1.parquet` (which is also labeled but has a different naming convention per the locked decision).

**Logic:**
```
1. Check if data/processed/spy_h1_labeled.parquet exists.
2. If exists: check modification time vs src/data/labels/valid_fvg.py modification time.
   If parquet is newer → print "up-to-date, skipping" and exit 0.
3. If missing or stale: load spy_h1.parquet (already written by pipeline.py).
   The pipeline.py `build_pipeline()` already writes spy_h1.parquet with label column.
   So this script just symlinks or copies spy_h1.parquet → spy_h1_labeled.parquet.
   (If spy_h1.parquet already has the label column — confirmed above — this is trivial.)
4. Print positive counts and exit.
```

**Idempotent:** Running twice produces same result. Uses `pathlib.Path.stat().st_mtime` for staleness check.

**Note:** Since `spy_h1.parquet` already contains the `label` column (confirmed: label counts {0: 10985, 1: 2186, 2: 1515} on the full dataset), the persist_labels step is mostly a copy/symlink + staleness-check wrapper. No re-labelling needed.

### `scripts/train_xgboost.py`

```
Usage: python scripts/train_xgboost.py [--seed 42] [--output-dir checkpoints/xgboost]

Steps:
1. Set seed=42 globally.
2. Log git_sha, python version, xgboost version, timestamp.
3. Load spy_h1_train.parquet, spy_h1_val.parquet, spy_h1_test.parquet.
4. Extract features: X_train, y_train = extract_window_features(train_df, stride=1)
5. Extract features: X_val, y_val = extract_window_features(val_df, stride=60)
6. Extract features: X_test, y_test = extract_window_features(test_df, stride=60)
7. Compute sample_weight from class_weights.json.
8. Instantiate XGBoostFVGClassifier with DEFAULT_PARAMS.
9. Fit with eval_set=[(X_val, y_val)], early stopping.
10. On val set: classification_report, confusion_matrix. Print.
11. On test set: classification_report, confusion_matrix. Print.
12. Save model to checkpoints/xgboost/xgb_seed42.ubj + meta.json.
13. Write evaluation log to .nb-suite/test-logs/11-May-26/xgboost-baseline.md.
```

**Sample count sanity check (step before training):**
- Assert len(y_test) > 0 and sum(y_test != 0) >= 50. If < 50 positives in test, warn and continue (don't abort — just flag in log).

---

## Sequential build order

### Phase 1: Feature engineering module (src/features/)

**Model:** haiku
**Files:**
- Create: `src/features/__init__.py`
- Create: `src/features/window_features.py`
- Create: `tests/features/__init__.py`
- Create: `tests/features/test_window_features.py`

**Acceptance criteria:**
- `extract_window_features(df_100rows, stride=1)` returns X shape `(41, 35)` (100-60=40 windows + 1 for inclusive end = 41). Verify exact count.
- `extract_window_features(df_100rows, stride=60)` returns X shape `(1, 35)` (100-60=40, integer division 40//60 = 0... actually 1 window). Confirm off-by-one does not exist.
- No NaN or inf in X for real H1 data (load 500 rows from spy_h1_train.parquet as fixture).
- **Causality test:** Mutate row `t+1` of df, assert X[t-60] row is unchanged. See full test spec below.
- FVG-locality features: for a hand-crafted 60-row fixture with a known bullish FVG at bars 56, 57, 58 (0-indexed), assert `gap_bull > 0` and `gap_bear == 0` at window t=59 (label bar).
- RSI range check: `0 <= rsi_14 <= 100` for all windows.
- `above_ma20` is binary (0 or 1).
- All 35 features present in correct column order.

**Tests must run without network access or large parquet file** — use a 100-row synthetic fixture generated from numpy random with valid OHLCV structure (high > max(open, close), low < min(open, close), volume > 0).

**Effort:** 2–3 hours

---

### Phase 2: XGBoost model wrapper (src/models/)

**Model:** haiku
**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/xgboost_baseline.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_xgboost_baseline.py`

**Acceptance criteria:**
- `XGBoostFVGClassifier()` initialises without error.
- `fit(X_train_small, y_train_small, X_val_small, y_val_small)` completes without error on a 200-sample synthetic dataset (balanced or imbalanced does not matter for unit test).
- `predict(X)` returns ndarray of int in `{0, 1, 2}`, shape `(n_samples,)`.
- `predict_proba(X)` returns ndarray shape `(n_samples, 3)`, rows sum to 1.0 (± 1e-6).
- `save(path)` writes `.ubj` file and `.ubj.meta.json` sidecar. Both exist on disk.
- `load(path)` round-trips: model predictions on X match before-save predictions exactly.
- `meta.json` contains keys: `params`, `git_sha`, `python_version`, `xgboost_version`, `timestamp`, `n_estimators_used`.
- `sample_weight` parameter accepted in `fit()` without error.

**Effort:** 1–2 hours

---

### Phase 3: Persist-labels script (scripts/persist_labels.py)

**Model:** haiku
**Files:**
- Create: `scripts/persist_labels.py`

**Acceptance criteria:**
- Running script when `spy_h1.parquet` exists: creates `spy_h1_labeled.parquet` (or confirms it is current).
- Running script twice is idempotent (second run prints "up-to-date").
- Script prints: total rows, positive counts per class, positive rate.
- Exit code 0 on success, 1 on missing source parquet.

**Effort:** 30 minutes

---

### Phase 4: Training script + full end-to-end run (scripts/train_xgboost.py)

**Model:** sonnet (orchestration + eval writeup)
**Files:**
- Create: `scripts/train_xgboost.py`
- Create: `checkpoints/xgboost/` directory (via mkdir_p in script)
- Create: `.nb-suite/test-logs/11-May-26/xgboost-baseline.md`

**Acceptance criteria:**
- Script runs end-to-end without error: `python scripts/train_xgboost.py`
- Training completes (early stopping or max_epochs).
- Evaluation on test set printed to stdout and written to `.nb-suite/test-logs/`.
- Evaluation log contains: classification_report (per-class precision, recall, F1; macro; weighted), confusion matrix (3×3), sample counts per class in test set, training hyperparameters, git_sha, timestamp.
- Positive count sanity check logged: confirm test set has >= 50 positives.
- Model saved to `checkpoints/xgboost/xgb_seed42.ubj`.

**Effort:** 1 hour (script) + training run (~10 min) + eval writeup (~30 min)

---

### Phase 5: Tests and leakage gate

**Model:** haiku
**Files:**
- Extend: `tests/features/test_window_features.py` — add the mandatory causality test fixture

**Mandatory causality test (must pass before Phase 4 script is considered done):**

```python
def test_features_no_future_bar_leakage(spy_h1_train_fixture):
    """
    Mutate a bar at index K. Assert all windows ending before bar K are unchanged.
    """
    df = spy_h1_train_fixture.copy()  # 500-row fixture from train parquet
    K = 300  # mutation point (middle of fixture)

    X_before, _ = extract_window_features(df, stride=1)

    # Corrupt bar K+1 (future relative to windows ending at K or earlier)
    df_mutated = df.copy()
    df_mutated.iloc[K + 1] = [99999.0, 99999.0, 0.01, 99999.0, 9999999.0, 0, 0, 0]

    X_after, _ = extract_window_features(df_mutated, stride=1)

    # Windows 0..K-59 end at bars 59..K — they should be unchanged
    # Window at index j ends at bar j+59. Windows where j+59 < K are unaffected.
    safe_count = max(0, K - 60)
    if safe_count > 0:
        np.testing.assert_array_equal(
            X_before[:safe_count], X_after[:safe_count],
            err_msg="Feature values changed for windows not touching the mutated bar — causality violated"
        )
```

**Acceptance:** Test passes with `pytest tests/features/test_window_features.py::test_features_no_future_bar_leakage`.

**Effort:** 30 minutes

---

## Test strategy

### Unit tests (TDD)

| Test file | Tests |
|-----------|-------|
| `tests/features/test_window_features.py` | Shape assertions, no NaN, causality fixture, FVG-locality correctness, RSI bounds, binary feature values |
| `tests/models/test_xgboost_baseline.py` | Fit runs, predict shape, predict_proba sums to 1, save/load round-trip, meta.json keys |

### ML evaluation (not unit test — run from script)

| Check | Where |
|-------|-------|
| Positive count in test ≥ 50 | `train_xgboost.py` sanity check |
| Per-class F1 on test | `xgboost-baseline.md` evaluation log |
| Confusion matrix | Same |
| Val mlogloss training curve (first/last N) | Same |
| Comparison baseline: naive predictor (always predict 0) | Computed inline — majority baseline F1 for comparison |

### Sanity checks in script

1. **Label distribution check:** Before training, print label counts for train, val, test. If any class has 0 samples in val, log warning (early stopping on val mlogloss is still valid but minority-F1 check will return NaN for that class — expected with 29-window val but our val has 470 positives so this is not a concern here).
2. **Feature matrix sanity:** Assert `np.isfinite(X_train).all()` and same for val/test. If fails, print the offending feature names and row index.
3. **Positive count:** `sum(y_test != 0)` must be >= 50. With 897 positives in test and stride=60 → test = 3514/60 ≈ 58 windows, of which ~25% positive ≈ 14–15. **Wait — this is wrong.** See risk below.

---

## Window count calculation (critical)

With stride=60 and 3,514 test rows:
- Total windows = floor((3514 - 60) / 60) + 1 = floor(3454/60) + 1 = 57 + 1 = 58 windows
- Positive windows ≈ 25% × 58 ≈ 14–15 windows

**This is below the ≥50 threshold stated in requirements.** The test set window count is structurally small because stride=60 was chosen for evaluation (non-overlapping). The label count (897 positives in 3514 rows) does NOT translate to 897 positive windows — it translates to ~58 windows total.

**Resolution:** Use stride=1 for evaluation as well (same as train). This gives 3,514 - 60 + 1 = 3,455 test windows, ~25% positive = ~863 positive windows. F1 statistics are then reliable.

**Locked decision for this plan:** Use `stride=1` for all splits in the XGBoost feature extraction. This gives:
- Train: 7,056 - 60 + 1 = 6,997 windows → ~1,700 positive
- Val: 1,757 - 60 + 1 = 1,698 windows → ~425 positive
- Test: 3,514 - 60 + 1 = 3,455 windows → ~863 positive

Note: overlapping windows in train/val/test are expected for XGBoost evaluation (this is label-preserving, not i.i.d. — we report test F1 as the measure, not per-window accuracy). The window overlap is documented in the eval log.

**Update `scripts/train_xgboost.py` accordingly:** Call `extract_window_features(df, stride=1)` for all splits. Positive count sanity check becomes: `sum(y_test != 0) >= 200` (generous lower bound).

---

## Hyperparameter defaults (final)

```python
DEFAULT_PARAMS = {
    "n_estimators": 300,          # maximum trees; early stopping may reduce this
    "max_depth": 4,               # shallow — reduces overfit on ~7k samples
    "learning_rate": 0.05,        # conservative; literature default for small datasets
    "min_child_weight": 5,        # regularises leaf minimum sum of weights
    "subsample": 0.8,             # row sampling per tree
    "colsample_bytree": 0.8,      # feature sampling per tree
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": ["mlogloss", "merror"],
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",        # fastest on CPU/Apple Silicon; no CUDA required
    "early_stopping_rounds": 30,  # stop if val mlogloss doesn't improve for 30 rounds
}
```

**Justification:**
- `max_depth=4`: standard for financial tabular data at this scale; Fischer & Krauss (2018) used shallow trees for analogous tasks. Prevents memorizing the ~7k windows.
- `min_child_weight=5`: forces each leaf to cover at least 5 samples' worth of weight. With 25% positive rate, this is ~20 actual samples per leaf — prevents micro-splits on rare patterns.
- `subsample=0.8`, `colsample_bytree=0.8`: standard regularisation defaults from Chen & Guestrin (2016).
- `learning_rate=0.05`: conservative; fast enough for 300 trees, slow enough to use early stopping meaningfully.
- `early_stopping_rounds=30`: with lr=0.05 and 300 trees, 30 rounds = 1.5 LR steps of patience — reasonable.

**NO tuning in this plan.** These are baseline defaults. Hyperparameter search is Phase 4.1 territory.

---

## Evaluation output format

The evaluation log at `.nb-suite/test-logs/11-May-26/xgboost-baseline.md` must contain:

```markdown
# XGBoost Baseline Evaluation — Phase 4 Step 0

**Date:** 2026-05-11
**Git SHA:** <hash>
**Python:** <version>
**XGBoost:** <version>
**Seed:** 42

## Dataset summary
| Split | Windows | Positives | Positive rate |
...

## Hyperparameters
<table of DEFAULT_PARAMS>

## Training
- Final n_estimators used: <N> (early stopping at round <N>)
- Best val mlogloss: <value>
- Val merror at best: <value>

## Naive baseline (always-majority)
F1 macro: <value>   ← majority-class predictor for reference

## Validation results
<classification_report>
<3×3 confusion matrix>

## Test results (primary)
<classification_report>
<3×3 confusion matrix>
- Macro-F1: <value>
- Minority class F1 (bull): <value>
- Minority class F1 (bear): <value>
- Positive count sanity: <n_pos> windows (>= 200 expected)

## Comparison to naive baseline
<table: metric, naive, xgboost, delta>

## Phase 4 handoff
- XGBoost test macro-F1: <value> — this is the floor DL models must beat.
- Model saved at: checkpoints/xgboost/xgb_seed42.ubj
```

---

## Risk register

| Risk | Likelihood | Blast radius | Mitigation |
|------|------------|--------------|------------|
| Feature engineering has subtle causality leak | Medium | All downstream DL comparisons invalid | Mandatory causality test in Phase 5. Must pass before eval is reported. |
| XGBoost overfits on ~7k windows with stride=1 | Medium | Inflated train F1, low test F1 | Early stopping on val mlogloss. `min_child_weight=5`, `max_depth=4`. Report train vs test F1 gap in log. |
| Test positive count too low (stride=1 gives 863 positives — healthy) | Low | F1 statistic unreliable | Resolved by stride=1 decision. If for some reason count < 200, flag in log and continue. |
| XGBoost not installed in venv | Low | Training blocked | Check: `pip install xgboost`. Add to install deps in CLAUDE.md if missing. |
| Feature NaN from zero-volume bars or price collapse | Low | Training error | `.fillna(0.0)` on X before training. Assert no NaN after fill. |
| 3.2% vs 25% positive rate discrepancy vs prompt | Already resolved | — | Prompt estimate was for strict 6-criteria mode; actual deployed defaults (crit3 OFF, crit2 loose) give 25%. Locked at current defaults per task brief. |

---

## Pre-mortem

Imagining this build failed:

1. **Feature causality leak was subtle.** The FVG-locality features used `close[t+1]` (reaction candle bar N+2) but also used `low[t+1]` when computing gap features relative to bar t — crossing the boundary. The causality test fixture was not tight enough to catch this because the synthetic fixture had uniform prices. Fix: use real H1 data slice in causality test with actual price variation.

2. **XGBoost macro-F1 was inflated by overlapping windows (stride=1).** With stride=1, adjacent windows share 59 of 60 candles. The model learned the label for window t from the very similar window t-1 in the training set. This is not time leakage (train and test are temporally separated) but is misleading as "effective sample count." Reported F1 looks good but the model may fail on genuinely unseen patterns. Mitigation: report both stride=1 and stride=60 F1 in the eval log as a diagnostic. The headline number is stride=1 (consistent with what DL models will train on).

3. **`min_child_weight=5` was too loose.** With 25% positive rate and class weights [0.217, 1.097, 1.686], the effective weight-sum per leaf requirement was satisfied by very few majority-class samples. Model grew deep trees and overfit. Fix: increase to `min_child_weight=10` and note in Phase 4.1 tuning plan.

4. **Script failed at feature extraction on boundary windows.** Windows at the very start of val_df or test_df had fewer than 60 prior bars within that split. Should have loaded val with 60 rows of train appended as context (warm-up buffer), then stripped those from X/y before training. Not needed for XGBoost (no temporal state) but is a gotcha for DL models — document in Phase 4 LSTM plan.

---

## Out of scope (this plan)

- Hyperparameter tuning (Phase 4.1 or Phase 5 territory)
- SHAP feature importance analysis (optional Phase 5 addition — not here)
- Any DL model code
- Focal loss
- Multi-seed runs (single seed=42 baseline only)
- Block-bootstrap CI on test (DL evaluation territory per R7 — XGBoost baseline uses point estimate F1)
- `src/training/` module (DL training infrastructure — not needed for XGBoost)
- `src/eval/` module (DL evaluation infrastructure — not needed here; use sklearn directly)

---

## Phase 4 LSTM handoff

After this plan is built and the eval log is written:

**The XGBoost test macro-F1 is the floor.** Every DL architecture must beat it or the DL section is academically weak. The evaluation log's test macro-F1 and per-class F1 become the comparison baseline in the final notebook.

**Files produced by this plan:**
```
src/features/__init__.py
src/features/window_features.py
src/models/__init__.py
src/models/xgboost_baseline.py
scripts/persist_labels.py
scripts/train_xgboost.py
tests/features/__init__.py
tests/features/test_window_features.py
tests/models/__init__.py
tests/models/test_xgboost_baseline.py
checkpoints/xgboost/xgb_seed42.ubj
checkpoints/xgboost/xgb_seed42.ubj.meta.json
.nb-suite/test-logs/11-May-26/xgboost-baseline.md
```

**Commands to run in order:**
```bash
# 0. Verify deps
pip show xgboost  # if missing: pip install xgboost

# 1. Run tests (TDD — before script)
pytest tests/features/ tests/models/ -v

# 2. Persist labels (confirm spy_h1_labeled.parquet)
python scripts/persist_labels.py

# 3. Train XGBoost baseline
python scripts/train_xgboost.py --seed 42

# 4. Verify eval log
cat .nb-suite/test-logs/11-May-26/xgboost-baseline.md
```

**Expected artifacts for LSTM plan:**
- `src/features/window_features.py` — reuse for DL feature baseline comparison (optional)
- `checkpoints/xgboost/xgb_seed42.ubj` — never re-trained; comparison fixed
- `.nb-suite/test-logs/11-May-26/xgboost-baseline.md` — read by LSTM plan for baseline F1 targets

---

## Confidence

**High** that the feature specification is causal and correct. All 35 features reference only bars ≤ t. The FVG-locality features directly encode the 3-candle pattern with correct bar index offsets (t-3, t-2, t-1, t for N-1, N, N+1, N+2 respectively).

**High** that the dataset is ready — `spy_h1.parquet` with label column exists on disk, splits are written, class weights are computed. No re-running of the pipeline is required.

**Medium** on expected XGBoost F1 — the 25% positive rate means the baseline is not extreme-imbalance. XGBoost may produce respectable minority F1. If it does, DL models need to show something extra (e.g., better calibration, direction accuracy) to justify the added complexity. That argument is Phase 5 territory.

**Medium** on whether stride=1 for evaluation is the right choice for XGBoost. Unlike DL, XGBoost does not have temporal state, so overlapping windows are genuinely i.i.d. features from its perspective. The F1 reported with stride=1 is the "dense evaluation" metric that matches the DL training distribution. This is a methodological choice — document it explicitly in the eval log.

## What I couldn't verify

- Whether `xgboost` is already installed in the project venv. Builder must check `pip show xgboost` before Phase 2.
- Whether `src/models/` directory already exists. Confirmed: it does NOT exist. Builder creates it.
- Exact window count arithmetic edge cases with the specific parquet row counts (7056, 1757, 3514). The formula is `n_windows = n_rows - window_size + 1` for stride=1. These are correct but must be asserted in tests.
- Whether the `data/gold_labels.csv` (75 candles, κ=1.0) from the git status is already re-annotated under 6-criteria or still the geometric gold labels — this does not affect the XGBoost plan either way.
