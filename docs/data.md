# Data

This document covers everything about the data used in this project: where it comes from, how
it is cleaned and transformed, how labels are generated, and how it is prepared for model
training. The top of each section is written for a general reader; the precise technical
details follow for developers.

---

## 1. Source

The project uses hourly candlestick bars for US equity ETFs (SPY, QQQ, IWM, DIA) covering
2016 to 2025. A candlestick bar summarises all trades in a fixed time window as four
prices (open, high, low, close) plus total volume.

**Why Alpaca, not Yahoo Finance?** The project initially evaluated Yahoo Finance as a free
data source. Empirical testing on 8 May 2026 confirmed that Yahoo's hourly history is
hard-capped at 730 days (about 2 years). Nine years of history are required to train, validate,
and test on non-overlapping time periods. Alpaca Markets (free paper account) provides the
full history without that cap.

Data is fetched with `adjustment="raw"` (no split/dividend price adjustment) so that bar
prices are exactly what traded at the time. This is the correct choice for a pattern-detection
task: adjusted prices alter the raw geometry of the patterns the model is looking for.

---

## 2. Raw to H1: the processing pipeline

**Plain language.** Alpaca is queried for 1-minute bars. Those bars are filtered down to
regular stock-market hours, then aggregated into 1-hour bars. Finally, bars with obviously
bad data are removed.

**Step-by-step:**

1. **Download.** 1-minute bars are pulled for each symbol via `alpaca.data.StockBarsRequest`
   with `TimeFrame.Minute`. Results are cached locally as parquet (a compressed, column-oriented table file format) to avoid repeat API calls.

2. **RTH filter.** Only bars with a timestamp between 09:30 and 15:59 ET are kept (regular
   trading hours, not pre-market or after-hours data). This gives 390 minutes per
   full trading day.

   Why not use Alpaca's native hourly bars? Clock-aligned hourly bars (e.g., 09:00-10:00)
   mix pre-market and regular-session prints. The custom resample anchored at 09:30 avoids
   this.

3. **Resample to H1.** The filtered minute bars are resampled to 1-hour bars anchored at
   09:30 ET:
   ```
   resample("1h", closed="left", label="left", offset="30min")
   ```
   `closed="left"` and `label="left"` mean each bar is labelled by its opening minute and
   includes all trades up to (but not including) the next hour mark. The 09:30 offset shifts
   the grid so bars fall at 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30. That's seven bars
   per full session.

4. **Quality filter.** Bars are dropped if any of the following is true:
   - Volume is zero or negative
   - Any OHLC price is zero or negative
   - Low > high (impossible bar)
   - Low > min(open, close) or high < max(open, close) (OHLC integrity violation)

5. **Session tagging.** Each bar receives a `session_type` column: `"full"` for normal
   sessions, `"half"` for NYSE early-close days (e.g., day before Thanksgiving). This is
   determined using the `exchange_calendars` library's NYSE calendar.

**Sanity gate.** For any pull covering more than 365 days, the pipeline asserts that at
least 5,000 H1 bars were returned. Fewer bars indicate a silent data error.

---

## 3. The FVG label: what it is and how it is generated

### What is an FVG?

A **Fair Value Gap (FVG)** is a price pattern involving three consecutive candles where the
outer two candles leave a gap that the middle candle does not fill. The gap is a zone the
market skipped through quickly. Smart Money Concepts (SMC) theory holds that price tends to
return to these zones later, making them meaningful for trading decisions.

- **Bullish FVG:** the high of candle N-1 is strictly below the low of candle N+1, and
  candle N has a bullish body (close > open). The zone spans from `high[N-1]` to `low[N+1]`.
- **Bearish FVG:** the low of candle N-1 is strictly above the high of candle N+1, and
  candle N has a bearish body (close < open). The zone spans from `low[N+1]` to `high[N-1]`.

### Why "Valid FVG"?

A bare geometric FVG (just the gap) occurs often and is noisy. This project uses a stricter
definition ("Valid FVG") with five criteria applied simultaneously. Only FVGs passing all
five are labelled positive. This reduces noise and produces a label that is more meaningful
for training.

### The five criteria (implemented in `ValidFVGLabeller`)

All five are evaluated at candle N (the FVG middle bar). No future data beyond N+2 is used.

| # | Name | Bullish condition | Bearish condition |
|---|------|-------------------|-------------------|
| 1 | Geometric gap | `high[N-1] < low[N+1]` AND `close[N] > open[N]` | `low[N-1] > high[N+1]` AND `close[N] < open[N]` |
| 2 | Reaction (loose) | `close[N+2] >= high[N-1]` (price stays above zone bottom) | `close[N+2] <= low[N-1]` (price stays below zone top) |
| 3 | S/R confluence | Disabled by default (`require_crit3=False`); see note below |
| 4 | Priority | All passing candidates are labelled; no elimination by proximity |
| 5 | Gann box position | FVG zone bottom <= 50-bar swing midpoint | FVG zone top >= 50-bar swing midpoint |
| 6 | Recent BOS | A bullish break of structure within the past 50 bars | A bearish break of structure within the past 50 bars |

**Criterion 2 (reaction): why "loose"?** The strict version requires `close[N+2]` to fall
inside the gap zone. On SPY H1 data, the gaps are typically narrow (about 1 point wide) and price
almost always overshoots. Strict Crit 2 passes only about 10% of geometric FVGs and produces too
few positives to train on. The loose version checks only that price did not reverse back
through the zone.

**Criterion 3 (S/R confluence): why disabled?** Enabling all six strict criteria yields
only 6-29 positive labels on the full SPY H1 dataset. Disabling S/R confluence (and using
loose Crit 2) raises this to about 467 positives, the minimum needed for reliable model
training. The S/R overlay is still computed and can be re-enabled via
`ValidFVGLabeller(require_crit3=True)`.

**Break of Structure (BOS):** a BOS occurs when a candle's close exceeds a prior 50-bar
swing high (a local peak in price, the highest point before price pulls back) or falls below a prior 50-bar swing low (a local trough, the lowest point before price bounces). It
indicates that the market has broken its recent directional structure, which is a
precondition for a valid FVG in SMC theory.

**Gann midpoint** (from a trading-theory framework by W.D. Gann; "Gann box position" means where price sits within a defined reference range): the midpoint of the 50-bar swing range, computed as
`swing_low + 0.5 * (swing_high - swing_low)`. A bullish FVG should sit in the lower half
of the range (demand zone), a bearish FVG in the upper half (supply zone).

### Why is the label placed at N+2?

The label is assigned to candle N+2, not candle N. The reason: Criterion 2 requires
observing how candle N+2 closes. That close is only known once candle N+2 is complete.
Placing the label at N+2 means the model only ever sees data up to the point where the label
is knowable. No peeking into the future. This is enforced by a mandatory pytest fixture
that checks the label index offset.

### Weak supervision: no manual labelling

Labels are generated entirely programmatically. No human annotated individual bars. The
`ValidFVGLabeller` is a deterministic, vectorised function. No Python row-loops, no
per-bar state. This is called "weak supervision": domain rules produce labels at scale
instead of human effort.

The project does include a gold annotation tool (`notebooks/02-gold-annotation.ipynb`) for
validation purposes. A human labels a sample of bars to check that the programmatic labels
are correct. But the training data itself is fully automated.

---

## 4. Class imbalance

On SPY H1 data (2016 to 2025), the label distribution is approximately:

| Class | Value | Approximate rate |
|-------|-------|-----------------|
| None | 0 | ~97% |
| Bullish FVG | 1 | ~1.5% |
| Bearish FVG | 2 | ~1.5% |

This is a severe imbalance. A model that predicts "None" for every bar would achieve about 97%
accuracy, a misleading number. **Accuracy is not used as a metric.** The primary metric
throughout this project is **F1 score on the minority (FVG) classes**. F1 measures the
balance between precision (of the FVGs the model flagged, how many were real?) and recall
(of all real FVGs, how many did the model find?).

To compensate during training, class weights are computed from the training split using
inverse-frequency weighting:

```
weight[c] = total_windows / (count[c] * num_classes)
```

Weights are normalised so they sum to `num_classes` and are passed to the loss function.
They are persisted to `data/processed/class_weights_fvg_valid.json` (and legacy alias
`class_weights_spy_h1.json`) and recomputed from the training split each time `build_pipeline()`
runs, never from the validation or test splits.

---

## 5. Schema

### Parquet columns (labelled H1 DataFrame)

| Column | Type | Description |
|--------|------|-------------|
| `open` | float64 | Bar open price |
| `high` | float64 | Bar high price |
| `low` | float64 | Bar low price |
| `close` | float64 | Bar close price |
| `volume` | float64 | Total share volume traded in the bar |
| `session_type` | Categorical | `"full"` or `"half"` (NYSE early-close) |
| `raw_label` | int | Ternary label: `{-1=bearish, 0=none, 1=bullish}` |
| `label` | int | Encoded label: `{0=none, 1=bullish, 2=bearish}` |

**`raw_label` vs `label`.** Both come from the same detector: they are two encodings of one
labelling pass. `ValidFVGLabeller.label()` (`src/data/labels/valid_fvg.py`) returns the raw
ternary code `{-1, 0, 1}` directly from the SMC criteria; this is written to `raw_label` and is
the human-readable form (sign carries direction: -1 for bearish, +1 for bullish). `label` is that
same series passed through `.encode()` (`encoded_map = {0: 0, 1: 1, -1: 2}`), which remaps the
negative class to a non-negative contiguous index `{0, 1, 2}` because PyTorch's cross-entropy
expects class indices in `[0, n_classes)`. Both columns are attached side by side in
`src/data/process.py` / `pipeline.py` after `label` → `encode`. Rule of thumb: use `raw_label` for
reading, filtering, and counting positives (`raw_label != 0`); use `label` for what the models train
and predict on.

**`session_type`.** Per-bar NYSE session tag set during download
(`_tag_session_type` in `src/data/download.py`, via `exchange_calendars`): `"full"` for normal regular-session bars and `"half"` for bars on
early-close days (for example, day after Thanksgiving, Christmas Eve). It is metadata only: used for
session-aware filtering and diagnostics, not fed to the models as a feature.
| `symbol` | str | Ticker symbol; present only in multi-symbol pooled files |

Index: `DatetimeIndex` in America/New_York timezone, bar open time. Not guaranteed regular
(gaps for weekends and holidays are expected and correct).

Files are written to:
- `data/processed/spy_h1_full.parquet`: full SPY labelled series
- `data/processed/spy_h1_{train,val,test}.parquet`: split files
- `data/processed/multisym_{h1,5m,15m}_{train,val,test}.parquet`: pooled multi-symbol splits

### Windowing

Models consume 60-bar sliding windows, not individual candles. Each window is a
`(60, 5)` array of raw OHLCV values. The label for the window is the label of the last
bar in the window (position 59, i.e., the most recent candle).

Window strides:
- Training: stride = 1 (every possible window, maximum overlap for data efficiency)
- Validation and test: stride = 60 (non-overlapping windows, unbiased evaluation)

### Per-window normalisation

Each window is normalised independently before being fed to a model. No global statistics
are precomputed or leaked across the train/val/test boundary.

- **OHLC columns (0-3):** z-score (rescale so mean equals 0, spread equals 1, making different price levels comparable across windows) using the mean and std computed across all 240 OHLC
  values in the window (60 bars x 4 columns). A single scalar mean and std is applied to
  all four price columns together to preserve relative price levels within the window.
- **Volume column (4):** log1p transform first (compresses the large dynamic range of
  volume), then z-score using within-window log-volume mean and std.
- Degenerate windows (std < 1e-8) produce a zero output rather than NaN or an error.

Output dtype: float32.

### Multi-symbol pooled dataset

The multi-symbol pipeline (`build_multi_symbol_pipeline`) processes each symbol
independently through download, label, and temporal split, then concatenates the per-symbol
splits into pooled train/val/test DataFrames. Within each pooled split, rows are sorted by
`(symbol, timestamp)` so that each symbol's bars are contiguous and chronologically ordered.

A `symbol` column guards against cross-symbol windows: any 60-bar window whose bars span
more than one symbol is silently skipped during dataset construction. This prevents the
model from seeing a window where the first 30 bars are SPY and the last 30 are QQQ.

### Dataset meta sidecar

Each pipeline run writes `data/processed/dataset_meta.json` (or
`data/processed/multisym_dataset_meta.json`) recording:
- `labeller_name`, `window_size`, `start`, `end`
- `split_boundaries`
- `row_counts` per split
- `symbols` and `per_symbol_row_counts` (multi-symbol only)

---

## 6. Train/val/test split

**Plain language.** Time series data must be split in chronological order. You cannot
randomly shuffle the bars and assign 80% to training. If you did, the model would be trained
on bars from 2024 and tested on bars from 2019. It would effectively have seen the future
during training. This is called time-leakage and produces optimistic, invalid results.

**Boundaries (from `src/data/split.py`):**

| Split | Start | End |
|-------|-------|-----|
| Train | 2016-01-01 (data start) | 2021-12-31 |
| Validation | 2022-01-01 | 2022-12-31 |
| Test | 2023-01-01 | 2025-12-31 |

No bar appears in more than one split. The split is identical for all symbols in the
multi-symbol setting: applied per symbol before pooling.

The split is enforced at the date level (`index.normalize() <= boundary`) so that intraday
timestamps near midnight cannot bleed across boundaries.

Rigor and evaluation methodology are documented in `docs/architecture.md` and the reports
under `reports/rigor/`.
