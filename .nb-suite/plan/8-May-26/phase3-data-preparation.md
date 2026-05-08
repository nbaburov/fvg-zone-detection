# Phase 3 — Data Preparation Plan

> **Ready for /nb:build.**
> Work type: New feature (greenfield data pipeline)
> Recommended mode: /nb:build — all foundations are sequential; no independent workstreams until the gold-set UI, which can be built in parallel with normalisation/windowing once the FVG detector is validated.
> Test strategy: TDD (pytest)

**Goal:** Build the complete SPY H1 data pipeline from Alpaca minute-bar download through labelled, normalised, windowed (60×5) tensors with temporal train/val/test splits — and validate label quality with a 75-candle hand-labelled gold set — so Phase 4 can begin with a clean, fully characterised dataset.

**Architecture:** Three-layer pipeline. Layer 1 (ingest + clean): `download.py` pulls Alpaca 1-min bars, resamples to 09:30-anchored H1, enforces OHLC integrity, tags session types. Layer 2 (label + quality): `src/data/labels/` registry with ABC base class and concrete FVG subclass; gold-set annotation tool. Layer 3 (normalise + window + split): causal rolling normalisation, (60,5) sliding windows with strided iterator, temporal split with exact date boundaries. Every layer has a matching test module.

**Tech stack / constraints:**
- Python 3.12+
- `alpaca-py` — Alpaca free tier paper account credentials (`.env`)
- `exchange_calendars` — NYSE session schedule
- `pandas` 2.x, `numpy` 2.x
- `plotly` — annotation UI and notebook figures
- `scikit-learn` — Cohen's kappa, class weight computation
- `pytest` — TDD throughout
- `python-dotenv` — `.env` loading
- Data cache: `data/raw/spy_minute.parquet`, `data/processed/spy_h1.parquet`
- No `smartmoneyconcepts` import anywhere in Phase 3 code

**Breaking changes:** None — greenfield. No existing `src/` code.

**Out of scope (mandatory):**
- Any model code (Phase 4)
- OB / BOS / CHoCH / Liquidity / S&D labellers — ABC and registry built, only `FVGLabeller` concrete class implemented
- Multi-timeframe data (H1 only)
- Zone-bounds regression labels (Option C from R3) — extension only
- Full-sequence (per-candle 60-label) labelling (Option D from R3) — extension only
- ATR-filtered FVG variant for ablation study — Phase 5
- Synthetic / GAN augmentation (R3: explicitly rejected)
- Trade qualification or signal generation
- Live / real-time data path

**Resolved ambiguities:**
- Data source: Alpaca free tier, `TimeFrame.Minute` pull then pandas-resample — NOT `TimeFrame.Hour` direct (clock-hour alignment produces contaminated 09:00-10:00 bar; R5 confirms)
- Label index: N+1 (label placed on the first bar where the 3-candle FVG pattern is fully closed, i.e., the bar AFTER the gap-confirming candle)
- Label encoding: `{none: 0, bullish: 1, bearish: 2}` as PyTorch integer class indices (raw detector uses {-1, 0, 1}; encoded as {0→0 (none), 1→1 (bull), -1→2 (bear)} via LabelEncoder at window-assembly time)
- Window shape: (60, 5) — 60 candles × OHLCV features (open, high, low, close, volume). Stride = 1 on train, stride = 60 on val and test.
- Train/val/test boundaries: 2018-01-01–2021-12-31 (train), 2022-01-01–2022-12-31 (val), 2023-01-01–2024-12-31 (test). Exact date is the first available H1 bar on or after the boundary date.
- Normalisation: causal per-window rolling z-score on OHLC, log-scale volume normalised per-window. Statistics never leak from future candles.
- Gold set size: 75 candles, stratified (25 FVG-rich / 25 low-volatility / 25 random by year). Kappa < 0.6 = Phase 4 blocked until resolved.
- Positive label rate check: compute bull+bear frequency after labelling all candles; if < 1% stop and debug detector; if < 3% surface to user.
- Notebook: `notebooks/01-data-understanding.ipynb` covers CRISP-DM Phases 2 + 3 and is the Status Update 1 deliverable.

---

## Sequential phase — foundations first

### Foundation 1: Project scaffolding and environment

**Model:** haiku
**Files:**
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/data/labels/__init__.py` — label registry (name → class mapping)
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `.env.example`
- Create: `pyproject.toml` (or `requirements.txt` — matches existing project pattern; since no `pyproject.toml` exists yet, create `requirements.txt`)
- Modify: `CLAUDE.md` — update architecture section to reflect actual `src/data/` module paths

**What it does:** Establishes directory structure so all subsequent foundations have concrete import paths. The `src/data/labels/__init__.py` defines the label registry dict `LABELLERS: dict[str, type[BaseLabeller]]` — this must exist before the FVG concrete class is registered. Creates `.env.example` with the two required keys. Does NOT create `data/` directory (created by download step, not scaffolding). No logic — structure only.

**Contract:**

`.env.example`:
```
ALPACA_API_KEY=your_paper_account_key_here
ALPACA_API_SECRET=your_paper_account_secret_here
```

`src/data/labels/__init__.py` registry signature:
```python
LABELLERS: dict[str, type["BaseLabeller"]] = {}

def register(name: str):
    """Decorator. @register("fvg") on a BaseLabeller subclass adds it to LABELLERS."""
    def decorator(cls): ...
    return decorator
```

---

### Foundation 2: Label ABC and FVG concrete class

**Model:** sonnet
**Depends on:** Foundation 1
**Files:**
- Create: `src/data/labels/base.py`
- Create: `src/data/labels/fvg.py`
- Create: `tests/data/test_fvg_labeller.py`

**What it does:** Defines `BaseLabeller` as an abstract base class with the contract every labeller must implement. Implements `FVGLabeller` as the only concrete subclass in Phase 3. Registers it as `"fvg"` in the registry. All logic is vectorised — no Python-level for-loops over candle rows. The falsification fixture (9-candle synthetic DataFrame with one known bullish FVG and one known bearish FVG at known indices) is written as a pytest parametrised test before the implementation exists (TDD). Tests assert: label appears at N+1 not N, first and last candle are always label 0, bull and bear are correctly distinguished.

**Contract:**

`src/data/labels/base.py`:
```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseLabeller(ABC):
    label_index_offset: int        # bars after pattern close before label knowable; FVG = 1
    num_classes: int               # output head size; FVG = 3
    class_names: list[str]         # for viz tooltips; FVG = ["none", "bullish", "bearish"]
    encoded_map: dict[int, int]    # raw detector int → PyTorch class index

    @abstractmethod
    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Input:  DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
        Output: Series[int] same index as df.
                Values are RAW detector codes (before PyTorch encoding).
                FVG: {-1=bearish, 0=none, 1=bullish}.
                First label_index_offset rows must be 0 (no lookback available).
                Last label_index_offset rows must be 0 (no lookahead available).
                No NaN values permitted.
        """
        ...

    def encode(self, raw: pd.Series) -> pd.Series:
        """Map raw detector codes to PyTorch class indices via encoded_map. Non-override."""
        ...
```

`src/data/labels/fvg.py`:
```python
@register("fvg")
class FVGLabeller(BaseLabeller):
    label_index_offset = 1
    num_classes = 3
    class_names = ["none", "bullish", "bearish"]
    encoded_map = {0: 0, 1: 1, -1: 2}

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Bullish FVG at i: high[i-1] < low[i+1] AND close[i] > open[i]. Label placed at i+1.
        Bearish FVG at i: low[i-1] > high[i+1] AND close[i] < open[i]. Label placed at i+1.
        Vector operation only. i ranges from 1 to len(df)-2.
        """
        ...
```

**Falsification fixture (mandatory TDD):**

Synthetic 9-candle DataFrame where:
- Candles 2–4 form a bullish FVG (high[2] < low[4], candle 3 is bullish body)
- Candles 5–7 form a bearish FVG (low[5] > high[7], candle 6 is bearish body)

Assertions:
1. `labels.iloc[4] == 1` (bullish, placed at i+1 = 4)
2. `labels.iloc[7] == -1` (bearish, placed at i+1 = 7)
3. `labels.iloc[0] == 0` (first candle always none)
4. `labels.iloc[8] == 0` (last candle always none)
5. No other index is non-zero
6. No NaN values

---

### Foundation 3: Alpaca download and H1 resample

**Model:** sonnet
**Depends on:** Foundation 1
**Files:**
- Create: `src/data/download.py`
- Create: `tests/data/test_download.py`
- Create: `data/raw/.gitkeep` (directory marker; actual parquet excluded from git)

**What it does:** Pulls SPY 1-minute OHLCV bars from Alpaca using the free-tier paper account credentials from `.env`. Converts UTC index to `America/New_York`. Applies RTH filter (09:30–15:59 inclusive). Resamples to H1 with `closed='left', label='left'`. Drops zero-volume bars (defensive invariant check). Enforces OHLC integrity (5 rules from R5). Tags each bar with session type (`'full'` or `'half'`) using `exchange_calendars` NYSE schedule. Caches result to `data/raw/spy_minute.parquet` (minute bars) and returns the cleaned H1 DataFrame. If `data/raw/spy_minute.parquet` already exists, loads from cache instead of making API call (idempotent).

The module is structured so credentials are loaded via `python-dotenv` at call time, not at import time (no side effects on import). Network calls are isolated to a single function so tests can monkeypatch it.

**Contract:**

```python
def download_spy_h1(
    start: str = "2018-01-01",
    end: str | None = None,          # None = today
    use_cache: bool = True,
    cache_path: str = "data/raw/spy_minute.parquet",
) -> pd.DataFrame:
    """
    Returns H1 DataFrame with DatetimeIndex (America/New_York, bar open time).
    Columns: open, high, low, close, volume (float64), session_type (Categorical: 'full'|'half').
    Index frequency: not guaranteed regular (gaps for holidays/weekends — expected).
    Raises: EnvironmentError if ALPACA_API_KEY or ALPACA_API_SECRET not set.
    Raises: ValueError if returned bar count < 5000 (sanity check).
    """
    ...

def _resample_minute_to_h1(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Internal. Applies resample('1h', closed='left', label='left').agg(open/high/low/close/volume)."""
    ...

def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows violating any of:
      low <= min(open, close), high >= max(open, close),
      low <= high, all OHLC > 0, volume > 0.
    Logs count of dropped rows. Does not raise on drops.
    """
    ...

def _tag_session_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds session_type column ('full' or 'half') using exchange_calendars NYSE schedule.
    Half-day sessions identified by early_close attribute on NYSE calendar.
    """
    ...
```

**Test coverage:**
- Mock the Alpaca SDK call with a fixture DataFrame of 100 minute bars covering 2 full RTH sessions and 1 half-day session.
- Assert: output has columns `[open, high, low, close, volume, session_type]`.
- Assert: output index is `America/New_York` timezone.
- Assert: no bar with timestamp outside 09:30–15:59 ET.
- Assert: half-day session tagged as `'half'`.
- Assert: a synthetic zero-volume bar is dropped.
- Assert: a bar with `low > min(open, close)` is dropped.
- Assert: `EnvironmentError` raised when env vars absent.
- Cache hit path: write a fixture parquet, call with `use_cache=True`, assert Alpaca SDK never called.

**Escalation:** If Alpaca rate-limits during the full 8-year minute-bar pull, add `time.sleep(0.3)` between pagination chunks. SDK auto-paginates; if it blocks, chunk the date range into 1-year chunks with explicit retry on HTTP 429.

---

### Foundation 4: Persist processed H1 and label

**Model:** haiku
**Depends on:** Foundation 2, Foundation 3
**Files:**
- Create: `src/data/process.py`
- Create: `data/processed/.gitkeep`
- Create: `tests/data/test_process.py`

**What it does:** Thin orchestration module that (a) loads or downloads the H1 DataFrame from `download.py`, (b) instantiates the labeller by name from the registry, (c) runs `labeller.label()` on the full DataFrame, (d) calls `labeller.encode()` to convert raw codes to PyTorch class indices, (e) attaches the encoded labels as a `label` column to the DataFrame, (f) saves to `data/processed/spy_h1.parquet`. Also computes and logs label frequency (bull %, bear %, none %) after labelling — raises `ValueError` if bull+bear < 1%, logs a warning if bull+bear < 3%.

**Contract:**

```python
def build_labelled_dataset(
    labeller_name: str = "fvg",
    h1_cache_path: str = "data/raw/spy_minute.parquet",
    output_path: str = "data/processed/spy_h1.parquet",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns H1 DataFrame with all columns from download_spy_h1()
    plus: raw_label (int, raw detector codes), label (int, PyTorch class index 0/1/2).
    Writes output_path as parquet.
    Raises: ValueError if positive label rate (bull+bear) < 1%.
    Warns: if positive label rate < 3%.
    """
    ...
```

**Test coverage:**
- Fixture H1 DataFrame (50 rows, synthetic), mock `FVGLabeller.label()` returning known Series.
- Assert: output has `raw_label` and `label` columns.
- Assert: `label` values are in `{0, 1, 2}` only.
- Assert: `ValueError` raised when mock returns < 1% positives.
- Assert: parquet written at output path.

---

### Foundation 5: Causal normalisation

**Model:** sonnet
**Depends on:** Foundation 4
**Files:**
- Create: `src/data/normalize.py`
- Create: `tests/data/test_normalize.py`

**What it does:** Applies per-window causal normalisation to OHLCV features. For each (60,5) window: OHLC features are z-scored using only statistics computed from the 60 candles in that window (not from the global dataset — no future leakage between windows). Volume is log1p-transformed then z-scored using within-window statistics. The normalisation is applied at window-extraction time (Foundation 6), not as a separate DataFrame column pass. This module provides the normalisation functions used by the window builder.

Causal contract is strict: statistics used to normalise window W may not depend on any candle outside window W's 60 bars. The anti-lookahead test asserts this by computing normalised windows in two orders (forward and reverse) and verifying each window's values are identical regardless of the order candles were processed.

**Contract:**

```python
def normalise_window(window: np.ndarray) -> np.ndarray:
    """
    Input:  np.ndarray of shape (60, 5), dtype float64.
            Columns: [open, high, low, close, volume] — raw values.
    Output: np.ndarray of shape (60, 5), dtype float32.
            OHLC columns (0:4): z-score using window mean and std.
              mean = np.mean(window[:, 0:4])   # scalar across all OHLC cells
              std  = np.std(window[:, 0:4])    # scalar; if < 1e-8, return zeros
            Volume column (4): log1p then z-score using within-window stats.
    Raises: nothing. Returns zeros if std == 0 (degenerate window).
    Pure function. No side effects. No state.
    """
    ...
```

**Test coverage:**
- Unit test: fixed 60×5 array with known values → assert output mean ≈ 0, std ≈ 1 for OHLC.
- Unit test: volume column uses log1p → assert output values differ from raw volume z-score.
- Anti-lookahead test: extract window W from position 100 and from position 200 in the same H1 DataFrame. Assert `normalise_window(W_100)` does not depend on row 200's values. Concretely: mutate the H1 DataFrame at row 200, recompute window 100, assert it is unchanged.
- Edge case: all-zero volume in a window → assert no NaN, returns zeros in volume column.
- Edge case: window with std == 0 (all prices identical) → assert no exception, returns zeros.

---

### Foundation 6: Sliding window dataset builder

**Model:** sonnet
**Depends on:** Foundation 5
**Files:**
- Create: `src/data/window.py`
- Create: `tests/data/test_window.py`

**What it does:** Builds the windowed dataset from the labelled H1 DataFrame. For each valid window position (index i to i+59, where the label is the encoded class of the bar at i+59), extracts a (60,5) raw array, calls `normalise_window`, pairs with the integer label at position i+59. Returns a list of `(np.ndarray shape (60,5) dtype float32, int label)` tuples or wraps them in a `torch.utils.data.Dataset` subclass for Phase 4 compatibility. Stride is a parameter: `stride=1` for train, `stride=60` for val/test.

Boundary exclusion is critical: the window is valid only if the label candle (i+59) has a non-ambiguous label. First candle of the dataset (index 0) cannot be labeled (no i-1 for FVG check). Last candle cannot be labeled (no i+1 for N+1 label placement). The window builder does NOT re-derive these exclusions — it simply skips any window whose label candle has a raw_label of 0 due to boundary (the labeller already handles this). It does exclude windows that straddle session boundaries (a window whose 60 candles span a holiday gap is dropped by default, configurable).

**Contract:**

```python
class SMCWindowDataset(torch.utils.data.Dataset):
    """
    Iterates (window_tensor: Tensor[60, 5], label: int) pairs.
    """
    def __init__(
        self,
        df: pd.DataFrame,              # labelled H1 DataFrame with 'label' column
        labeller: BaseLabeller,
        stride: int = 1,               # 1 for train, window_size for val/test
        window_size: int = 60,
        drop_cross_session_windows: bool = True,
    ): ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]: ...

    @property
    def label_counts(self) -> dict[int, int]:
        """Returns {class_index: count} for all windows in this dataset."""
        ...

def build_windows(
    df: pd.DataFrame,
    labeller: BaseLabeller,
    stride: int = 1,
    window_size: int = 60,
    drop_cross_session_windows: bool = True,
) -> list[tuple[np.ndarray, int]]:
    """
    Returns list of (window_array shape (60,5) float32, encoded_label int).
    Calls normalise_window on each window before returning.
    Does not hold all windows in memory simultaneously if dataset is large — uses generator internally, materialises to list only on return.
    """
    ...
```

**Test coverage:**
- Fixture: 200-row H1 DataFrame with stride=1. Assert `len(dataset) == 200 - 60 + 1 - boundary_exclusions`.
- Assert: every window array has shape (60, 5) and dtype float32.
- Assert: window at index k has label matching `df['label'].iloc[k + 59]`.
- Assert: stride=60 produces non-overlapping windows (consecutive windows differ by exactly 60 bars in their candle ranges).
- Assert: a window straddling a known holiday gap (injected into fixture DataFrame) is dropped when `drop_cross_session_windows=True`.
- Anti-lookahead: windows from val/test split contain no bar indices from train split.
- Assert: `label_counts` sums to total window count.

---

### Foundation 7: Temporal split

**Model:** haiku
**Depends on:** Foundation 6
**Files:**
- Create: `src/data/split.py`
- Create: `tests/data/test_split.py`

**What it does:** Splits the labelled H1 DataFrame into train, validation, and test subsets using temporal boundaries. Returns three DataFrames (not window datasets — window building happens per-split in `process.py`). Boundaries are defined by first available bar on or after the boundary date. Enforces strict ordering: train end < val start, val end < test start, no bar appears in more than one split. Logs bar counts and positive label rates per split.

**Contract:**

```python
SPLIT_BOUNDARIES = {
    "train_end":  "2021-12-31",   # last bar on or before this date in train
    "val_start":  "2022-01-01",
    "val_end":    "2022-12-31",
    "test_start": "2023-01-01",
    "test_end":   "2024-12-31",   # test does not extend to today — holdout is fixed
}

def temporal_split(
    df: pd.DataFrame,
    boundaries: dict[str, str] = SPLIT_BOUNDARIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df).
    All three share the same column schema as df.
    No bar appears in more than one subset.
    Raises: ValueError if any split has 0 bars, or if positive label rate in any split == 0.
    """
    ...
```

**Test coverage:**
- Fixture DataFrame 2018–2024 with mock labels.
- Assert: `len(train) + len(val) + len(test) == len(df)` (no rows lost or duplicated).
- Assert: `train.index.max() < val.index.min()`.
- Assert: `val.index.max() < test.index.min()`.
- Assert: `ValueError` raised when any split has zero rows.
- Assert: default boundaries produce the expected date ranges.

---

### Foundation 8: Gold-set annotation tool

**Model:** sonnet
**Depends on:** Foundation 4 (needs labelled H1 data), Foundation 2 (FVGLabeller for programmatic labels)
**Files:**
- Create: `src/data/annotate.py`
- Create: `scripts/annotate_gold_set.py`
- Create: `data/gold_labels.csv` (created on first run, not committed)
- Create: `data/gold_labels.example.csv` (3-row example showing schema, committed)

**What it does:** Produces the 75-candle stratified gold set and provides a Plotly-based annotation UI. Sampling logic: loads labelled H1 DataFrame, applies three-strata sampling strategy (25 FVG-rich windows from programmatic-positive-dense periods, 25 low-volatility windows by ATR quantile, 25 random windows stratified by year). Each sampled candle is the label candle of a 7-candle display window (3 before, the target candle, 3 after). The annotation UI shows OHLCV candlestick chart without showing the programmatic label. Annotator selects Bullish FVG / Bearish FVG / None, optionally marks "ambiguous." Saves each annotation to `data/gold_labels.csv` incrementally (append mode — safe to interrupt and resume). After all 75 candles are annotated, computes and prints Cohen's kappa between programmatic labels and human labels. If kappa < 0.6, prints a prominent warning: "PHASE 4 BLOCKED — label quality gate failed. Surface to user before training."

**Contract:**

`data/gold_labels.csv` schema:
```
candle_index, datetime, programmatic_label, human_label, annotator_note
```
Where `programmatic_label` and `human_label` are raw codes (0=none, 1=bull, -1=bear), `annotator_note` is free text or empty.

```python
def sample_gold_set(
    df: pd.DataFrame,
    n_fvg_rich: int = 25,
    n_low_vol: int = 25,
    n_random: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns DataFrame of 75 rows from df, stratified per protocol.
    Each row includes candle_index (integer position), datetime, and programmatic_label.
    """
    ...

def compute_kappa(gold_csv_path: str) -> float:
    """
    Loads gold_labels.csv. Computes Cohen's kappa on (programmatic_label, human_label).
    Returns float in [-1, 1]. Prints breakdown by strata.
    Raises: FileNotFoundError if path does not exist.
    """
    ...

def run_annotation_ui(
    df: pd.DataFrame,
    gold_set_df: pd.DataFrame,
    output_path: str = "data/gold_labels.csv",
) -> None:
    """
    Opens Plotly figure for each unannotated candle in gold_set_df.
    Displays 7-candle candlestick (3 before + target + 3 after).
    Does NOT display programmatic label.
    Waits for user keyboard input (b=bullish, e=bearish, n=none, a=ambiguous+none).
    Appends annotation to output_path after each input.
    On resume: skips already-annotated candle_index values.
    """
    ...
```

**Test coverage:**
- `sample_gold_set`: fixture 1000-row H1 DataFrame → assert exactly 75 rows returned, no duplicates, all three strata present.
- `compute_kappa`: fixture gold CSV with known agreement/disagreement → assert kappa matches sklearn's `cohen_kappa_score` output.
- `run_annotation_ui`: not unit-tested (interactive). Manual verification documented in Phase 3 build log.

---

### Foundation 9: Integration — full pipeline and dataset export

**Model:** haiku
**Depends on:** Foundations 1–8
**Files:**
- Create: `src/data/pipeline.py`
- Create: `tests/data/test_pipeline.py`

**What it does:** Top-level pipeline function that chains all foundations: download → label → split → window → return `(train_dataset, val_dataset, test_dataset, class_weights)`. Class weights computed from train split's label frequency (inverse frequency, normalised to sum to `num_classes`). Persists class weights to `data/processed/class_weights.json` for Phase 4 consumption. Also persists train/val/test split DataFrames to `data/processed/spy_h1_{split}.parquet`. This is the single entry point Phase 4 will call.

**Contract:**

```python
def build_pipeline(
    labeller_name: str = "fvg",
    window_size: int = 60,
    seed: int = 42,
) -> tuple[SMCWindowDataset, SMCWindowDataset, SMCWindowDataset, torch.Tensor]:
    """
    Returns (train_dataset, val_dataset, test_dataset, class_weights).
    class_weights: Tensor of shape (num_classes,), dtype float32.
                   Computed on train split only. Persisted to data/processed/class_weights.json.
    train_dataset stride = 1.
    val_dataset stride = 60.
    test_dataset stride = 60.
    Raises: ValueError if any dataset has 0 positive-class windows.
    """
    ...
```

Persisted outputs:
- `data/processed/spy_h1.parquet` — full labelled H1 DataFrame
- `data/processed/spy_h1_train.parquet`
- `data/processed/spy_h1_val.parquet`
- `data/processed/spy_h1_test.parquet`
- `data/processed/class_weights.json` — `{"0": w0, "1": w1, "2": w2}`

**Test coverage:**
- End-to-end integration test using fixture (synthetic 500-row DataFrame, mock Alpaca call, known FVG positions).
- Assert: train + val + test window counts sum to expected total (minus boundary exclusions).
- Assert: no candle index appears in both train and val, or val and test.
- Assert: class_weights shape == (3,), sums to 3.0 (normalised).
- Assert: `class_weights.json` written with correct keys.

---

### Foundation 10: CRISP-DM notebook (Status Update 1 deliverable)

**Model:** sonnet
**Depends on:** Foundations 1–9 (all code working, data on disk)
**Files:**
- Create: `notebooks/01-data-understanding.ipynb`

**What it does:** Jupyter notebook covering CRISP-DM Phase 2 (Data Understanding) and Phase 3 (Data Preparation) in full narrative form. This is the Status Update 1 deliverable submitted for peer review on 17 May. Follows `nb-notebook` education standard: full markdown narrative, figures with captions and observation cells, formal but human voice. All figures use Plotly (interactive) with static fallback.

**Notebook outline (section-level):**

```
1. Business Understanding (brief recap)
   1.1 Problem statement
   1.2 ML reframing: sequence classification, not price prediction
   1.3 Success criteria for this status update

2. Data Understanding
   2.1 Data source selection rationale (yfinance failure, Alpaca decision)
   2.2 Raw data profile: bar schema, timestamp semantics, known quality issues
   2.3 SPY H1 overview: date range, bar count, session calendar
       Figure 2.3.1: Full-period candlestick chart (Plotly, sampled)
       Caption + Observation cell
   2.4 Missing data and gaps: holiday/half-day distribution
       Figure 2.4.1: Session-type bar chart (full vs half-day by year)
       Caption + Observation cell
   2.5 Volume profile: intraday seasonality
       Figure 2.5.1: Mean volume by hour-of-day across full dataset
       Caption + Observation cell

3. Data Preparation
   3.1 Cleaning pipeline: 9-step procedure with row counts at each step
       Figure 3.1.1: Row count waterfall from raw minute bars to clean H1
       Caption + Observation cell
   3.2 FVG detector specification
       3.2.1 Mathematical definition (bullish and bearish cases)
       3.2.2 Label index convention (N+1, not N) with diagram
       3.2.3 Falsification test result: 9-candle synthetic fixture
   3.3 Label distribution
       Figure 3.3.1: Label frequency bar chart (none / bullish / bearish)
       Caption + Observation cell
       Figure 3.3.2: Label frequency over time (rolling 30-day window)
       Caption + Observation cell
   3.4 Label quality analysis (gold set)
       3.4.1 Sampling protocol: 3 strata, 75 candles
       3.4.2 Agreement table: programmatic vs human by strata
       3.4.3 Cohen's kappa result + 95% CI interpretation
       3.4.4 Qualitative disagreement analysis: 3 example candles where rule and human disagreed
   3.5 Train/val/test split
       Figure 3.5.1: Timeline showing split boundaries with bar counts + positive rates per split
       Caption + Observation cell
   3.6 Normalisation
       3.6.1 Method: per-window z-score (causal, no lookahead)
       3.6.2 Anti-lookahead test result
       Figure 3.6.1: Before/after normalisation on 3 sample windows
       Caption + Observation cell
   3.7 Window dataset statistics
       Figure 3.7.1: Window count per split and per class
       Caption + Observation cell
       Table: tensor shapes emitted to Phase 4

4. Summary and handoff to Phase 4
   4.1 What Phase 3 produced (file paths, shapes, class weights)
   4.2 Known limitations and open questions
   4.3 Plan for Phase 4 (architecture overview, first milestone)
```

Every section marked with an Observation cell must contain at least 2 sentences of substantive observation (not just "the chart shows X"). Formal but human voice throughout. No AI-pattern writing.

---

## Test coverage

**Per module — concrete behaviours, not "test the function":**

| Module | What to test | Key failure modes | Edge cases in scope |
|--------|-------------|-------------------|---------------------|
| `labels/base.py` | ABC cannot be instantiated; `encode()` maps all raw codes to valid class indices | Subclass missing `label()` → TypeError at instantiation | Registry lookup of unregistered name raises KeyError |
| `labels/fvg.py` | **9-candle falsification fixture** (see Foundation 2); bull/bear direction correct; first/last candle always 0; no NaN; pure vector operation (no row-loop) | Off-by-one on N vs N+1 — the canonical error; swapped bull/bear; gap check uses wrong candle pair | First candle of dataset (no i-1), last candle (no i+1), consecutive FVGs, overlapping FVG conditions |
| `download.py` | Schema columns present; index in NY tz; no AH bars in output; half-day tagged; zero-volume dropped; OHLC-integrity violator dropped; EnvironmentError on missing creds; cache hit skips API | Timezone drift on DST transition; extended-hours bars slip through; corrupt OHLC bar not caught | DST transition day, half-day session (Black Friday), zero-volume RTH bar (SPY invariant violation) |
| `process.py` | Output has `raw_label` and `label` columns; `label` values in {0,1,2}; ValueError if < 1% positive; parquet written | Wrong encoding map applied; ValueError threshold wrong | All-zero label output (detector broken) |
| `normalize.py` | OHLC output mean ≈ 0 and std ≈ 1 per window; volume uses log1p; anti-lookahead (mutating row outside window does not change window output); std=0 returns zeros not NaN | Future-candle statistics used (anti-lookahead failure — the most dangerous subtle bug) | Zero-volume window, all-identical-price window |
| `window.py` | Window shape (60,5) dtype float32; label at position k+59; stride=60 produces non-overlapping windows; cross-session window dropped when flag set; `label_counts` sums correctly | Window-label misalignment (off by one); overlapping val/test windows; boundary window not excluded | Holiday gap spanning window boundary, single-session dataset |
| `split.py` | Strict ordering of splits; zero rows lost or duplicated; ValueError on empty split; correct date boundaries | Time leakage (candle appears in two splits) | Split boundary falls on holiday (no bar that day — must use next trading day) |
| `annotate.py` / `sample_gold_set` | Exactly 75 rows; no duplicates; all 3 strata present; `compute_kappa` matches sklearn | Strata sampling fails on sparse positives; kappa computed on wrong column pair | Fewer than 25 FVG-rich candles in dataset (rare data edge case) |
| `pipeline.py` | End-to-end fixture run; no candle in both train and val; class_weights shape and sum; all parquets written; SMCWindowDataset instances returned | Cross-split candle leak; wrong class weights (computed on full dataset instead of train only) | Minimal 70-bar dataset (just enough for one window per split) |

**Mandatory fixtures (create in `tests/conftest.py`):**
- `spy_9candle_fvg` — 9-candle synthetic DataFrame with known bull and bear FVG at known indices
- `spy_h1_fixture` — 500-row synthetic H1 DataFrame with RTH structure, one half-day session, and one 3-day holiday gap
- `spy_half_day_session` — 7-bar half-day H1 DataFrame (09:30–13:00)
- `spy_dst_transition` — 2 bars spanning a DST transition (UTC offset changes between them; index in NY tz should be unchanged in local time)
- `spy_zero_volume_bar` — fixture with 1 injected zero-volume bar in RTH
- `spy_ohlc_integrity_violation` — fixture with 1 bar where `high < max(open, close)`
- `gold_labels_fixture` — minimal `gold_labels.csv` with 10 rows, known kappa = 0.75

---

## Edge cases and gotchas

- **Alpaca H1 vs minute pull:** `TimeFrame.Hour` from Alpaca is clock-aligned — first daily bar is 09:00–10:00 ET, mixing 30 minutes of pre-market with 30 minutes of RTH. Plan specifies minute pull + resample. Do not use `TimeFrame.Hour` even as a shortcut for testing.
- **UTC timestamps in Alpaca response:** bar timestamp = bar open time in UTC. `14:30 UTC` = `09:30 ET` in winter (EST, UTC-5). Convert to `America/New_York` before `.between_time()` — not to `US/Eastern` (they behave differently at the DST boundary in pytz).
- **Last bar of RTH is a 30-minute bar:** resample anchored at 09:30 produces bars at 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30. The 15:30 bar covers 15:30–16:00 (30 min, not 60 min). This is correct and expected. Do not drop it. The `volume` field will reflect only 30 minutes of trades. Consider whether to weight this bar differently in Phase 5 ablation — not in Phase 3.
- **FVG boundary candles:** first 2 candles and last candle of the DataFrame get `raw_label = 0` by construction (no i-1 or no i+1). These are not excluded from windows — they contribute as context within windows. They are only excluded as the *label candle* (the 60th bar of a window). The window builder handles this naturally by matching label to position 59 in each window, which will be 0 for boundary positions.
- **Session gap windows:** a 60-candle window that includes a holiday gap (e.g., candles 30–59 are from the week after Thanksgiving, candles 0–29 from before) should be dropped by default. This is NOT the same as dropping candles from the DataFrame — the DataFrame has gaps by design. The window builder detects cross-session windows by checking if any two consecutive candles in the window are more than 90 minutes apart (> 1 H1 bar gap).
- **Class weight computation on train only:** class weights must be computed exclusively from the training split. If computed on the full dataset, they leak the class distribution of val/test. `pipeline.py` enforces this; any direct call to compute weights must receive `train_df`, not the full `df`.
- **Parquet and timezone:** `pd.read_parquet()` preserves timezone-aware DatetimeIndex correctly in pandas 2.x. Do not convert to UTC when writing — preserve `America/New_York` timezone in the parquet file. Verify on reload.
- **Gold set annotation tool:** the programmatic label must NOT be displayed to the annotator during the annotation session. The `run_annotation_ui` function must load the display data from the raw OHLCV DataFrame, not from the labelled one. This is an anchoring bias prevention measure per R3.
- **Cohen's kappa on 75 samples:** confidence intervals will be wide (~±0.15 at 95%). State this explicitly in the notebook. Kappa is a directional gate, not a precise measurement.
- **Positive label rate variability:** the exact positive rate on SPY H1 2018–2024 is unknown until the pipeline runs. The 3%/1% thresholds in `process.py` are conservative guards; the expected rate is 5–15% based on R3. If the rate is < 3%, check the FVG detector against the 9-candle fixture first before assuming a data problem.

---

## Risk register

| Risk | Likelihood | Blast radius | Reversibility | Mitigation |
|------|------------|--------------|---------------|------------|
| Alpaca minute-bar pull times out or rate-limits on 8-year pull | Medium | Foundation 3 blocked | Easy — chunk by year with retry | Add chunked-request fallback (1-year chunks, `time.sleep(0.3)` between chunks). Cache partial results per chunk. |
| FVG positive rate < 1% after labelling (detector bug or definition mismatch) | Low-Medium | All of Phase 4 (bad labels = no useful training signal) | Medium — fix detector and re-run (fast, data cached) | Falsification fixture catches off-by-one. Label density check in `process.py` raises immediately. Fix before advancing. |
| Cohen's kappa < 0.6 on gold set | Medium | Phase 4 blocked until resolved | Hard — requires investigative work to determine if rule is wrong or gold set is too small | Gold set gate is explicit. If kappa fails, extend to 100 candles with more FVG-rich strata before deciding. Surface to user — do not auto-advance. |
| Normalisation inadvertently uses future statistics (anti-lookahead failure) | Low | All Phase 4 training results invalid (model "knows the future" during training) | Hard — requires retraining everything | Explicit anti-lookahead unit test in Foundation 5 catches this. Review the test before calling Foundation 5 done. |
| Alpaca returns duplicate timestamp rows (known SDK edge case on pagination boundary) | Low-Medium | Windowing produces duplicate windows | Easy — deduplicate on index after download | Add `df = df[~df.index.duplicated(keep='first')]` after download and log duplicate count. |

---

## Phase 4 handoff contract

What Phase 4 receives from Phase 3 (concrete, no ambiguity):

**Files on disk:**
- `data/processed/spy_h1_train.parquet` — labelled H1 DataFrame, train split
- `data/processed/spy_h1_val.parquet` — labelled H1 DataFrame, val split
- `data/processed/spy_h1_test.parquet` — labelled H1 DataFrame, test split
- `data/processed/class_weights.json` — `{"0": w0, "1": w1, "2": w2}`, computed on train split
- `data/gold_labels.csv` — 75-candle annotation with kappa logged
- `data/processed/spy_h1.parquet` — full labelled H1 DataFrame

**PyTorch entry point:**
```python
from src.data.pipeline import build_pipeline
train_ds, val_ds, test_ds, class_weights = build_pipeline(labeller_name="fvg")
# train_ds[i] -> (Tensor[60, 5], int)  stride=1
# val_ds[i]   -> (Tensor[60, 5], int)  stride=60
# test_ds[i]  -> (Tensor[60, 5], int)  stride=60
# class_weights -> Tensor[3] float32   sum ≈ 3.0
```

**Tensor shapes:**
- Input: `(batch, 60, 5)` — batch × sequence × features
- Label: `(batch,)` — integer class indices, values in `{0, 1, 2}`
- `num_classes = 3`

**Label encoding:**
- `0` = none
- `1` = bullish FVG
- `2` = bearish FVG

**Approximate window counts (estimated, exact numbers logged in notebook):**
- Train (2018–2021, stride=1): ~9,000–10,000 windows
- Val (2022, stride=60): ~100–130 windows
- Test (2023–2024, stride=60): ~200–270 windows

**Labeller registry key:** `"fvg"` — Phase 4 must pass this string to `build_pipeline`. Do not hardcode FVGLabeller directly; use the registry.

**Class weights usage in Phase 4:** pass directly to `torch.nn.CrossEntropyLoss(weight=class_weights)`. Weights were computed as `num_classes / (class_count * total_windows)` per standard inverse-frequency formula.

---

## What to flag if found

- Alpaca bar count for any full trading day substantially lower than expected (< 5 bars for a standard RTH day in 2020–2024) — may indicate IEX vs SIP feed issue for historical data.
- Positive label rate outside the 3–20% range after full labelling — investigate detector before proceeding.
- Any candle with a non-zero label in the first 2 positions or last 1 position of the full DataFrame — labeller boundary contract violated.
- Any window in val or test that shares candle indices with train — temporal leakage, block Phase 4 immediately.
- Gold set kappa < 0.6 — surface to user, document finding, do not auto-advance to Phase 4.
- Half-day bar count substantially different from expected 7 bars (09:30–13:00 = 4 full H1 bars: 09:30, 10:30, 11:30, 12:30; plus potentially a partial bar) — investigate resample behaviour on early-close sessions.
- `exchange_calendars` NYSE calendar missing any known holiday in 2018–2024 — validate against official NYSE holiday list for 2020 COVID circuit-breaker days specifically.
