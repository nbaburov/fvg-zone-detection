# Data Model

## On-disk artifacts

### `data/raw/spy_minute.parquet`
Raw 1-minute SPY bars from Alpaca, full history 2016+, RTH only.

| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime64[ns, UTC] (index) | Bar open time |
| open / high / low / close | float64 | OHLC prices, `adjustment="raw"` |
| volume | int64 | Share count |
| trade_count | int64 | Number of trades (Alpaca) |
| vwap | float64 | Volume-weighted average price |

### `data/processed/spy_h1.parquet`
H1 resampled from 1-min via `closed='left', label='left', offset='30min'`. RTH only (09:30–15:59 ET). Cleaned (zero-volume + OHLC integrity dropped).

| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime64[ns, UTC] (index) | H1 bar open, anchored 09:30 ET |
| open / high / low / close | float64 | OHLC for the hour |
| volume | int64 | Sum of 1-min volume |
| session_type | category | `regular` / `half_day` |

### `data/processed/spy_h1.parquet`
Full H1 dataset with label columns. (`spy_h1_labeled.parquet` was the legacy name — deprecated and deleted after ValidFVG migration.)

| Field | Type | Description |
|-------|------|-------------|
| fvg_valid | int8 | **Canonical training target** — `ValidFVGLabeller` @ N+2, 6-criteria, ~3% positive rate |
| fvg | int8 | Historical raw `FVGLabeller` @ N+1, ~25% positive rate (kept for rawfvg checkpoints) |

### `data/processed/spy_h1_{train,val,test}.parquet`
Temporal split — no shuffle.
- train: 2016–2021 (10,577 bars; none=96.8%, bull=1.9%, bear=1.3%)
- val: 2022 (1,757 bars)
- test: 2023–2025 (5,257 bars)

Label column: `fvg_valid`. Schema = `spy_h1.parquet` schema.

### `data/processed/class_weights.json`
Inverse-frequency weights computed on SPY-only train split, consumed by `WeightedCE`.

```json
{"0": 0.34, "1": 50.21, "2": 48.93}
```

### Multi-symbol dataset (phase-4 expansion) — `data/processed/multisym/`

#### `{spy,qqq,iwm,dia}_h1.parquet`
Per-symbol H1 bars with labels, 2016–2025 (full history used for downstream alignment). Schema = `spy_h1.parquet` schema (timestamp index, OHLCV columns + fvg/fvg_valid labels).

#### `multisym_h1_{train,val,test}.parquet`
Pooled concatenation of all 4 symbols after split (per-symbol temporal split, then row-stacked). Row counts:
- train: ~42k rows (10.5k/symbol × 4, 2016–2021)
- val: ~7k rows (1.75k/symbol × 4, 2022)
- test: ~21k rows (5.25k/symbol × 4, 2023–2025)

Label distribution (pooled train, fvg_valid):
- Bull: 817 (vs SPY-only 201, **4.06×**)
- Bear: 531 (vs SPY-only 142, **3.74×**)
- None: ~41k

#### `dataset_meta.json`
Split metadata + provenance.

```json
{
  "symbols": ["spy", "qqq", "iwm", "dia"],
  "train_rows_per_symbol": {"spy": 10577, "qqq": 10540, ...},
  "test_rows_per_symbol": {"spy": 5257, ...},
  "train_date_range": ["2016-01-04", "2021-12-31"],
  "test_date_range": ["2023-01-03", "2025-12-30"],
  "pooled_train_bull": 817,
  "pooled_train_bear": 531,
  "cross_symbol_window_guard": true,
  "split_boundary_provenance": {...}
}
```

#### `class_weights_multisym.json`
Inverse-frequency weights computed on pooled multi-symbol train, consumed by `WeightedCE` in multi-symbol models.

```json
{"0": 0.00097, "1": 0.0612, "2": 0.0941}
```

### `data/gold_labels.csv`
Human-annotated gold set used to validate `ValidFVGLabeller` (Cohen's κ = 1.0 against current labeller config).

| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime64 | Anchor bar |
| programmatic_label | int | Label from current labeller |
| human_label | int | Human annotation |
| stratum | string | Sampling stratum (`fvg_rich`, `low_vol`, `random`) |
| notes | string | Annotator comment |

## In-memory shapes

### `windows_raw`, `windows_norm`
Shape `(N, 60, 5)`, float32. Columns: `[open, high, low, close, volume]`.
- `windows_raw` = un-normalised, used by XGBoost feature extraction + viz.
- `windows_norm` = per-window min-max normalised, used by DL adapters.

### `future_ohlcv` (inspector, optional)
Shape `(N, lookahead_bars, 5)`. NaN-padded when window has no full future tail. Used to compute trade outcomes.

### Adapter contract
```
ModelAdapter.predict_proba(windows: np.ndarray) -> np.ndarray
  windows shape: (N, 60, 5)
  return shape:  (N, 3)  — class probabilities for {none, bull, bear}
```

**Multi-symbol note:** In `src/data/window.py::_window_generator` and `src/features/window_features.py` (XGB path), cross-symbol window guards enforce that no window contains bars from different symbols (temporal-continuity assertion). Pooled models must reload data with `_sort_if_multisym` helper in `src/rigor/seed_sweep.py` to preserve per-symbol ordering.

## Live session logs (`logs/paper/<session-id>/`)

### `bars.parquet`
Every H1 bar consumed by the live harness. Same schema as `spy_h1.parquet`. Used for replay.

### `events.jsonl`
One JSON object per inference call.

| Field | Type | Description |
|-------|------|-------------|
| ts | str (ISO) | H1 bar close time |
| model | str | Adapter name |
| proba | list[float] | Length-3 class probabilities |
| decision | str | `bull` / `bear` / `none` |
| reason | str | Threshold gate / blackout / cooldown / etc. |

### `trades.sqlite`
Three tables.

**orders**
| Field | Type |
|-------|------|
| id | INTEGER PK |
| ts | TEXT |
| client_order_id | TEXT |
| side | TEXT (`buy` / `sell`) |
| qty | INTEGER |
| entry | REAL |
| sl | REAL |
| tp | REAL |
| direction | INTEGER (1=bull, 2=bear) |

**fills**
| Field | Type |
|-------|------|
| id | INTEGER PK |
| order_id | INTEGER FK orders.id |
| ts | TEXT |
| price | REAL |
| qty | INTEGER |
| leg | TEXT (`entry` / `sl` / `tp`) |

**equity_snapshots**
| Field | Type |
|-------|------|
| ts | TEXT |
| equity | REAL |
| cash | REAL |
| drawdown_pct | REAL |

## Relationships

- A `spy_h1_labeled` row → has one or more 60-bar windows ending at or after it.
- A 60-bar window → references bars `[t-59, t]` and (in outcome mode) future bars `[t+1, t+lookahead]`.
- A live `events.jsonl` entry → optionally produces one `orders` row → produces 1–3 `fills` rows (entry + SL or TP leg).

## Indexes

- All parquets indexed on `timestamp`.
- `trades.sqlite.orders.ts` indexed for replay query speed.
