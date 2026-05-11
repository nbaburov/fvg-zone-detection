# Plan: Model Inspection / Live-Comparison Tool

**Date:** 11-May-26  
**Depth:** Standard  
**Feature type:** New feature — no existing code to modify  

---

## Goal

Single CLI script that loads any subset of trained models, runs live inference on a chosen data slice, surfaces per-window visualisations, aggregate stats (F1/confusion), and a disagreement explorer. Replaces the deleted `notebooks/03-model-comparison.ipynb` + `scripts/precompute_predictions.py` flow.

---

## Out of scope

- Web UI or live dashboard server
- Model retraining or fine-tuning inside this tool
- Support for non-FVG label schemes (only `fvg_valid` / ternary 3-class)
- Streaming or incremental inference
- Precomputed/cached predictions (all inference is live)

---

## Resolved ambiguities

- **Subprocess isolation for xgboost+torch segfault**: Only required on Python 3.14+. Current env (Python 3.12 per lock) is unaffected. Document the risk; no subprocess workaround needed now. Revisit if env is upgraded.
- **Label encoding**: `ValidFVGLabeller.encoded_map = {0:0, 1:1, -1:2}`. Minority class for F1 = classes 1 and 2 (bullish + bearish FVG). "FVG present" = `label != 0`. Report macro-F1 on classes 1+2 separately, and combined FVG-vs-none binary F1 as headline.
- **OHLCV normalisation**: `normalise_window` in `src/data/normalize.py` is applied per-window in `SMCWindowDataset`. LSTM adapter must apply the same normalisation. XGBoost adapter uses raw features via `extract_window_features` (already causal, no norm needed).
- **Disagreement metric**: agreement = fraction of windows where all loaded models predict the same class. Disagreement score per window = `1 - agreement_fraction_across_models`.
- **Output location**: `reports/inspect/<YYYY-MM-DD_HHMMSS>/` — already gitignored.
- **Model selection at CLI**: adapter auto-discovery by scanning `src/inspect/adapters/` — no central if/else.

---

## Architecture overview

```
scripts/inspect_models.py        # CLI entrypoint — arg parsing, orchestration
src/inspect/
  __init__.py
  base.py                        # ModelAdapter ABC
  registry.py                    # auto-discover + instantiate adapters by name
  runner.py                      # runs inference across all models, returns InspectionResults
  stats.py                       # F1/confusion/agreement computation
  viz.py                         # per-window Plotly chart + disagreement HTML
  report.py                      # writes markdown summary + calls viz
  adapters/
    __init__.py
    lstm_adapter.py              # wraps FVGLSTMClassifier
    xgboost_adapter.py           # wraps XGBoostFVGClassifier
    # future: cnn_lstm_adapter.py, xlstm_adapter.py, transformer_adapter.py
tests/
  inspect/
    test_base_adapter.py         # unit: adapter interface contract
    test_registry.py             # unit: discovery + instantiation
    test_stats.py                # unit: F1/confusion/agreement with known arrays
    test_runner_integration.py   # integration: tiny 5-window slice, 2 adapters
```

Data flow:
1. CLI parses args → calls `registry.load_adapters(names, checkpoint_dir)`
2. `runner.run(adapters, df_slice, labeller)` → builds windows, runs each adapter, returns `InspectionResults`
3. `stats.compute(results)` → per-model metrics + agreement matrix
4. `report.write(results, stats, output_dir)` → Plotly HTML per window (subset) + markdown summary

---

## Data model / types

```python
# src/inspect/base.py
from abc import ABC, abstractmethod
import numpy as np

class ModelAdapter(ABC):
    name: str  # must be set as class attribute

    @abstractmethod
    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """
        windows: (N, 60, 5) float32 — already normalised OHLCV windows
        returns: (N, 3) float32 — class probabilities [none, bull, bear]
        """
        ...

    @property
    def predict(self) -> np.ndarray:
        # default: argmax of predict_proba — subclass can override
        ...

# src/inspect/runner.py
from dataclasses import dataclass

@dataclass
class InspectionResults:
    windows: np.ndarray          # (N, 60, 5) raw OHLCV (un-normalised, for plotting)
    labels: np.ndarray           # (N,) encoded int — ground truth
    timestamps: pd.DatetimeIndex # (N,) — anchor timestamp of each window's last bar
    probas: dict[str, np.ndarray]  # model_name -> (N, 3) probas
    preds: dict[str, np.ndarray]   # model_name -> (N,) argmax predictions
```

---

## Component breakdown

### `src/inspect/base.py`
Defines `ModelAdapter` ABC with `predict_proba(windows) -> (N,3)` and `name: str`. Windows are always `(N, 60, 5)` float32, pre-normalised (same normalisation as training). Adapters are stateless after loading — no internal state mutation during inference.

### `src/inspect/registry.py`
Auto-discovers adapter classes by importing all `src/inspect/adapters/*.py` files at runtime. Builds `{name: class}` dict. `load_adapters(names: list[str], checkpoint_dir: Path) -> list[ModelAdapter]` instantiates by name, passes checkpoint path, raises `KeyError` with clear message if unknown name. `list_available() -> list[str]` for `--list` CLI flag. No central if/else — new adapter = new file.

### `src/inspect/adapters/lstm_adapter.py`
- `name = "lstm"`
- Loads `FVGLSTMClassifier` from `{checkpoint_dir}/lstm_seed42.pt` (configurable via `checkpoint_file` kwarg)
- Applies `normalise_window` per-window before inference (matches training pipeline)
- Uses `torch.no_grad()`, moves to CPU (MPS unsafe for eval batch, CPU reliable)
- Returns `softmax(logits, dim=-1)` as probas
- Gracefully raises `FileNotFoundError` with path if checkpoint missing

### `src/inspect/adapters/xgboost_adapter.py`
- `name = "xgboost"`
- Loads `XGBoostFVGClassifier` from `{checkpoint_dir}/xgb_seed42.ubj`
- Calls `extract_window_features` to convert `(N, 60, 5)` windows to `(N, 35)` tabular features (windows are raw OHLCV here — XGBoost does NOT need normalised input)
- Returns `clf.predict_proba(X)` as `(N, 3)` float32
- Raises `FileNotFoundError` if checkpoint missing

### `src/inspect/runner.py`
Accepts `df_slice: pd.DataFrame` (must have `open, high, low, close, volume, label` columns + DatetimeIndex), `labeller: BaseLabeller`, `window_size=60`, `stride=1`. Builds windows using `_window_generator` logic (reuse or import from `src.data.window`). Stores raw OHLCV windows separately from normalised windows. Passes raw to viz, normalised to adapters. Returns `InspectionResults`.

### `src/inspect/stats.py`
`compute_stats(results: InspectionResults) -> InspectionStats`. Returns:
- Per-model: precision/recall/F1 for classes 1 (bull) and 2 (bear), macro F1 on FVG classes, binary FVG-vs-none F1
- Confusion matrix per model (3×3)
- Agreement matrix: (n_models × n_models) fraction of windows where model_i and model_j agree
- Disagreement index per window: fraction of model pairs that disagree — for surfacing top-K disagreement windows

### `src/inspect/viz.py`
`plot_window(raw_ohlcv: np.ndarray, timestamp: pd.Timestamp, gold_label: int, model_preds: dict[str, tuple[int, float]], output_path: Path) -> None`. Plotly candlestick (60 bars), gold label annotated on final bar (colour = bull/bear/none), per-model prediction bar with confidence as marker size. Saved as standalone HTML. Only generates for requested windows (top-K disagreement + optional random sample), not all N windows (avoids 10K HTML files).

### `src/inspect/report.py`
`write_report(results, stats, output_dir: Path, top_k: int = 20) -> Path`. Creates `output_dir/summary.md` with per-model metric tables and agreement matrix. Creates `output_dir/plots/window_{i}.html` for top-K disagreement windows + any explicitly requested windows. Returns path to summary.md.

### `scripts/inspect_models.py`
CLI entrypoint. Handles: arg parsing, data loading + optional date-range slicing, labeller instantiation, registry + adapter loading, runner execution, stats + report writing. Prints summary to stdout and writes to `reports/inspect/<timestamp>/`.

---

## CLI surface

```
python scripts/inspect_models.py \
  [--models lstm xgboost]            # default: all discovered adapters \
  [--dataset val|test|<path.parquet>] # default: test \
  [--start 2024-01-01]               # optional date filter (inclusive) \
  [--end   2024-03-31]               # optional date filter (inclusive) \
  [--top-k 20]                       # disagreement windows to plot (default 20) \
  [--checkpoint-dir checkpoints/]    # default: checkpoints/ \
  [--output-dir reports/inspect/]    # timestamp subdir appended automatically \
  [--list]                           # list available adapter names and exit \
  [--seed 42]                        # numpy/torch seed for any sampling
```

Example:
```bash
python scripts/inspect_models.py --models lstm xgboost --dataset test --top-k 30
python scripts/inspect_models.py --list
python scripts/inspect_models.py --models lstm --dataset data/processed/spy_h1_val.parquet --start 2023-06 --end 2023-09
```

---

## Sequential vs parallel

Phase 1 (foundation, must go first):
- `src/inspect/base.py` — ABC
- `src/inspect/registry.py` — discovery

Phase 2 (parallel after Phase 1):
- `src/inspect/adapters/lstm_adapter.py`
- `src/inspect/adapters/xgboost_adapter.py`
- `src/inspect/runner.py`
- `src/inspect/stats.py`

Phase 3 (parallel after Phase 2):
- `src/inspect/viz.py`
- `src/inspect/report.py`

Phase 4 (sequential last):
- `scripts/inspect_models.py` — wires all above

Phase 5:
- Tests

---

## Error handling

| Error | Where caught | Behaviour |
|-------|-------------|-----------|
| Checkpoint file missing | Adapter `__init__` | `FileNotFoundError` with full path, propagates to CLI which prints human message and exits 1 |
| Unknown model name | `registry.load_adapters` | `KeyError("Unknown adapter 'foo'. Available: lstm, xgboost")` |
| Date range produces 0 windows | `runner.run` | Returns `InspectionResults` with N=0, stats are zeroed, report notes "no windows in range" |
| Date range has 0 positive labels | `stats.compute` | F1 reported as 0.0 with a warning in summary.md — no crash |
| Adapter output shape mismatch | `runner.run` | Assert `probas.shape == (N, 3)`, raise `ValueError` with model name + actual shape |
| Parquet path not found | CLI | `FileNotFoundError` before any inference |

---

## Edge cases and constraints

- **XGBoost adapter receives raw OHLCV windows, LSTM adapter receives normalised.** Both receive the same `(N, 60, 5)` array from runner but XGBoost adapter calls `extract_window_features` internally. The runner stores both raw (for viz) and normalised (for torch adapters) arrays.
- **Cross-session window filtering**: runner must apply the same `_has_session_gap` logic as `SMCWindowDataset`. Import + reuse the function directly — do not re-implement.
- **Tiny date ranges**: if fewer than `window_size=60` bars in slice, runner returns 0 windows cleanly.
- **Adding a new model**: create `src/inspect/adapters/foo_adapter.py`, set `name = "foo"`, implement `predict_proba`. Registry picks it up automatically. No other file changes.
- **Python 3.14 xgboost+torch segfault**: not present in Python 3.12. Document in README of `src/inspect/` — if env is upgraded, isolate XGBoost adapter in a subprocess using `multiprocessing` start method `spawn`, communicating arrays via shared memory or temp file.

---

## Test coverage

| Test file | Scope | Model |
|-----------|-------|-------|
| `tests/inspect/test_base_adapter.py` | ABC cannot be instantiated; concrete subclass with stub `predict_proba` satisfies interface | haiku |
| `tests/inspect/test_registry.py` | discovers exactly the adapters present; raises on unknown name; `list_available` returns sorted list | haiku |
| `tests/inspect/test_stats.py` | known 10-window arrays → assert exact F1, confusion matrix, agreement values | sonnet |
| `tests/inspect/test_runner_integration.py` | load real val parquet, 5 windows, 2 stub adapters → InspectionResults has correct shapes; no crash on 0-positive-label slice | sonnet |

No test loads real checkpoints (heavy, slow). Adapters are tested with stubs. One integration smoke test exercises the full runner with real windowing logic but stub predictions.

---

## Risk register

| Risk | Likelihood | Blast radius | Reversibility | Mitigation |
|------|-----------|-------------|---------------|-----------|
| OHLCV normalisation mismatch between adapter and training | Medium | Silently wrong predictions | Easy (code fix) | Unit test adapter output shape; visual sanity check on known FVG window |
| XGBoost `predict_proba` returns (N, num_classes) with different class ordering | Low | Wrong metric assignment | Easy | Assert `clf.classes_` == `[0, 1, 2]` at load time |
| `_window_generator` import breaks (private function) | Low | Runner fails to build | Easy | Copy/expose as `src.data.window.iter_windows` public function |
| 10K HTML files if top-K not respected | Low | Slow, disk fill | Easy | Enforce top-K cap in report.py, default 20 |
| Plotly standalone HTML too large for many windows | Low | Disk usage | Easy | Use `include_plotlyjs='cdn'` — ~5KB per file instead of 3MB |

---

## Pre-mortem

- Runner silently reuses a stale normalisation parameter from training (e.g., wrong min/max) — results look plausible but F1 is unexpectedly low. Mitigation: verify adapter F1 matches the number from `train_lstm.py` evaluation on the same test split.
- XGBoost adapter receives normalised windows (not raw) and `extract_window_features` produces garbage features — predictions look wrong on the viz. Mitigation: assert `windows.min() >= -5 and windows.max() <= 5` triggers a warning that normalised input was passed.
- Disagreement explorer surfaces trivially uninteresting windows (all models predict class 0, none agree on which non-zero class) because most windows are class 0. Mitigation: compute disagreement only among FVG-positive predictions; add `--disagree-on-fvg-only` flag.
- `reports/` directory fills rapidly if tool is run many times. Mitigation: document that `reports/inspect/` is not committed and user should purge periodically.
