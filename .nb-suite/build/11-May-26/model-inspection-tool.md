# Build Log: Model Inspection / Live-Comparison Tool
Date: 11-May-26
Plan: `.nb-suite/plan/11-May-26/model-inspection-tool.md`

## Status: COMPLETE

## Files created

### Core package
- `src/inspect/__init__.py` (pre-existed, empty)
- `src/inspect/base.py` — ModelAdapter ABC
- `src/inspect/registry.py` — auto-discovery via pkgutil, `list_available`, `load_adapters`
- `src/inspect/runner.py` — window builder + inference orchestrator, InspectionResults dataclass
- `src/inspect/stats.py` — F1/confusion/agreement/disagreement computation
- `src/inspect/viz.py` — Plotly candlestick HTML per window (include_plotlyjs='cdn')
- `src/inspect/report.py` — markdown summary + top-K disagreement plots
- `src/inspect/adapters/__init__.py` (pre-existed, updated comment)
- `src/inspect/adapters/lstm_adapter.py` — wraps FVGLSTMClassifier, CPU inference
- `src/inspect/adapters/xgboost_adapter.py` — wraps XGBoostFVGClassifier, auto-subprocess on 3.14+
- `src/inspect/adapters/_xgb_worker.py` — subprocess worker for xgboost isolation

### CLI
- `scripts/inspect_models.py` — full CLI with argparse, dataset loading, label auto-compute

### Tests
- `tests/inspect/__init__.py`
- `tests/inspect/test_base_adapter.py` — 7 tests
- `tests/inspect/test_registry.py` — 6 tests
- `tests/inspect/test_stats.py` — 11 tests
- `tests/inspect/test_runner_integration.py` — 10 tests (1 requires val parquet)

## Test results
34/34 passed (uv run pytest tests/inspect/ -v)

## Deviations from plan

### drop_cross_session default changed to False
Plan assumed session-gap filtering. SPY H1 60-bar windows always span overnight
gaps (60 bars = 10 trading days). Training scripts use drop_cross_session_windows=False.
Keeping True produces 0 windows on all standard parquets. Runner default is now False,
matching training.

### Subprocess isolation implemented for Python 3.14+
Plan noted: "Only required on Python 3.14+. Current env (Python 3.12 per lock) is
unaffected." Actual env is Python 3.14.4. Loading xgboost model file after torch is
imported causes segfault (confirmed). Implemented subprocess worker (_xgb_worker.py)
that runs in a fresh process without torch. Protocol: stdin binary (N + float32 windows),
stdout binary (float32 probas). Per-call overhead ~0.5s — acceptable for inspection.

### No test loads real checkpoints
As planned. All test files use stub adapters. One integration test loads real val
parquet for windowing (skipped if parquet absent).

## Example invocations (verified working)

```bash
# List adapters
uv run python3 scripts/inspect_models.py --list
# Available adapters:
#   lstm
#   xgboost

# Single model
uv run python3 scripts/inspect_models.py --models lstm --dataset test \
  --start 2024-01-01 --end 2024-01-31 --top-k 5
# lstm  F1_bull=0.8108  F1_bear=0.8333  F1_fvg_macro=0.8221  F1_binary=0.8163

# Both models on test slice
uv run python3 scripts/inspect_models.py --models lstm xgboost --dataset test \
  --start 2024-01-01 --end 2024-02-28 --top-k 5
# lstm     F1_bull=0.8095  F1_bear=0.7429  F1_fvg_macro=0.7762  F1_binary=0.7899
# xgboost  F1_bull=0.5769  F1_bear=0.4898  F1_fvg_macro=0.5334  F1_binary=0.5621
# Overall agreement: 0.683
```

## Observations
- LSTM substantially outperforms XGBoost on the Jan-Feb 2024 test slice (binary F1: 0.79 vs 0.56)
- 68.3% agreement between models on this slice
- XGBoost subprocess adds latency but works correctly — consider caching worker process
  for repeated calls in future versions
