# SMC Data Challenge — FVG Auto-Detection on SPY H1

Deep-learning system that auto-detects **Smart Money Concepts (SMC) Fair Value Gap (FVG)** zones on SPY 1-hour candlestick data. Output = zone labels + confidence scores + Plotly chart overlays. **Not price prediction.**

Individual semester project, Fontys ICT 3rd year. Deadlines through 2026-06-20.

> SMC = price-action framework derived from ICT methodology. FVG = 3-candle gap pattern indicating institutional order-flow imbalance. Manual SMC traders draw these by hand; this project automates detection at scale.

## Status

| Component | State |
|-----------|-------|
| Data pipeline (Alpaca → H1 → splits) | ✅ |
| Labellers (raw `FVGLabeller`, validated `ValidFVGLabeller`) | ✅ |
| Gold annotation set (75 rows, κ = 1.0 vs raw `FVGLabeller`, κ = 0.89 vs `ValidFVGLabeller`) | ✅ |
| XGBoost baseline | ✅ trained, test F1 binary ≈ 0.62 |
| LSTM baseline | ✅ trained, test Macro F1 = **0.824** |
| Inspector tool (offline metrics + TP/SL outcome sim) | ✅ |
| Live paper-trading harness (Alpaca paper) | ✅ implemented, not yet run live |
| CNN-LSTM / xLSTM / Transformer | planned |
| Migration to ValidFVG training labels | planned |

See [`docs/models-status.md`](docs/models-status.md) for full results.

## Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.12 or 3.14 |
| Data | `alpaca-py` (Alpaca free tier, paper), `exchange_calendars`, `pandas`, `numpy`, `pyarrow` |
| ML | `torch`, `xgboost`, `scikit-learn` |
| Visualisation | `plotly` |
| Testing | `pytest` |

## Quick start

```bash
# 1. Clone + venv
git clone <repo>
cd smc-data-challenge
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install alpaca-py exchange_calendars torch pandas numpy plotly scikit-learn xgboost pytest

# 3. Configure Alpaca
cp .env.example .env
# fill ALPACA_API_KEY and ALPACA_SECRET_KEY (paper account)

# 4. Build the dataset (downloads ~5 GB of SPY 1-min bars on first run)
python -c "from src.data.pipeline import build_pipeline; build_pipeline()"

# 5. Train baselines
python scripts/train_xgboost.py
python scripts/train_lstm.py

# 6. Inspect on unseen data
python scripts/inspect_models.py \
    --models lstm xgboost \
    --dataset test \
    --start 2024-01-01 --end 2024-02-29 \
    --lookahead-bars 20 --tp-rr 2.0

# 7. (Optional) Live paper trade during market hours
python scripts/paper_trade.py \
    --model lstm:checkpoints/lstm/lstm_seed42.pt \
    --session lstm-001 \
    --dry-run
```

Open `reports/inspect/<timestamp>/plots/timeline_lstm.html` to see model trades on the full slice. Open `summary.md` for F1 / outcome tables.

## Docs

| Doc | Purpose |
|-----|---------|
| [`docs/architecture.md`](docs/architecture.md) | Code structure, folder layout, scripts table, Mermaid diagrams for components + data flow |
| [`docs/data-model.md`](docs/data-model.md) | On-disk parquet schemas, in-memory shapes, adapter contract, live session log format |
| [`docs/models-status.md`](docs/models-status.md) | Splits, class balance, weights, trained model configs, current test results, what's validated and what isn't |
| [`docs/idea.md`](docs/idea.md) | Original project brief — SMC background, novelty argument, scope clarification |
| [`docs/assignment.md`](docs/assignment.md) | Fontys assignment specification + deadlines |
| [`docs/fvg-label-guide.md`](docs/fvg-label-guide.md) | Manual annotation guide for the gold set |
| `CLAUDE.md` | Conventions for AI-assisted development (Claude Code rules) |
| `.nb-suite/` | Research + plan + build logs (not docs but useful context) |

## Project layout (short version)

```
src/data/      acquisition, labelling, splitting, windowing
src/features/  feature engineering for non-DL models
src/models/    architectures (LSTM, XGBoost)
src/training/  loss, early stop, train utils
src/inspect/   offline model inspection toolkit
src/live/      live paper-trading harness
scripts/       CLI entry points (train, inspect, paper_trade)
notebooks/     CRISP-DM presentation notebooks
tests/         pytest, mirrors src/
docs/          this directory
```

Full tree + per-folder explanation in [`docs/architecture.md`](docs/architecture.md).

## Testing

```bash
pytest                         # all
pytest tests/inspect/ -q       # inspector (34 tests)
pytest tests/live/ -q          # live harness (47 tests)
pytest tests/data/labels/ -q   # labeller suites (incl. anti-lookahead fixtures)
```

## Critical constraints (the non-negotiables)

- **Temporal split only.** Train 2018–2022 / val 2022–2023 / test 2023–2024. No shuffling.
- **No lookahead.** Labels use only information available at candle close. ValidFVGLabeller places labels at N+2 (reaction candle), raw FVGLabeller at N+1. Mandatory pytest fixture enforces this.
- **F1 on minority class is the headline metric.** Accuracy is meaningless on 75/15/10 imbalance.
- **Weighted cross-entropy** for the DL models; sample weights for XGBoost. Inverse-frequency weights persisted to `class_weights.json`.
- **Live windowing = training windowing.** The live harness imports `normalise_window` and `_MAX_INTRA_WINDOW_GAP_MINUTES` directly from the training package. No duplication, no drift.

## Hardware notes

Targets Apple Silicon. MPS-specific gotchas (documented in `CLAUDE.md`):
- `nn.LSTMCell` slower than CPU → use sequence-batched `nn.LSTM`
- `nn.MultiheadAttention` + bool mask + dropout produces NaN → add `x + 0` post-attention
- Deterministic mode = 8× slowdown → CPU for reproducibility runs

## Author

Nikola Baburov — Fontys ICT, Data Science & AI, 3rd year, Semester 6.

## Licence

Academic project. No licence granted for redistribution.
