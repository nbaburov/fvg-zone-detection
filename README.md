# SMC Data Challenge — Auto-Detecting FVG Zones on Stock Charts

An AI system that automatically spots **Fair Value Gap (FVG)** zones on hourly stock candles and labels them with a confidence score, plus interactive chart overlays. **It is not price prediction** — it finds *where* these zones are, not which way price will move.

> **FVG** = a price gap left by a fast move that traders watch for price to revisit (a so-called "institutional footprint"). Smart-money traders draw these by hand; this project automates the detection and tests it rigorously.

Individual semester project — Fontys ICT, Data Science & AI, 3rd year. New to the project? **Start with [`docs/overview.md`](docs/overview.md).**

## The headline finding (plain)

The task is **data-hungry, not in need of a bigger model.** A simple model (gradient boosting) wins on this small dataset, and the neural networks **improve when given more data** (we proved this by pooling four tickers). The chosen deliverable is the best neural-net detector (**CNN-LSTM**, F1 0.675 on unseen data); gradient boosting (**XGBoost**, 0.738) is the mandatory control. See [`docs/models.md`](docs/models.md) for the full story and [`docs/evaluation.md`](docs/evaluation.md) for how we tested it fairly.

## Docs

| Doc | Read it for |
|---|---|
| [`docs/overview.md`](docs/overview.md) | **Start here.** What the project is, in plain words, for any reader |
| [`docs/data.md`](docs/data.md) | Where the data comes from, how candles are labelled, the schema + splits |
| [`docs/architecture.md`](docs/architecture.md) | The software: what lives where, how data flows (developer view) |
| [`docs/models.md`](docs/models.md) | The five models — what each is + why, their results, diagrams, the honest diagnosis |
| [`docs/evaluation.md`](docs/evaluation.md) | How we tested fairly: no time-leakage, multi-seed, confidence intervals, the G1–G10 validation sprint |
| [`docs/trading-simulation.md`](docs/trading-simulation.md) | Exploratory study: would trading on these signals have made money? (honest: mostly no) |
| [`docs/assignment.md`](docs/assignment.md) | The course brief + deadlines |
| `CLAUDE.md` | Conventions for AI-assisted development |
| `.nb/` | Research, plans, and build logs (provenance trail, not front-facing docs) |

## Stack

| Layer | Tool (what it does) |
|---|---|
| Language | Python 3.12 |
| Data | `alpaca-py` (price data), `exchange_calendars` (trading-hours), `pandas`/`numpy`/`pyarrow` |
| ML | `torch` (neural nets), `xgboost` (gradient boosting), `scikit-learn` (metrics) |
| Charts | `plotly` |
| Tests | `pytest` |

## Quick start

```bash
# 1. Clone + virtual environment
git clone <repo> && cd smc-data-challenge
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Configure data access (free Alpaca paper account)
cp .env.example .env        # then fill ALPACA_API_KEY + ALPACA_SECRET_KEY

# 4. Build the dataset (downloads price data, labels it, splits it)
python -c "from src.data.pipeline import build_pipeline; build_pipeline()"
# (Optional) pooled multi-ticker dataset (the data-bound lever):
python -c "from src.data.pipeline import build_multi_symbol_pipeline; build_multi_symbol_pipeline(['SPY','QQQ','IWM','DIA'], start='2016-01-01')"

# 5. Train
python scripts/training/train_xgboost.py
python scripts/training/train_lstm.py
# (Optional) score multi-ticker models on the fixed SPY test set:
python scripts/rigor/eval_spy_test.py --models cnn_lstm lstm transformer xgb

# 6. Inspect on unseen data (all 5 architectures), with trade-outcome simulation
python scripts/inspect_models.py --dataset test --lookahead-bars 20 \
    --models cnn_lstm lstm transformer xgboost

# 6b. Compare exit strategies with realistic execution (see trading-simulation.md)
python scripts/inspect_models.py --dataset test --all-exit-strategies --realistic --all-seeds \
    --models cnn_lstm lstm transformer xgboost

# 7. (Optional) live paper-trade during market hours
python scripts/paper_trade.py --model lstm:checkpoints/lstm/lstm_seed42.pt --session demo --dry-run
```

Results land in `reports/inspect/<timestamp>/` — open `summary.md` for the metric + outcome tables and `plots/` for the chart overlays.

## Project layout

```
src/data/      get + label + split + window the price data (incl. multi-ticker pooling)
src/features/  hand-engineered features for the gradient-boosting model
src/models/    the five architectures (LSTM, CNN-LSTM, Transformer, xLSTM, XGBoost)
src/training/  loss functions, early stopping, seeding
src/strategy/  the FVG trade-exit rules + realism guards (single source of truth for the trade-sim)
src/inspect/   offline analysis: run any model on unseen data, simulate trades
src/live/      live paper-trading harness (Alpaca paper account)
scripts/       command-line entry points (data, training, rigor/sweeps, inspect, paper-trade)
notebooks/     the project narrative (00–07, incl. presentation decks)
tests/         pytest suite, mirrors src/
docs/          the docs indexed above
```
Full per-folder explanation in [`docs/architecture.md`](docs/architecture.md).

## Testing

```bash
make test                  # full suite (XGBoost tests run isolated — see CLAUDE.md)
pytest tests/strategy/ -q  # the trade-exit logic
pytest tests/inspect/ -q   # the inspection toolkit
```

## The non-negotiables (why the results are trustworthy)

- **No peeking into the future.** Time-ordered split only (train 2016–2021 / validate 2022 / test 2023–2025), never shuffled; labels use only information available at the candle's time. Enforced by a mandatory test.
- **The right metric.** F1 on the rare FVG classes — *not* accuracy (97% of candles are "none", so always-say-none scores 97% but is useless).
- **No lucky runs.** Five random seeds per model; confidence intervals via 1000-iteration bootstrap.
- **Honest reporting.** What failed is reported (Transformer instability, xLSTM underfitting), and uncertain results are labelled "supported, not proven." See [`docs/evaluation.md`](docs/evaluation.md).

## Author

Nikola Baburov — Fontys ICT, Data Science & AI, Semester 6.

## Licence

Academic project. No licence granted for redistribution.
