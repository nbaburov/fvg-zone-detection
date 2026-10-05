# FVG Zone Detection

Finds Fair Value Gap (FVG) zones on stock candles and labels each with a confidence score, with interactive chart overlays. It finds *where* the zones are; it does not predict which way price moves.

> **FVG:** a price gap left by a fast three-candle move, which Smart Money Concepts (SMC) traders expect price to revisit. Traders draw these by hand; this project detects them and tests the detection rigorously.

Individual project, Fontys University of Applied Sciences (Data Science & AI, 2026).

Write-up: [Teaching a machine to see Fair Value Gaps](https://nb.nb-limited.com/writing/teaching-a-machine-to-see-fair-value-gaps).

## What it does

Five architectures detect FVG zones on SPY hourly, 15-minute and 5-minute candles: XGBoost, LSTM, CNN-LSTM, Transformer and xLSTM.

**Headline finding:** the task is data-bound, not model-bound. Gradient boosting wins on the small single-ticker set, and the neural networks improve when four tickers (SPY, QQQ, IWM, DIA) are pooled. The chosen detector is the CNN-LSTM, F1 0.675 [0.637, 0.710] on unseen 2023 to 2025 data; XGBoost, the control, reaches 0.738 [0.689, 0.779].

*Revised in October 2026: the limit is the 60-candle window, not the data. See the [evaluation note](#evaluation-note-october-2026).*

Why the numbers hold up:

- **No lookahead.** Time-ordered split (train 2016 to 2021, validate 2022, test 2023 to 2025), never shuffled; labels use only information available at the candle's time. Enforced by a test.
- **The right metric.** F1 on the rare FVG classes; about 97% of candles are "none", so accuracy is meaningless.
- **No lucky runs.** Five seeds per model, 1,000-iteration bootstrap confidence intervals.
- **Honest reporting.** Failures (Transformer instability, xLSTM underfitting) are reported, and uncertain results are labelled "supported, not proven".

A trading simulation checks whether acting on the signals would have made money (mostly no), and a paper-trading harness runs a model live against an Alpaca paper account.

| Doc | Read it for |
|---|---|
| [`docs/overview.md`](docs/overview.md) | start here: the project in plain words |
| [`docs/data.md`](docs/data.md) | data source, labelling, schema and splits |
| [`docs/models.md`](docs/models.md) | the five models, results and diagnosis |
| [`docs/evaluation.md`](docs/evaluation.md) | the evaluation protocol and the G1 to G10 validation sprint |
| [`docs/trading-simulation.md`](docs/trading-simulation.md) | the exploratory trading study |
| [`docs/presenation/`](docs/presenation/) | the final presentation (script, slides, notes) |

## Quickstart

Requires Python 3.12 and a free [Alpaca](https://alpaca.markets) paper account for market data. Market data is not included in this repository; the pipeline downloads and labels it.

```bash
git clone https://github.com/nbaburov/fvg-zone-detection.git
cd fvg-zone-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set ALPACA_API_KEY and ALPACA_SECRET_KEY

# build the dataset (download, label, split)
python -c "from src.data.pipeline import build_pipeline; build_pipeline()"
python -c "from src.data.pipeline import build_multi_symbol_pipeline; build_multi_symbol_pipeline(['SPY','QQQ','IWM','DIA'], start='2016-01-01')"

# train and inspect on unseen data
python scripts/training/train_xgboost.py
python scripts/training/train_lstm.py
python scripts/inspect_models.py --dataset test --lookahead-bars 20 --models cnn_lstm lstm transformer xgboost

make test                   # about 1,080 tests; data-dependent ones skip until the dataset is built
```

Trained checkpoints are included in `checkpoints/`. Inspection results land in `reports/inspect/<timestamp>/` (`summary.md` plus chart overlays). The offline presentation demo is built with `python scripts/demo_animator.py`.

## Architecture

```
src/data/      download, label, split and window the price data (incl. multi-ticker pooling)
src/features/  hand-engineered features for the gradient-boosting model
src/models/    LSTM, CNN-LSTM, Transformer, xLSTM, XGBoost
src/training/  losses, early stopping, seeding
src/strategy/  FVG trade-exit rules and realism guards
src/inspect/   offline analysis and trade simulation
src/live/      paper-trading harness
src/demo/      offline FVG-replay presentation demo
scripts/       command-line entry points
notebooks/     the project narrative
tests/         pytest suite, mirrors src/
```

Stack: Python 3.12, PyTorch, XGBoost, scikit-learn, pandas and pyarrow, `alpaca-py` for market data, `exchange_calendars` for trading hours, Plotly for charts, pytest. Details in [`docs/architecture.md`](docs/architecture.md); conventions and the test gate in [`docs/development.md`](docs/development.md).

## Configuration

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | yes (data download) | Alpaca paper-account key |
| `ALPACA_SECRET_KEY` | yes (data download) | Alpaca paper-account secret |
| `ALPACA_PAPER` | no | keep `true`; `false` enables live trading |

## Status

Finished individual project, frozen at the June 2026 hand-in plus later dependency and documentation updates. It receives no feature work. Trained checkpoints are included; market data is not, and the pipeline rebuilds it.

## Evaluation note (October 2026)

A re-read after the hand-in found a comparison the project never ran: the labelling rule itself, given only the window the models see.

- **The rule beats every model on the same input.** `ValidFVGLabeller` run on just the last 60 candles before each SPY H1 test position (n = 5,198) scores macro-F1 0.821. The four pooled models, on the same 60 candles, score 0.577 to 0.738, and the untuned xLSTM 0.369.
- **With a longer window it is exact.** On a common position set the rule scores 0.530, 0.682, 0.821, 0.958, 1.000 and 1.000 for 20, 40, 60, 80, 100 and 120 candles. The break-of-structure criterion looks back a 50-candle swing plus a 50-candle break window, about 100 candles, so a 60-candle window cannot show it in full.
- **So the June verdict is better read as input-bound than data-bound.** Pooling tickers and finer timeframes help because more examples help any model approximate the rule; they cannot add history the window does not hold.
- **The best trading cell is not evidence of an edge.** XGBoost on 15-minute SPY (+58R) won 34.5% of its trades, against a 33.3% break-even for the 2:1 bracket before costs, and was the best of 192 setups, 54 of which ended positive.

Reproduce with `python scripts/rigor/eval/rule_window_baseline.py` once the dataset is built. The obvious next steps: use the rule with at least 100 candles as the detector, and point a model at the question the rule cannot answer, whether a valid gap holds.

## Author

Nikola Baburov.

## License

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may read, run, modify and share it for any noncommercial purpose, as long as the copyright notice comes along.

This repository is a showcase, so it doesn't take issues or pull requests. Forks are welcome under the license. Commercial use needs a separate paid license; contact [NB Limited](https://nb-limited.com).
