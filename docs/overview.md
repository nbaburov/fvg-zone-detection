# Project Overview

## What this project is

This project builds an AI system that automatically finds and labels a specific price pattern called a Fair Value Gap (FVG) on stock price charts, and draws those zones on an interactive chart with a confidence score. The stock used is SPY, an exchange-traded fund that tracks the S&P 500 index (the 500 largest US companies).

The system is **not** a price predictor. It does not say "the price will go up". It finds *where* a particular pattern exists on a historical chart and scores how confident it is that the pattern is valid.

## What a Fair Value Gap is

When a large, fast price move happens (say, a sudden surge upward across three consecutive candles, where each candle represents one hour of trading), it can leave a visible "gap" in the chart: a range of prices that were barely traded. Traders who follow a style called Smart Money Concepts (SMC) believe these gaps act as magnets: price tends to return to them later because large institutional buyers or sellers left unfilled orders there.

Detecting these gaps manually is straightforward for a single chart but tedious at scale: you scan three consecutive candles and check whether the high of the first and the low of the third do not overlap. Additional filters confirm the gap is meaningful (the right size, in a trending context, not immediately filled). That rule-based structure makes it a well-defined pattern-recognition task, which is exactly where AI excels.

## Why it is interesting

Smart Money Concepts and FVGs are extremely popular with retail traders (they appear in millions of social media posts and trading guides) yet there is almost no academic or AI research on them. This project applies proper machine-learning discipline to that gap: clean historical data, a leak-free labelling system, five different AI model architectures compared head-to-head, and statistical evaluation across multiple random seeds. It is, to our knowledge, one of the first rigorous ML treatments of this trading concept.

## What was built

- **Data pipeline.** Downloads nine years of hourly SPY price data (2016-2025), applies the FVG detection rules to label (assign a category to) each bar: bullish FVG, bearish FVG, or none. Splits the data chronologically into training, validation, and test sets with no data from the future used to train or evaluate.
- **Five AI models.** A gradient-boosted tree model (XGBoost, which works on hand-crafted numeric features), a recurrent network (LSTM), a combined convolutional-recurrent network (CNN-LSTM), a Transformer (the architecture behind modern language models), and a newer variant called xLSTM. All five were trained and evaluated on the same data under the same conditions.
- **Rigorous evaluation.** Because valid FVGs are rare (roughly 3% of hourly bars), accuracy is a misleading measure. A model that always says "no FVG" would be 97% accurate but useless. The primary metric is F1 score on the minority class, run across five different random seeds (each a different random starting point for training, so we don't report a lucky one) to check consistency.
- **Interactive chart tool.** A Plotly-based viewer that overlays detected FVG zones on the price chart, so results can be inspected visually.
- **Trading simulation study.** An exploratory analysis asking: if you acted on these FVG signals, what would the trade outcomes look like? This is a retrospective study, not a live strategy.

## Headline finding

The most important result is about the nature of the task, not the models: this is a **data-hungry problem**, not a **needs-a-bigger-model** problem. On the original nine years of single-stock hourly data, the simplest model (XGBoost, F1 = 0.72) beat all five neural networks because that data provides only a small effective number of independent training examples (roughly 117 by one estimate). Although there are around 13,000 hourly bars in the training period, the 60-bar windows overlap heavily: each new window shifts by just one bar, so they are far from independent, leaving roughly 117 truly non-overlapping examples.

Two follow-up experiments confirmed the diagnosis by adding *data* rather than model complexity:

- **Pooling four tickers** (SPY + QQQ + IWM + DIA into one training set) lifted every recurrent network: the CNN-LSTM rose from F1 0.64 to 0.675. The Transformer, held back by training instability rather than by data, did not move.
- **Resampling to faster 5-minute candles** (~12x more labelled bars, at zero data-collection cost) lifted the neural networks further and *reversed the ranking*: the best neural network (a tuned CNN-LSTM, F1 = 0.713 - the project's best deep-learning result) overtook XGBoost at that timeframe.

So more data, not a bigger model, is what moves the needle. The chosen deliverable is the **CNN-LSTM** detector (the "carrier" architecture carried forward to the final product); **XGBoost** (F1 = 0.738 on hourly data) remains the mandatory control. Full numbers and the honest diagnosis are in `docs/models.md`.

## Where to go next

| Document | What it covers |
|---|---|
| `docs/data.md` | How the price data is downloaded, resampled, and labelled; what each column means |
| `docs/models.md` | All five model results, per-model summaries, and the data-bound diagnosis |
| `docs/trading-simulation.md` | The retrospective study of what trading on FVG signals would have looked like |
| `docs/architecture.md` | Technical layout of the codebase for developers |

**Suggested reading order for a non-technical reader:**
1. `docs/overview.md` (this page) - what the project is and why it matters
2. `docs/data.md` - where the data comes from and how patterns are labelled
3. `docs/models.md` - what each AI model is and what it scored
4. `docs/evaluation.md` - how the testing was kept fair
5. `docs/trading-simulation.md` - what would have happened if you traded on the signals
