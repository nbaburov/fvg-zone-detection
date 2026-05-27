# Project Idea: Automating Smart Money Concepts on SPY

**Student:** Nikola Baburov | **April 2026** | Individual

---

## Big Idea

**Smart Money Concepts (SMC)** = price-action framework derived from ICT methodology. Identifies footprints institutional players leave in candlestick data.

Key structures:
- **FVG** — Fair Value Gap
- **OB** — Order Block
- **BOS** — Break of Structure
- **CHoCH** — Change of Character
- **Liquidity Pools**
- **Supply & Demand Zones**

**Problem:** Every SMC practitioner draws these by hand. Subjective. Unbacktestable at scale. Rule-based scripts exist but are context-blind — same rule treats high-volume post-BOS FVG same as low-volume drift FVG. Wrong.

**Goal:** Deep learning system that automates the **FVG zone-detection sub-task** of SMC on SPY. Output = zone labels + confidence scores + interactive chart overlays.

**Scope clarification (mandatory framing):** Project automates *identification* of FVG zones — the pattern-recognition primitive inside SMC. It does NOT automate SMC trading, which additionally requires multi-timeframe trade qualification (HTF bias, LTF entry trigger). Multi-TF qualification = explicit out-of-scope for MVP. See `.nb/research/8-May-26/timeframe-deep.md`.

**NOT price prediction.** Model identifies *what FVG zones exist and where*, not direction.

---

## Why Novel

- Academic DL finance literature = price prediction + classical indicators. Saturated.
- SMC has zero published academic treatment, zero DL applications.
- Structure detection = sequence labelling problem → needs temporal context across hundreds of candles → exactly where RNN/attention architectures win over classical methods.
- Related work: Chen & Tsai (2020) 90.7% accuracy on candlestick CNN; Ramadhan et al. (2022) 82.7% CNN-LSTM. This extends into harder, context-dependent domain.

---

## Data

- **Source:** SPY hourly OHLCV 2018–present via **Alpaca Markets free tier** (`alpaca-py` SDK, paper account, `adjustment="raw"`). ~13,500 candles target. (yfinance rejected 8-May-26 — Yahoo caps H1 history at 730 days; see `.nb/research/8-May-26/data-source-validation.md`.)
- **Timeframe:** **H1 single-frame.** FVG identification is a self-contained 3-candle geometric pattern in canonical SMC literature — multi-TF is used for trade qualification, not identification. H1 is the SMC zone-identification layer. ~13.5k candles fits LSTM/xLSTM minimum. Confirmed via Deep multi-perspective research: `.nb/research/8-May-26/timeframe-deep.md`.
- **Labels:** Generated programmatically via **custom vectorised FVG detector** (`src/data/label.py`). `smartmoneyconcepts` library rejected 8-May-26 — v0.0.27 confirmed to use 1-bar future lookahead (`shift(-1)`); PR #95 fix unmerged. Custom detector places label at index N+1 (first bar where 3-candle pattern is closed). Approach is still **weak supervision** (Snorkel-style programmatic labelling) — no manual annotation. See `.nb/research/8-May-26/smc-library-validation.md`.

Multi-timeframe (H1 + Daily fusion) = optional extension. Cheapest path = broadcast Daily ATR + direction as 2 extra input features (~2.5 days). NOT Temporal Fusion Transformer. Trigger conditions: H1 baseline F1 > 0.45 AND val plateau AND regime-correlated errors.

---

## MVP (First Step)

Scope: **FVG detection only**. Clearest definition, most consistent labels, immediately verifiable on chart.

Pipeline:
1. Download SPY H1 OHLCV 2018–2024 via yfinance → remove zero-volume candles → rolling min-max normalisation
2. Apply `smart-money-concepts` → per-candle ternary labels: `bullish / bearish / none`
3. Format as 60-candle sliding windows (60×5 tensors) → train LSTM with dropout + weighted cross-entropy (handles class imbalance)
4. Plotly candlestick chart — rule-generated zones vs model-predicted zones side by side

---

## Model Stages (revised per R6, 8 May 2026)

| Stage | Architecture | Notes |
|---|---|---|
| 0 | **XGBoost baseline (NEW)** | Boring control. Effective sample size ~117 (stride=1 = 60× overlap) → small-data regime likely favors gradient boosting. Mandatory academic control. |
| 1 | LSTM (2-layer stacked, 64 units, dropout) | Sequential DL baseline |
| 2 | **CNN-LSTM (1D conv kernel 3–5 + LSTM)** — expected best DL | FVGs are 3-candle local formations → conv front-end directly encodes the pattern's locality |
| 3 | **xLSTM** (see below) | Teacher feedback — explore. 2025 MDPI study found it *worse* than vanilla LSTM on short-term financial forecasting; included for academic completeness, don't expect it to win |
| 4 | Transformer (lightweight, time-permitting) | Tiny only — small-data regime risks overfit. Skippable. |

### xLSTM — Teacher Feedback

**Feedback:** AI teacher/expert flagged that since LSTM is core architecture, xLSTM (Extended LSTM) should be explored as a direct upgrade candidate.

**What it is:** xLSTM (Beck et al., 2024) extends classic LSTM with two new cell types:
- **sLSTM** — scalar memory, new gating via exponential gates, normaliser state for numerical stability
- **mLSTM** — matrix memory with covariance update rule, fully parallelisable (no sequential dependency like classic LSTM)

Combined into xLSTM blocks and stacked into architecture. Designed to close gap between LSTMs and Transformers on long-sequence tasks while keeping recurrent inductive bias.

**Why relevant here:**
- Financial time series = long sequences with regime-dependent structure. xLSTM's matrix memory can store richer temporal context than classic LSTM hidden state.
- mLSTM is parallelisable → faster training than classic LSTM on same hardware.
- Recurrent structure still preferred over Transformer for shorter sequences / lower data regimes (SPY H1 ~16k candles is not large).

**Plan:** After LSTM baseline validated, swap in xLSTM and compare F1 on FVG minority class directly. If xLSTM > LSTM baseline, use as primary architecture going forward.

**Reference:** Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended long short-term memory. *arXiv:2405.04517*.

---

## Known Bottlenecks

- **Class imbalance:** FVGs in ~5–10% candles. Primary metric = F1 on positive class, NOT accuracy.
- **Lookahead leakage:** Some labels confirmed after the fact. Pipeline must use only info available at candle time.
- **Label noise:** Rule-based labels are ceiling on performance.
- **Non-stationarity:** Rolling normalisation helps but doesn't fully solve regime drift.

---

## Success Criteria

- Pipeline runs end-to-end from single script, no manual intervention
- Three eval layers reported separately on 2023–2024 holdout:
  1. **Detection F1** — does model find FVG zones (binary)
  2. **Bias-conditional F1** — given detection, is bull/bear direction right
  3. **Per-class F1** (full ternary, primary headline) — meaningfully above naive baseline
- Plotly chart renders bias on each zone (green = bullish/buy bias, red = bearish/sell bias) + confidence tooltip
- At least one case where model differs from custom rule detector, with plausible explanation

No specific F1 threshold committed pre-baseline — methodologically unsound without prior benchmark.

---

## Tools

Python, PyTorch, **alpaca-py**, pandas, numpy, `smart-money-concepts`, Plotly, scikit-learn

---

## Ethics

- **Market manipulation risk:** Automating liquidity detection at institutional scale could engineer the stop hunts it detects. Strictly research/educational.
- **Data bias:** Temporal train/test split enforced, limitations documented honestly.
- **Retail harm:** Results include uncertainty estimates, no trading edge claimed.

---

## Key References

- Attridge (2023) — `smart-money-concepts` library
- Fischer & Krauss (2018) — LSTM on S&P 500, excess returns vanished post-2010
- Chen & Tsai (2020) — CNN candlestick classification, 90.7%
- Ramadhan et al. (2022) — CNN-LSTM candlestick, 82.7%
- Lim et al. (2021) — Temporal Fusion Transformer
- Ratner et al. (2020) — Snorkel weak supervision framework
