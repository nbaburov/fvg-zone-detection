# Labeled FVG / SMC Datasets — Research

> **Ready for /nb:plan.**
> Question: Do pre-labeled FVG or SMC datasets exist on Kaggle, HuggingFace, GitHub, or in academic papers that could replace or augment the custom rule-based labeler?
> Verdict: No usable pre-labeled FVG dataset exists for this project. Every candidate either lacks FVG labels entirely, uses a lookahead-contaminated rule, targets the wrong asset/timeframe, or is not publicly downloadable. Must iterate the custom detector — not bypass it.

**Confidence:** High
**Why this confidence:** Searched Kaggle, HuggingFace, GitHub, Google Scholar / ResearchGate, and web broadly. The only paper directly on FVG+DL (Suresh et al. ICIVC 2025) is paywalled and shares no public dataset. The only Kaggle dataset superficially matching ("Human Labeled OHLCV") is synthetic data with buy/sell annotations, not FVG structure labels. No remaining candidate survived the "does it have actual FVG direction labels on OHLCV bars?" test. Confidence is high that the search space is exhausted at Standard depth; a manual trawl of Kaggle's ~80k finance datasets could surface something missed, but the signal-to-noise ratio of the search results makes a hidden dataset unlikely.
**Depth used:** Standard

---

## Project context

- Custom FVG rule produced kappa = 0.284 vs 75-candle human gold set (threshold = 0.6 required).
- 27/28 disagreements = rule missed FVGs the human annotator found. Rule is too restrictive, not mislabeling positives.
- Asset: SPY H1, 2018-present (~13,500 candles).
- Label target: ternary per-candle (bullish FVG / bearish FVG / none), label at N+1 (causal).
- Need: a dataset that either (a) replaces the rule with pre-labeled FVGs on SPY or a closely related equity asset, or (b) provides a labeled reference set to tune the rule against.

---

## Dataset inventory

### 1. Kaggle — "Human Labeled OHLCV Stock Market Data" (barathanaslan)

**URL:** https://www.kaggle.com/datasets/barathanaslan/human-labeled-synthetic-stock-market-data
**Asset / TF:** Synthetically generated stock charts (not SPY, not real market data)
**Sample count:** Unknown — page did not load full metadata
**Labels:** Human annotations on synthetic charts — likely buy/sell or bullish/bearish directional tags, NOT FVG structure labels
**Labeling method:** Human annotators on generated images; methodology not published
**License:** Not retrieved
**Viable for this project:** No
**Reason:** Synthetic data. No FVG-specific labels. Even if it contained pattern annotations, synthetic OHLCV with image-level labels (chart annotations) would not map to per-candle ternary FVG encoding. No evidence of FVG, order block, BOS, or CHoCH in the label schema.

---

### 2. Kaggle — "Candle Stick Patterns (500+ Unique)" (mineshjethva)

**URL:** https://www.kaggle.com/datasets/mineshjethva/candle-stick-patterns
**Asset / TF:** Not retrieved (page blocked)
**Labels:** Named candlestick patterns (Doji, Hammer, Engulfing, etc.) — traditional TA, not SMC/ICT
**Labeling method:** Rule-based (standard candlestick definitions)
**Viable for this project:** No
**Reason:** FVG is not a traditional candlestick pattern. This dataset covers the Japanese candlestick taxonomy (single-bar and two-bar patterns), not the 3-candle ICT imbalance structure. Labels are incompatible with FVG ternary encoding. No transfer possible.

---

### 3. HuggingFace — JonusNattapong/xauusd-trading-ai-smc-v2

**URL:** https://huggingface.co/JonusNattapong/xauusd-trading-ai-smc-v2
**Type:** MODEL (XGBoost classifier), not a dataset
**Asset / TF:** XAUUSD (Gold), Daily primary + 15m/30m/1m secondary
**Labels:** Binary price direction (rises in 5 days: yes/no)
**FVG handling:** FVG_Size and FVG_Type_Encoded used as *features*, computed from the `smartmoneyconcepts` library — which R2 confirmed has 1-bar lookahead. Labels are binary price direction, NOT FVG structure annotations.
**License:** MIT
**Viable for this project:** No
**Reason:** This is a trained model artifact, not a labeled dataset. The "FVG features" are derived using the lookahead-contaminated library. The binary target (price direction 5 days ahead) is a different ML task from ours (FVG zone detection at the current candle). Asset mismatch (XAUUSD vs SPY). Timeframe mismatch (daily vs H1 SPY).

---

### 4. HuggingFace — darkknight25/trading_dataset_v2

**URL:** https://huggingface.co/datasets/darkknight25/trading_dataset_v2
**Asset / TF:** AAPL (Apple stock), daily candles, 2023–2025
**Sample count:** 157,016 rows (157k entries in JSONL/Parquet format)
**Labels:** Binary buy/sell signals from traditional TA indicators (EMA, MACD, RSI, Bollinger, Stochastic)
**FVG / SMC labels:** None
**License:** MIT
**Viable for this project:** No
**Reason:** No FVG labels. Designed for LLM instruction-tuning (prompt-response pairs), not DL pattern detection. Wrong asset (AAPL vs SPY). Wrong label semantics. No market structure annotations of any kind.

---

### 5. GitHub — starckyang/smc_quant

**URL:** https://github.com/starckyang/smc_quant
**Type:** Trading bot / strategy repository, not a labeled dataset
**Content:** Python code implementing BOS/CHoCH and FVG detection for backtesting
**Labels:** Generated programmatically using the same `smartmoneyconcepts` library (or similar rule logic) — same lookahead problem as R2
**Published CSV dataset:** No
**Viable for this project:** No
**Reason:** Code repository, not a dataset. Any labels it produces would be generated by the same rule class as our existing detector, with likely the same or worse lookahead contamination.

---

### 6. Academic — Suresh et al. "A Deep Learning Approach to Identify Fair Value Gaps (FVGs) in Forex Markets" (ICIVC 2025, pub. Jan 2026)

**URL:** https://link.springer.com/chapter/10.1007/978-3-032-10670-4_36
**Authors:** Navneeth Suresh et al.
**Venue:** ICIVC 2025 (International Conference on Intelligent Vision and Computing), published in Springer proceedings January 6, 2026
**Asset / TF:** Forex (specific pair unknown — page is paywalled and redirects to login)
**Architecture:** CNN applied to candlestick chart images (computer vision approach)
**Public dataset link:** None found — paywalled chapter with no companion repository on GitHub or Zenodo
**Viable for this project:** No (and inaccessible)
**Reason:** Paywalled. No public dataset. Uses a computer vision / image-based approach (CNN on chart images) rather than OHLCV time series, which is architecturally incompatible with our (60, 5) window format. Even if accessible, image-level FVG labels on forex charts would not transfer directly to per-candle ternary labels on SPY H1 OHLCV. The paper is noted as the only peer-reviewed FVG+DL work found — but it contributes no usable dataset.

---

### 7. TradingView / LuxAlgo / HypaTrader — exported SMC labels

**Source:** TradingView export documentation + LuxAlgo SMC indicator
**Finding:** TradingView allows export of raw OHLCV data to CSV (chart data export). LuxAlgo's SMC indicator marks BOS/CHoCH, order blocks, equal highs/lows, and FVGs on chart as visual overlays. However:
- Indicator overlay data (zone coordinates) is NOT directly exportable to CSV via TradingView's standard export. Only OHLCV price bars export cleanly.
- No community-published CSV export of LuxAlgo FVG labels found.
- LuxAlgo's FVG detection is a closed-source Pine Script implementation — the exact detection logic and any lookahead behavior is not auditable.
- No "LuxAlgo FVG labeled dataset" repository exists on GitHub or Kaggle.
**Viable for this project:** No
**Reason:** Label methodology opaque, no lookahead audit possible, no downloadable label CSV exists.

---

### 8. Synthetic FVG benchmark datasets

**Finding:** No published synthetic FVG dataset exists for ML benchmarking. Searched specifically for "synthetic FVG dataset," "generated candles fair value gap benchmark," and "FVG machine learning benchmark." Zero results. The only synthetic data that appeared was the barathanaslan Kaggle dataset (§1 above), which does not contain FVG labels.
**Viable for this project:** Not applicable — does not exist.
**Reason:** FVG is a niche-enough concept that no benchmark dataset has been published for it. The broader candlestick pattern benchmark datasets (e.g., Bulkowski-style classical patterns) do not include FVG as a labeled class.

---

### 9. Adjacent — crypto/forex labeled datasets for transfer learning

**Candidates found:** darkknight25 (§4), JonusNattapong (§3), smc_quant (§5), plus various forex/crypto OHLCV datasets (unlabeled).
**Transfer learning viability assessment:**
- FVG is a geometric pattern defined entirely by relative OHLCV price relationships. The pattern definition is asset-agnostic — a bullish FVG is `high[i-1] < low[i+1] AND close[i] > open[i]` regardless of asset.
- However: (a) no FVG-labeled dataset exists on any asset to transfer from, and (b) even if labels existed on EUR/USD forex, the candle geometry statistics differ from SPY H1 (different volatility regimes, different gap frequencies). A model pre-trained on XAUUSD daily FVGs would need fine-tuning on SPY H1 anyway, which requires labeled SPY H1 data — the very thing we lack.
- Transfer of FVG labels across assets is theoretically plausible but moot: no labeled source dataset exists to transfer from.
**Viable for this project:** No

---

## Recommendation

**No usable pre-labeled dataset found. Must iterate the custom detector.**

The kappa=0.284 result tells us the direction of the problem: the rule has low recall (27/28 disagreements = rule missed FVGs the human found). This is a false negative problem, not a false positive problem. The fix is in the rule, not in switching to a different dataset source.

Concrete next steps in priority order:

1. **Diagnose why the rule misses FVGs.** The 27 missed FVGs need to be examined. Hypotheses:
   - The gap condition `high[i-1] < low[i+1]` is too strict when there is wick overlap (wicks touch but bodies don't overlap — some practitioners count this as FVG).
   - Doji candles on the middle bar (`close ≈ open`) are being excluded by the `close > open` (bullish) condition.
   - Micro-gaps that are technically valid but below some implicit ATR threshold were accepted by the human but rejected by the rule (suggesting the rule may need no explicit size filter, since humans are already including micro-gaps).
   - Examine the 27 missed cases from `data/gold_labels.csv` — group them by the failure mode. This takes 30–60 minutes and directly informs the fix.

2. **Relax the middle candle condition.** The `close > open` constraint for bullish FVGs rejects doji candles. ICT's original definition requires the middle bar to be bullish, but some practitioners define it as any bar where the gap exists regardless of body direction. Try removing or relaxing this constraint and re-compute kappa.

3. **Do NOT add an ATR size filter.** The kappa=0.284 problem is under-detection, not over-detection. Adding a size filter would make it worse.

4. **Re-run kappa on the 75-candle gold set after detector fix.** The gold set was built for exactly this purpose — use it as the iteration target. Target: kappa ≥ 0.6 before proceeding to Phase 4.

5. **If kappa remains below 0.6 after rule relaxation:** extend the gold set to 150 candles with more FVG-rich strata. At that point, consider whether to hand-label a training supplement (50-100 confirmed FVG examples) to augment the programmatic labels — but this is a last resort, not the first step.

---

## What /nb:plan needs to know

- No external FVG dataset exists to replace or augment the custom detector. This path is closed.
- The fix is diagnostic: read the 27 missed cases from `data/gold_labels.csv`, classify by failure mode, then modify `src/data/labels/fvg.py` accordingly.
- The gold set (75 candles) is the iteration loop target. Every rule change must be evaluated against it before committing.
- Do not hand-label at scale yet — try the relaxed rule first.
- The Suresh et al. (ICIVC 2025) paper uses chart images, not OHLCV time series. It is not a useful architectural reference for our pipeline but confirms that FVG DL research exists and that the concept is ML-tractable.

---

## Open questions

- What are the specific 27 failure modes in `data/gold_labels.csv`? This is the key diagnostic and must be done manually before any code changes.
- Does removing the `close > open` condition on the middle bar raise recall without collapsing precision? Unknown until tested against gold set.
- Would a "body gap only" variant (gap between bodies, not wicks) produce higher agreement with the human annotator? Some practitioners use body-to-body gaps. Unknown.

---

## What I couldn't verify

- Full content of the Suresh et al. (ICIVC 2025) paper — paywalled, page redirects to Springer login. Cannot confirm: exact dataset, labeling method, whether dataset is public, or whether FVG labels are per-candle or image-level.
- Full metadata of the barathanaslan Kaggle dataset — the Kaggle page did not render full content in web fetch. Assessed as non-viable based on title, description snippet, and pattern: "synthetically generated financial charts and annotations" strongly implies image-level labels for classical TA patterns, not FVG structure annotations.
- Whether any non-indexed Kaggle dataset (uploaded after ~late 2025) contains FVG labels. Kaggle's dataset index is large (~80k finance datasets). A targeted manual browse of the 50–100 most recent "stock market labeled" datasets could surface something, but is unlikely given that FVG is niche and the SMC/ICT community does not have a culture of sharing labeled training data.
- LuxAlgo's internal FVG detection logic — closed-source Pine Script. Cannot determine if it has lookahead or what its exact detection rule is.

---

## Falsification attempts

**"The xauusd-trading-ai-smc-v2 HuggingFace model must expose its training data somewhere — find the CSV."**

Checked the model card. The model is an XGBoost checkpoint with 23 features. The training CSV (`smc_features_dataset.csv`) is listed in the model card as a file but contains derived FVG features (FVG_Size, FVG_Type_Encoded), not raw FVG direction labels. The underlying labeling was done via the `smartmoneyconcepts` library — confirmed 1-bar lookahead from R2. Even if extracted, these labels have the same contamination problem as our original detector. Not viable.

**"A candlestick pattern dataset with 500+ patterns must include FVG as one of them."**

FVG is an ICT/SMC concept, not a traditional Japanese candlestick pattern. The traditional candlestick taxonomy (covered by Kaggle's candle-stick-patterns dataset and similar) includes patterns like Doji, Harami, Engulfing, Three White Soldiers — not market structure imbalance patterns. FVG is absent from classical TA literature predating 2015 and would not appear in any traditional candlestick pattern dataset.

**"Transfer learning from a large unlabeled OHLCV dataset (e.g., crypto 1m data) could provide self-supervised pretraining."**

Self-supervised pretraining on unlabeled OHLCV is a different research direction (e.g., contrastive learning on time series). It does not solve the labeling problem — it might improve representation quality but the model still needs FVG-labeled fine-tuning data. This is an interesting extension but not a solution to the kappa=0.284 problem. The immediate fix is the rule, not the model architecture.

---

## Sources

- Kaggle "Human Labeled OHLCV Stock Market Data": https://www.kaggle.com/datasets/barathanaslan/human-labeled-synthetic-stock-market-data
- HuggingFace JonusNattapong/xauusd-trading-ai-smc-v2: https://huggingface.co/JonusNattapong/xauusd-trading-ai-smc-v2
- HuggingFace darkknight25/trading_dataset_v2: https://huggingface.co/datasets/darkknight25/trading_dataset_v2
- Suresh et al. "A Deep Learning Approach to Identify Fair Value Gaps (FVGs) in Forex Markets" (ICIVC 2025): https://link.springer.com/chapter/10.1007/978-3-032-10670-4_36
- ResearchGate entry for Suresh et al.: https://www.researchgate.net/publication/399500256_A_Deep_Learning_Approach_to_Identify_Fair_Value_Gaps_FVGs_in_Forex_Markets
- GitHub starckyang/smc_quant: https://github.com/starckyang/smc_quant
- GitHub joshyattridge/smart-money-concepts: https://github.com/joshyattridge/smart-money-concepts
- TradingView chart export guide: https://www.tradingview.com/support/solutions/43000537255-how-to-export-chart-data/
- LuxAlgo SMC indicator page: https://www.luxalgo.com/library/indicator/smart-money-concepts-smc/
- arXiv "Visual Chart Representations for Cryptocurrency Regime Prediction" (2605.00875): https://arxiv.org/abs/2605.00875
- MDPI "Candlestick Pattern Recognition in Cryptocurrency" (2024): https://www.mdpi.com/2079-3197/12/7/132
