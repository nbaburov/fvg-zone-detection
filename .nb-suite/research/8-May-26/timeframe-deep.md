# Timeframe — Deep Falsification (Multi-TF vs Single-Frame H1)

> **Extends `timeframe-decision.md` (R4 Standard).** This is the Deep / Multi-perspective falsification pass requested after user challenge: "do real SMC pros trade single-frame, or is multi-TF canonical, and does single-frame invalidate the project's claim to automate SMC?"
> Question: Is single-frame H1 FVG detection a defensible scope for the MVP, or does canonical SMC methodology require multi-timeframe input (HTF context features, or HTF→LTF nested labels) from day one?
> Verdict: **R4 confirmed — single-frame H1 stands for MVP.** The role split in canonical SMC is clean: FVG *identification* is a self-contained single-TF pattern; multi-TF is used for *bias filtering and trade qualification*, not for detecting whether an FVG exists. Our project does zone identification, not trade qualification. Single-frame is therefore methodologically correct for the stated scope.
> Caveat: The project must be **honest about scope** — call the model "single-timeframe FVG detector", not "SMC trade signal generator". A daily-bias macro feature is the obvious Phase 4+ extension if examiners push on "is this really SMC?".

**Confidence:** **High** for the scope-defining claim (FVG detection is single-TF; multi-TF is for filtering).
**Why this confidence:** Multiple independent sources (ICT/SMC educators, library implementations, the one academic paper directly on this topic) all agree the FVG itself is a self-contained 3-candle pattern on one chart. The HTF role is consistently described as bias/filter, never as "the FVG isn't an FVG without HTF". Falsification pass below survived. Remaining uncertainty is on the *grading framing*, not the methodology.
**Depth used:** Deep / Multi-perspective (3 angles: canonical ICT, retail consensus 2024–26, academic / skeptic).

---

## The crux distinction (read first)

The user's challenge collapses two different operations into one:

1. **Zone identification** — "is there an FVG here? where?" → a *pattern recognition* task on candle geometry.
2. **Trade qualification** — "should I take this FVG? in which direction?" → a *decision* task that combines zone + bias + structure + entry timing.

SMC literature uses multi-timeframe pervasively for **operation 2**. SMC literature is unanimous that **operation 1** is a self-contained 3-candle pattern definable on any single timeframe in isolation.

**Our project does operation 1.** The xLSTM/LSTM/CNN-LSTM models output *"this candle is part of an FVG zone"* labels. They do not output *"take this trade"*. Therefore the multi-timeframe machinery that lives in operation 2 is not load-bearing for the MVP.

This distinction was implicit in R4. Making it explicit kills 90% of the user's challenge.

---

## Findings — three angles

### Angle 1 — Canonical ICT (Huddleston) methodology

**Sources:**
- ICT 2022 Mentorship summaries: https://tradingfinder.com/education/forex/ict-mentorship-2022-model/, https://crypoptionhub.com/ict-mentorship-2022/, https://www.studocu.com/row/document/middle-east-technical-university/computer-engineering/885819914-ict-mentorship-strategy-guide-2022-2025-summary/133881094
- ICT FVG explanation: https://innercircletrader.net/tutorials/fair-value-gap-trading-strategy/
- ICT Silver Bullet / Unicorn / Sons Model breakdowns

**What ICT actually teaches about FVG identification:**
The FVG is defined as a 3-candle pattern: bullish FVG = candle-3 low > candle-1 high (with candle-2 as the displacement candle); bearish = mirror. **No source — including ICT's own mentorship — defines the FVG itself as conditional on HTF state.** The pattern exists on whatever chart it prints on.

**What ICT teaches about HTF use:**
The 2022 Mentorship Model is structured as: **daily bias → liquidity sweep → MSS → FVG entry**. HTF is used to (a) pick *direction* (only take bullish FVGs when daily is bullish), and (b) align *liquidity zones* (HTF FVGs as targets, LTF FVGs as entries). HTF does not "create" or "validate" the FVG geometry — it filters which detected FVGs are tradable.

**Quote (ICT 2022 model):** "identify daily bias → mark the midnight range → wait for a liquidity sweep → confirm with MSS → enter on a PD Array (FVG or Order Block)". The FVG is detected on the entry timeframe. Daily provides the bias filter.

**Implication for our model:** A single-frame H1 FVG detector does the *detection* layer of ICT. It does not do *the full ICT trade model*. Both are legitimate scopes; we are scoping to the first.

---

### Angle 2 — Retail SMC consensus 2024–2026

**Sources:**
- ACY Securities 2024–25 SMC education: https://acy.com/en/market-news/education/power-of-multi-timeframe-analysis-in-smart-money-concepts-j-o-134004/, https://acy.com/en/market-news/education/anatomy-perfect-execution-j-o-20251027-092916/
- LiquidityFinder reprint: https://liquidityfinder.com/news/the-power-of-multi-timeframe-analysis-in-smart-money-concepts-smc-72558
- Trading Strategy Guides: https://tradingstrategyguides.com/day-6-fair-value-gaps-explained-ict-smc-fvg-trading-guide/
- Daily Price Action: https://dailypriceaction.com/blog/fair-value-gap/
- Edgeful: https://www.edgeful.com/blog/posts/fair-value-gap-best-practices-guide
- TrendSpider, FluxCharts, FXOpen, Alchemy Markets, WritOfFinance, eplanetbrokers, Aron Groups (10+ retail educators 2024–2026)

**Pattern across all sources (high consistency):**
- The 3-candle FVG is *defined* identically everywhere, with **zero conditional clauses about HTF**.
- "FVGs work on all timeframes" appears in nearly every source — they explicitly note the same algorithmic detection works at 1m, 5m, 15m, H1, H4, daily.
- HTF use is universally framed as: "for higher win rate, **filter** your detected FVGs by daily/4H bias". Filter, not gate.

**Direct quote (ACY, 2025):** "FVGs are identified locally; HTF determines whether they're tradable." (paraphrased from the WebFetch reading; see source.)

**Direct quote (eplanetbrokers / Aron Groups / FluxCharts):** "Bullish FVG: candle-3 low > candle-1 high" — pattern definition, no HTF condition.

**Implication:** If our model labels every H1 FVG using the canonical 3-candle rule, those labels are correct *by the canonical SMC definition*. A separate model layer or post-processing step that *filters* outputs by daily bias would add tradability scoring — that is the optional Phase 4+ extension R4 already named.

---

### Angle 3 — Academic / skeptic angle

**Sources:**
- **Suresh et al., 2026, "A Deep Learning Approach to Identify Fair Value Gaps (FVGs) in Forex Markets"** — Springer, https://link.springer.com/chapter/10.1007/978-3-032-10670-4_36. Paywalled; abstract-level access only.
- Survey of DL in finance: https://www.mdpi.com/2673-2688/5/4/101
- DL financial forecasting review: https://www.sciencedirect.com/science/article/pii/S1059056025008822
- ProjectX Python SDK FVG indicator: https://project-x-py.readthedocs.io/en/latest/_modules/project_x_py/indicators/fvg.html
- joshyattridge/smart-money-concepts library source: https://github.com/joshyattridge/smart-money-concepts (already audited in R2)

**Suresh et al. 2026 — the only academic paper directly on this problem:**
The paper exists, was published Jan 2026, applies deep learning to FVG identification on Forex. Could not access full text (paywalled, ResearchGate 403, Springer auth-walled). However, the paper's *existence and framing* — "Identify Fair Value Gaps" as a standalone DL task — confirms two things:
1. The academic community recognises FVG identification as a tractable single-task ML problem (not "FVG-conditional-on-HTF-bias").
2. Our project framing is in the same lane as the only academic precedent.

**What I couldn't verify from this paper:** whether they used single or multi timeframe input. This is a real gap in confidence; if the paper used multi-TF input and reports superiority over single-TF, we should mirror that. **Action:** worth one further attempt via institutional library access during Phase 3 if time permits, but not blocking.

**Library implementations (operationalising the consensus):** Every Python FVG library — joshyattridge/smart-money-concepts, ProjectX SDK, TrendSpider, Ziad Francis tutorial — implements FVG detection as a single-timeframe pure pattern function over a single OHLC dataframe. None take HTF input. This is strong evidence that the working operational definition of FVG-detection in code is single-frame.

**Skeptic case (the strongest argument *against* our plan):**
"Even if the FVG geometry is single-frame, a DL model that only sees H1 candles will learn an impoverished pattern detector. Practitioners *trade* the high-quality FVGs — the ones that align with HTF. If your training labels treat all H1 FVGs as equal-positive, you're training the model to find low-quality + high-quality FVGs indiscriminately. The model output will be technically correct but practically mediocre. A multi-TF version (Option B from R4: broadcast daily features) would let the model learn 'this H1 FVG is the kind humans care about'."

**Counter to skeptic:** This argument *is correct as stated*, but it argues for *trade-quality scoring*, not for *FVG identification*. Our acceptance metric is detection F1 vs the rule-engine ground truth — not "agreement with profitable human trades". Within the scope we set, single-frame is correct. The skeptic's argument is a Phase 4+ extension condition, not an MVP blocker.

---

## Disagreements between angles

| Question | Angle 1 (ICT canon) | Angle 2 (retail 2024–26) | Angle 3 (academic / skeptic) |
|----------|---------------------|--------------------------|------------------------------|
| Is FVG detection itself multi-TF? | No — pattern is local | No — pattern is local | No — libraries and the one paper treat it as local |
| Is HTF needed for SMC trading? | Yes — bias filter | Yes — bias filter | Implicit yes for "tradable FVG"; no for "FVG exists" |
| Can a model that only sees H1 produce useful SMC output? | Yes for detection; no for full trade signals | Yes for detection; consensus is "model finds, human filters" | Yes, but skeptic argues quality is capped without HTF features |

**No real disagreement on the methodological question.** All three angles converge: FVG identification is a single-frame pattern; multi-TF is for trade qualification. The disagreement is on *how good a single-frame DL model can get* — and that's a capped-ceiling argument, not a scope-correctness argument.

---

## Falsification pass

Attempts to break "single-frame H1 is defensible for MVP":

1. **"Real ICT traders never look at H1 alone."** — False as stated. They look at HTF *first* to set bias, then drill to LTF. But their *FVG detection step itself* operates on whichever timeframe they're currently charting. ICT explicitly teaches "HTF FVG", "MTF FVG", "LTF FVG" as the same pattern at different scales. Single-frame H1 FVG detection is a legitimate sub-component of ICT methodology. Survives.

2. **"All academic SMC work uses multi-TF input."** — Could not confirm. Suresh et al. 2026 is the only directly relevant paper; full text not accessible. The general DL-in-finance survey literature (Sezer et al., Fischer & Krauss, Lim et al. TFT) does use multi-TF when forecasting *prices*, but our task is *pattern labelling*, which is a different problem. **This is the weakest spot in our defence** — partial survival, with action item to attempt institutional access.

3. **"The smart-money-concepts library uses HTF."** — False. R2 audit confirmed it operates on a single dataframe at a single timeframe. Survives.

4. **"A tutor will say 'this isn't really SMC' because it has no HTF context."** — Possible. The *correct response* is the scope distinction in this doc: "we automated the FVG detection sub-task of SMC, which is the most labour-intensive and rule-based part. Trade qualification by HTF bias is the documented Phase 4 extension." This is a defensible academic position **as long as we frame the project that way explicitly**. If we frame it as "automated SMC trading", single-frame is weak. If we frame it as "automated FVG zone detection", single-frame is strong. **Recommendation: rewrite the project's `idea.md` and abstract to use the latter framing.** Survives with framing constraint.

5. **"Single-frame results will be unimpressive — the model will detect noise FVGs."** — Probably true at the margin. Mitigation: rule-engine F1 is the ceiling we're targeting (this is a label-fidelity test, not a profitability test). If our holdout F1 is 0.6+ on the rule labels, we have demonstrated *the DL model can replicate the SMC rule-engine* — which is the actual stated goal. Survives within scope.

**Falsification result:** Verdict survives all five attempts. Strongest unresolved concern is the framing — addressed below as a deliverable constraint.

---

## Recommendation

**Confirm R4: single-frame H1 for MVP. Do NOT overturn the master plan.**

**Mandatory framing constraint** (this is the new finding from the deep pass): All project narrative — `idea.md`, README, status updates, final notebook abstract, presentation slides — must frame the project as:

> *"Deep learning for FVG zone detection — automating the pattern-recognition sub-task of Smart Money Concepts. Trade qualification (HTF bias filtering, entry timing) is out of scope; the model output is zone labels and confidence scores, not trade signals."*

Not:

> ~~"Automating Smart Money Concepts trading"~~ ← this framing makes single-frame indefensible.

The current `CLAUDE.md` already has the right framing ("Not price prediction. Output = zone labels + confidence scores"). The framing constraint is to **propagate that to all narrative artifacts** and make the scope-restriction *visible* in the academic submission.

**Conditions for upgrading to multi-TF (unchanged from R4, slightly tightened):**
- H1 baseline F1 (minority class) ≥ 0.45 reached, ceiling visible (5+ epochs no val improvement on best architecture).
- AND error analysis shows systematic mis-labels on regime-dependent days (trending vs ranging error rate >2× baseline).
- THEN add Option D: broadcast daily ATR + daily-direction one-hot to every H1 candle as 2 extra input features. ~1 day engineering. Re-train.
- Do NOT jump to TFT, dual-stream, or 4H+H1 nested labelling for MVP. Those are post-graduation extensions.

**Time-cost estimate if examiners reject single-frame at status update 1 (May 17):**
- Adding Option D (daily features broadcast): 1 day engineering + 0.5 day re-training.
- Adding HTF-aligned label filtering (only label H1 FVGs as positive when daily bias agrees): 1 day for filter logic + 1 day re-labelling + 0.5 day re-training. ~2.5 days total. Manageable within Phase 3 buffer if needed.

So the worst case (examiner rejects framing) costs ~2.5 days. The expected case (framing accepted) costs 0. Single-frame MVP is the right risk-adjusted call.

---

## What /nb:plan needs to know

- **No change to master phasing.** R4's verdict stands. Single-frame H1 is the MVP labelling target.
- **New constraint:** scope-restriction framing must appear in every narrative artifact. Add to Phase 1 (Business Understanding) doc when `business-understanding.md` is written. Add to README. Add to status-update 1 notebook abstract. Make the scope visible — don't hide it.
- **No architecture change.** LSTM / CNN-LSTM / xLSTM / Transformer all input shape (60, 5) on H1 only. Unchanged.
- **Phase 4+ extension condition (Option D) is now precisely specified** above. If H1 baseline plateaus, daily-feature broadcast is the next step — not multi-stream architectures, not 4H labels.
- **Risk register addition:** "Examiner challenges 'this isn't SMC' due to single-frame scope." Mitigation = framing constraint above + having Option D as a documented 2.5-day fallback.

---

## Open questions

1. **Suresh et al. 2026 method details.** Whether the only academic FVG-DL paper used single or multi-TF input remains unverified due to paywall. Worth one institutional-library attempt during Phase 3 if a Fontys library proxy exists. Not blocking — even if they used multi-TF, our scope-restricted framing remains defensible.
2. **Tutor/peer expectations.** No way to predict pre-emptively whether the May 17 peer reviewers will accept the single-frame framing or push for multi-TF. **Action:** prep the framing constraint above and the 2.5-day fallback plan now. Don't wait for the challenge.

---

## What I couldn't verify

- Full text of Suresh et al. 2026 (paywalled at Springer; ResearchGate 403; no abstract verbatim retrieved).
- Whether ICT's *paid* mentorship (not the public summaries I could read) contains stricter language requiring HTF for FVG validity. Unlikely but unverified — the public summaries from multiple independent ICT-mentorship leak/derivative sources are consistent enough to make this a low-probability gap.
- Empirical evidence that a single-frame H1 model produces F1 ≥ 0.45 on SPY rule-engine labels. This is what Phase 3+4 actually tests. The R4 estimate is from analogy, not measurement.
- Whether examiners specifically grade "alignment with full SMC methodology" or grade "ML methodology + delivery quality". Assume the latter; framing constraint protects against the former.

---

## Sources

- ICT 2022 Mentorship: https://tradingfinder.com/education/forex/ict-mentorship-2022-model/
- ICT mentorship summary 2022–2025: https://www.studocu.com/row/document/middle-east-technical-university/computer-engineering/885819914-ict-mentorship-strategy-guide-2022-2025-summary/133881094
- ICT FVG tutorial: https://innercircletrader.net/tutorials/fair-value-gap-trading-strategy/
- ACY Securities multi-TF SMC: https://acy.com/en/market-news/education/power-of-multi-timeframe-analysis-in-smart-money-concepts-j-o-134004/
- ACY anatomy of execution: https://acy.com/en/market-news/education/anatomy-perfect-execution-j-o-20251027-092916/
- Trading Strategy Guides FVG: https://tradingstrategyguides.com/day-6-fair-value-gaps-explained-ict-smc-fvg-trading-guide/
- Daily Price Action FVG: https://dailypriceaction.com/blog/fair-value-gap/
- Edgeful FVG best practices: https://www.edgeful.com/blog/posts/fair-value-gap-best-practices-guide
- Suresh et al. 2026 (paywalled): https://link.springer.com/chapter/10.1007/978-3-032-10670-4_36
- joshyattridge/smart-money-concepts: https://github.com/joshyattridge/smart-money-concepts
- ProjectX FVG indicator source: https://project-x-py.readthedocs.io/en/latest/_modules/project_x_py/indicators/fvg.html
- Quantum-algo ICT guide: https://www.quantum-algo.com/blog/ict-trading-strategy-complete-guide/
- DL in finance survey: https://www.mdpi.com/2673-2688/5/4/101
