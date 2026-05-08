# Timeframes — SMC Project

> Plain-English explanations. Built up over time.

---

## Q: How many timeframes does our SMC project use and what combination, and why?

**Quick answer:** One timeframe — H1 (1-hour candles). Single-frame for MVP. Multi-timeframe (add Daily) is a backup option only if H1 model hits ceiling.

**Like this:** Think of looking at a city map. Daily candles = country map (too zoomed out, can't see streets). 5-minute candles = single street (too zoomed in, lose all sense of direction). H1 = neighbourhood map. Right level to see "where stuff happens" without drowning in detail.

**How it works:**

SMC traders use multiple timeframes top-down: Daily for big-picture bias, H1 for spotting tradeable zones, 5–15min for entering trades. H1 is the *zone-identification layer*. That's exactly what our model needs to learn.

Sample size matters too. SPY 2018–2024 gives ~13.5k H1 candles. LSTM/xLSTM need ~10k+ to learn structure. Daily candles only give ~1.5k — too few. 30-min gives ~24k but adds noise. H1 = sweet spot.

Multi-timeframe fusion (e.g. add Daily as macro context) is on the back-burner. Only triggers if all three are true: (1) H1 baseline F1 > 0.45, (2) val F1 plateaus despite tuning, (3) errors correlate with macro regime. Until then, single-frame keeps the MVP simple.

If we ever add Daily, we use the cheapest path: broadcast Daily features (ATR, trend) onto each H1 row. Not Temporal Fusion Transformer. TFT = 3-week build, overkill at this scale.

**Real example:** SPY 2024-03-15 09:30 ET to 16:00 ET = 7 H1 candles. Each candle becomes one row in our (60, 5) sliding window. We never look at the daily bar that contains those 7 hours — only at the previous 60 H1 bars leading up to the current one. One timeframe, one stream, one model.

**Reference:** `.nb-suite/research/8-May-26/timeframe-decision.md`, `.nb-suite/research/8-May-26/timeframe-deep.md` (Deep multi-perspective falsification — confirmed single-frame for *identification*).

---

## Q: But don't pros use multiple timeframes? HTF to verify FVG, LTF as main?

**Quick answer:** Pros do use multiple timeframes — but for *trade qualification*, not *FVG identification*. Identification is single-frame in every canonical SMC source. Our model does identification, so single-frame is correct.

**Like this:** Imagine spotting a pothole in the road. Identifying *that it's a pothole* = look at the pothole (single-frame). Deciding *whether to swerve* = check the road context, traffic, your speed (multi-frame). Two different jobs. Our model is the pothole spotter.

**How it works:**

The 3-candle FVG pattern is a self-contained geometric definition: candle N-1 high < candle N+1 low (bullish), or vice versa. Every Python library, every ICT tutorial, the one academic paper — all define detection on a single timeframe. The candles are next to each other on one chart. No HTF needed to "see" the pattern.

What pros call "HTF verification" is qualification: an H1 FVG is *more tradable* if Daily bias agrees, *less tradable* otherwise. The FVG still exists either way. Filtering tradable ones is a downstream decision — separate problem from finding them.

"LTF as main" refers to the trader's role. Zone identifier role uses H1. Entry trigger role uses 5–15min. Different jobs in the pipeline. Our model is the zone identifier.

**Real example:** SPY 2024-03-15 11:00 H1 candle has bullish FVG. Identification = single-frame fact, model labels it. Qualification = check Daily (was bullish that day) → tradable. Two separate calls. We automate the first one only. Multi-TF qualification = future feature, not MVP.

**Reference:** `.nb-suite/research/8-May-26/timeframe-deep.md` (5 falsification attempts, 5 survivals; canonical ICT + retail 2024–26 + one academic source all converge).

---

## Q: So what timeframes did we land on and does the plan cover all SMC concepts, not just FVG?

**Quick answer:** One timeframe — H1. One SMC concept — FVG only for MVP. Other SMC concepts (Order Block, BOS, CHoCH, Liquidity, S&D zones) explicitly out-of-scope, listed as future extensions.

**Like this:** Building a house. We're laying the foundation slab first (FVG detector). Walls, roof, plumbing (OB, BOS, CHoCH detectors) are in the blueprint but built later. Don't pour the slab and pretend the house is done — but also don't try to build the whole house in 43 days with one person.

**How it works:**

Timeframe = H1, single-frame. Locked. Backup option (broadcast Daily features) only triggers if H1 baseline plateaus.

SMC concepts covered = FVG only. The master plan and `idea.md` both list this as deliberate scope reduction. Reasoning: FVG has the cleanest geometric definition (3-candle pattern), most consistent labels, fastest to validate. Other concepts (OB, BOS) are messier — Order Block has 4+ competing definitions in retail SMC content alone. Building a noisy-label OB detector in parallel = wastes time, dilutes results.

The architecture is built to extend. Same data pipeline, same window shape (60×5), same model classes — just swap the label generator. Once FVG model works end-to-end, adding OB is ~3 days: write OB rule, regenerate labels, retrain. Final report can frame this as "FVG detector validated; OB extension demonstrated" if time permits — likely after Status Update 2 (7 June).

If demo is FVG-only on 17 June, framing is "automated FVG zone detection — the core pattern-recognition primitive of SMC". Honest, defensible, complete.

**Real example:** Status Update 1 (17 May) = FVG pipeline + LSTM baseline. Status Update 2 (7 June) = FVG with 4 architectures compared. Final (17/20 June) = FVG demo + write-up; OB + BOS extensions if Phase 4/5 finished early. Master plan's `Out of scope` section binds this — no scope creep mid-project.

**Reference:** `.nb-suite/plan/8-May-26/master-phasing.md` (Out of scope section), `docs/idea.md` (MVP scope).


