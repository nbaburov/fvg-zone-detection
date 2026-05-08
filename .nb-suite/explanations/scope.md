# Scope — What the Project Delivers

> Plain-English explanations. Built up over time.

---

## Q: I want the model to detect FVGs but also tell me if it means buy or sell. And later add other SMC concepts — is that too much?

**Quick answer:** Buy/sell *bias label* per FVG = free, already in the data. Actual buy/sell *trade signal* = scope creep. Extending to other SMC concepts (OB, BOS, etc.) = realistic for **maybe one** extension if Phase 4/5 finishes early — not all of them.

**Like this:** Restaurant menu. Saying "this dish is spicy" = free, that's a property of the dish (label). Saying "you should order this dish tonight" = different job (recommendation engine). And adding 5 new dishes to the menu = each one needs its own recipe, testing, photo. Not free.

**How it works:**

**Buy/sell bias is free.** The label already encodes direction. Bullish FVG (price gapped up) = "buy bias zone" by SMC theory. Bearish FVG = "sell bias zone". Our ternary label is `bullish / bearish / none` — chart just renders it with a colour and a tooltip: "Bullish FVG — potential buy zone". Zero extra model work. Add this to the Plotly viz layer in Phase 6.

**Buy/sell *signal* is NOT free.** A real signal answers "should you buy at $549.20 right now". That requires multi-timeframe qualification (Daily bias agrees?), mitigation tracking (has price filled the gap yet?), and entry trigger (LTF confirmation). That is *trade qualification* — explicitly out-of-scope per R4-deep research. Promising trade signals = different project, 2× the work, and academic ethics issue (retail harm, see `idea.md` ethics section).

**Extending to OB / BOS / CHoCH / Liquidity / S&D.** Each one = own labelling rule, own noise floor, own hand-label gold set, own retraining run. ~3 days minimum each. 5 concepts × 3 days = 15 days. Phases 4–6 already eat 33 of remaining 43 days. Math says: maximum **one** extension is realistic, only if FVG ships clean before 4 June. Order Block is the natural choice (most-cited SMC structure after FVG).

**Real example:** Final 17 June demo shows: Plotly chart with green boxes labelled "Bullish FVG (buy bias)" and red boxes "Bearish FVG (sell bias)". Hover tooltip = confidence score. Title = "FVG Zone Detection — Pattern Recognition Primitive of SMC". If Phase 4 finishes by 28 May (best case), Order Block extension added with same UI treatment. If not, FVG-only ships and final report lists OB/BOS/CHoCH as concrete future work with the existing pipeline.

**Reference:** `.nb-suite/plan/8-May-26/master-phasing.md` (Out of scope section + Phase 6 deployment), `docs/idea.md` (MVP scope), `.nb-suite/research/8-May-26/timeframe-deep.md` (qualification vs identification distinction).
