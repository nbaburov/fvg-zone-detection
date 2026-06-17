# Final Presentation — Content (full script)

**Target:** 5 minutes spoken (~750-850 words of "Say" text). This file is the SOURCE of truth for the story. The condensed presenter notes (one-liners I hold while speaking) come from this. The deck (web or pptx) is built from this.

**Audience guardrail (apply to every line):** *Can a non-technical AND non-financial person follow this? Is the story complete? Does it answer their question before they ask it?* Every special word gets a plain-language definition + an analogy on first use. No term appears from nowhere.

**The one-sentence throughline:** *Pro traders mark certain zones on a chart by hand; I taught five AIs to find them automatically, and the real lesson wasn't "build a smarter AI" — it was "feed it more data."*

**Arc (day 0 → now):** big idea → narrow to one honest MVP → the data and why it's hard → the five models → the surprise result → how I know it's true → what's good, bad, and ugly → the working product → what I learned and what's next.

---

## Beat 1 — Hook + title  ·  ~15s  ·  Slide: title + the FVG figure (`docs/report/fig_fvg.png`)

**Say:**
"Professional traders draw certain zones on price charts by hand and treat them as magnets that price comes back to. I spent this project teaching a computer to find those zones automatically — and the most interesting thing I found wasn't about AI at all. Let me walk you through it."

**Answers (lay question):** *What is this even about?* → A computer that spots a hand-drawn trading pattern.
**Show:** Title slide. One clean chart with a highlighted gap zone.

---

## Beat 2 — Think Big (the vision)  ·  ~30s  ·  Slide: "The big idea"

**Say:**
"First, the big picture — the 'Think Big' part of this assignment. The vision: an assistant that reads a price chart the way an experienced institutional trader does — automatically marking all the structures they look for, with a confidence score, in real time. Today traders do this by hand, it's slow, and everyone draws it slightly differently. An AI that does it consistently could be a genuine co-pilot for chart analysis."

**Answers:** *Why should I care / what's the ambition?* → Automate expert chart-reading; consistency at scale.
**Show:** A messy hand-drawn chart vs a clean AI-marked one.

---

## Beat 3 — The honest MVP + what an FVG is  ·  ~45s  ·  Slide: 3-candle FVG diagram

**Say:**
"For my realistic first version — the MVP — I narrowed that vision to ONE pattern on ONE stock. The pattern is called a Fair Value Gap. In plain terms: when the market moves up or down very fast over three price bars, it can leave a little price gap that barely traded — like a skipped step on a staircase. Traders believe price tends to come back and 'fill' that step. The stock is SPY, which simply tracks the 500 biggest US companies. To be clear what this is NOT: I'm not predicting whether the price goes up or down. I only find *where* this pattern exists and how sure the model is. And here's why it's worth doing properly: this pattern is everywhere in retail trading — millions of posts and videos — yet almost nobody has studied it with real machine-learning discipline. That gap is the opportunity."

**Answers:** *What's a Fair Value Gap? Is this stock-market fortune-telling? Why does it matter?* → A fast-move price gap (skipped stair-step); no, it detects a pattern, not direction; and it's a hugely popular but barely-researched concept.
**Show:** Three candles with the gap highlighted; a "NOT price prediction" callout.

---

## Beat 4 — The data, and why it's genuinely hard  ·  ~45s  ·  Slide: rarity + "no peeking"

**Say:**
"I used nine years of hourly SPY data, 2016 to 2025. Two things make this hard. First, the pattern is rare — only about 3 in 100 bars are a valid gap — so a lazy model that always says 'no gap' would be 97% 'accurate' and completely useless. Second, you can never let the model peek at the future: I trained on the older years and tested on the newest, never the other way around. I also caught a subtle bug in a popular open-source library that was quietly peeking one bar ahead — exactly the kind of leak that makes results look great and mean nothing. Catching that mattered more than any single model."

**Answers:** *How much data, and why isn't this easy?* → Rare pattern (accuracy lies) + strict no-future-peeking, and I caught a real leak.
**Show:** A 97/3 split bar; a timeline split (train past → test future); a "leak caught" badge.

---

## Beat 5 — The five models (the ladder)  ·  ~30s  ·  Slide: 5 model names, simple→complex

**Say:**
"I didn't bet on one model. I built five, from simpler to more advanced: a decision-tree model called XGBoost, then three kinds of neural network — an LSTM, a CNN-LSTM, and a Transformer, the same family that powers modern chatbots — plus a newer one called xLSTM. Each one looks at a sliding window of the last 60 hours of price and answers a single question: gap here, or no gap? Same data, same rules, head-to-head — and I deliberately included the simple model as the baseline to beat."

**Answers:** *What did you build, and how does it "look at" a chart?* → Five models, simple to advanced; each reads a 60-bar window and classifies gap/no-gap; compared fairly.
**Show:** Five labeled tiles on a "simple → complex" arrow.

---

## Beat 6 — The surprise (the headline finding)  ·  ~60s  ·  Slide: timeframe-inversion figure (`docs/report/fig_timeframe.png`)

**Say:**
"Here's the surprise. On the hourly data, the SIMPLEST model won — XGBoost beat all the fancy neural networks. Why? Because even though I had thousands of bars, they overlap so heavily that there were only about a hundred truly independent examples — and neural networks are data-hungry. So I tested that head-on: I added more data two ways — pooled four stocks instead of one, and switched to five-minute bars, about twelve times more examples, for free. Both lifted the neural networks, and on the five-minute bars the ranking flipped: the best neural network scored higher than XGBoost, 0.71 to 0.65, on the same data. I'll be straight with you — XGBoost's hourly score is still the single best overall, so the gap narrowed rather than vanished, and part of that flip is that XGBoost's hand-built features were designed for hourly bars and travel poorly to five-minute. But the direction is unmistakable: feed the neural network more data and it catches, then beats, the simple one. The bottleneck was never model power. It was data."

**Answers:** *Which model won, and so what?* → Simple beat fancy on hourly because data was the bottleneck; add data and at 5-min the neural net pulls ahead. Lesson: "more data, not a bigger model." (Stays honest: XGBoost's hourly score is still the single best overall.)
**Show:** Inversion compared WITHIN the 5-minute timeframe: XGBoost 0.654 vs CNN-LSTM 0.713 (neural net ahead). Honest footnote on the slide: "XGBoost's hourly 0.738 is still the single best score overall — compare within a timeframe, not across."

---

## Beat 7 — How I know it's real (validation = trust)  ·  ~30s  ·  Slide: "Trust the numbers"

**Say:**
"A number is only worth as much as the way you got it. So I trained every model five separate times with different random starts and reported the spread, not a lucky run. I put confidence ranges on the scores. And the whole codebase has over a thousand automated tests, including ones that specifically check nothing leaks from the future. The point is: these results are reproducible, not cherry-picked."

**Answers:** *Why should I believe your numbers?* → Multiple runs, confidence ranges, 1000+ tests, leak checks.
**Show:** Icons: 5 seeds · confidence bars · 1000+ tests · no-leak check.

---

## Beat 8 — Good, bad, and ugly (honesty)  ·  ~30s  ·  Slide: traffic-light honesty

**Say:**
"Being honest about what works and what doesn't. The good: it detects the pattern reliably, and the science behind it is solid. The bad and the ugly: as an actual money-making trading signal, the edge is thin. When I simulated trading on it with realistic costs, almost everything lost money — only one specific setup, on one timeframe, on SPY, came out positive. So this is a strong *detector* and an honest *study* — not a get-rich strategy, and I'm not going to pretend it is."

**Answers:** *Does it actually make money / what are the limits?* → Great detector, weak trading edge, stated plainly.
**Show:** Three columns — Good / Bad / Ugly — with one line each.

---

## Beat 9 — The product + live demo  ·  ~45s  ·  Slide: [DEMO VIDEO]

**Say:**
"This is more than a notebook. I built an inspection tool that draws the model's detections on the chart so you can audit them, and a paper-trading harness that runs the models on live market data. Here's the piece that ties it together — watch the bars arrive one at a time, and watch each of the four models draw its gaps live, then mark whether price respected the gap or not."

**Answers:** *What did you actually deliver — is it usable?* → A real, inspectable product, shown running.
**Show:** The recorded demo (scene picker: successes / failures / mixed). Let it play ~15-20s.

---

## Beat 10 — What I learned + what's next + close  ·  ~30s  ·  Slide: takeaway

**Say:**
"Over the semester I went from a blank repository to five validated models, an honest verdict, and a working tool. What I take away: for spotting a rare pattern, more and better data beats a fancier model nearly every time — and rigor, like catching that leak, is what separates a real result from a nice-looking one. Next steps are clear: more data still, the other Smart Money patterns beyond this one gap, and a live forward-test. Thank you — happy to take questions."

**Answers:** *What's the lesson, and where does it go?* → Data > model; rigor matters; clear next steps.
**Show:** One-line takeaway + 3 next-step bullets.

---

## Fact-check appendix (numbers I must not get wrong)

- FVG positive rate ≈ 3% of hourly bars (imbalanced; accuracy is meaningless → use F1 on the rare class).
- Data: SPY hourly, 2016-2025. Split: train 2016-2021, validate 2022, test 2023-2025 (time-ordered, no shuffle).
- Effective independent samples ≈ 117 (60-bar windows overlap, stride 1).
- Headline scores (macro-F1): XGBoost 0.738 (hourly, best control) · CNN-LSTM 0.640 → 0.675 (4-symbol pooled, hourly) → 0.713 (tuned, 5-minute — best deep-learning result) · LSTM 0.640 · Transformer 0.577 (unstable) · xLSTM 0.369 (weakest).
- **Timeframe-inversion — compare WITHIN a timeframe (different test sizes across TFs, so cross-TF is not apples-to-apples):** at 5-min, CNN-LSTM 0.713 (tuned) / 0.693 (untuned) **>** XGBoost 0.654 (neural net ahead). At 15-min, XGBoost 0.696 ≈ CNN-LSTM 0.691 (XGBoost marginally ahead). At hourly, XGBoost 0.738 > CNN-LSTM 0.675. So **0.738 (XGBoost, hourly) is still the single best score overall**; the DL-vs-GBM gap narrowed, it did not fully close on an absolute basis. Caveat: part of XGBoost's 5-min drop is a representation handicap — its hand-crafted features were built for hourly bars.
- Levers that helped: pool 4 tickers (SPY+QQQ+IWM+DIA); resample to 5-minute (~12x more labelled bars).
- Trading reality: with realistic costs, only 15-minute SPY with a fixed 2R exit was positive; 5-minute is detection-only; edge does not generalise to QQQ/IWM/DIA.
- Validation: 5 seeds, bootstrap confidence intervals, 1000+ automated tests, mandatory no-lookahead test; caught a 1-bar lookahead leak in the public `smartmoneyconcepts` library.
- Product: inspection/overlay tool, the bar-by-bar demo (being recorded), live paper-trading harness.
- Think Big = auto-detect all SMC structures as a real-time chart co-pilot. MVP = one pattern (FVG), one symbol (SPY).
