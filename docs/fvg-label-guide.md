# How to Label Valid FVG — Decision Guide

> This guide is for the gold annotation session in `notebooks/02-gold-annotation.ipynb`.
> Follow the **Quick Decision Flow** (5 questions) for clicking.
> Detailed criteria section explains the *why* behind each question.
> Only 4 criteria are active in the labeller; mitigation (#1) and S/R (#3) are excluded — see "Skipped" note in Quick Decision Flow.

---

## Quick Decision Flow (ELI5)

Use this for fast clicking. Full criteria detail below.

**Buttons in notebook:**
- `b — Bull FVG` (green) — bullish valid FVG
- `e — Bear FVG` (red) — bearish valid FVG
- `n — None` (gray) — no valid FVG
- `a — Ambiguous` (orange) — can't tell, saved as None + note

### THE GOLDEN RULE

> **You only click `b` (Bull) or `e` (Bear) if ALL FOUR questions Q1–Q4 pass.**
> **Any single NO at any step → click `n — None`. Stop. Don't check the rest.**

That is the entire logic. Now the questions:

```
─────────────────────────────────────────────────────────────────────
Q1. IS THERE A COLORED BOX ON THE CHART?
    Look for a green-shaded OR red-shaded rectangle.

    ❌ NO box         → click [n — None]. STOP.
    ✅ Green box      → it's bullish. Continue to Q2.
    ✅ Red box        → it's bearish. Continue to Q2.
─────────────────────────────────────────────────────────────────────
Q2. DOES THE N+2 CANDLE CLOSE ON THE CORRECT SIDE OF THE BOX?
    N+2 = the candle marked by the GREEN DOTTED vertical line.

    Bull box: is N+2's close ABOVE the bottom edge of the green box?
    Bear box: is N+2's close BELOW the top edge of the red box?

    ❌ NO  → click [n — None]. STOP.
    ✅ YES → Continue to Q3.
─────────────────────────────────────────────────────────────────────
Q3. IS THE BOX ON THE CORRECT SIDE OF THE BLUE DASHED LINE?
    The blue dashed line = Gann midpoint of the recent swing.

    Bull box: is the green box BELOW the blue line (lower half)?
    Bear box: is the red box  ABOVE the blue line (upper half)?

    ❌ NO  → click [n — None]. STOP.
    ✅ YES → Continue to Q4.
─────────────────────────────────────────────────────────────────────
Q4. DID PRICE BREAK A SWING POINT TO THE LEFT BEFORE THE BOX?
    Look LEFT of the red dashed line for STAR markers (★).

    Bull: any ORANGE ★ (↑BOS) star? = bullish break of structure
    Bear: any PURPLE ★ (↓BOS) star? = bearish break of structure

    Each star sits on the candle whose CLOSE broke the prior 50-bar swing
    extreme. The dash-dot line shows the exact level that got broken.

    Note: ★ stars are the AUTHORITATIVE BOS signal (matches labeller math).
    Purple ▼/▲ triangles are just informational swing markers — they may
    or may not appear regardless of BOS status. Use STARS for Q4.

    ❌ NO star left of N → click [n — None]. STOP.
    ✅ YES (correct color) → Continue to FINAL CLICK.
─────────────────────────────────────────────────────────────────────
FINAL CLICK (only if Q1–Q4 ALL passed):
    Green box + all 4 yes → click [b — Bull FVG]
    Red box + all 4 yes   → click [e — Bear FVG]
    Genuinely torn         → click [a — Ambiguous]
─────────────────────────────────────────────────────────────────────
```

**Quick sanity check:**
- Q2 YES alone does NOT mean bull. You must also pass Q3 and Q4.
- Only the FINAL CLICK box uses `b` or `e`. Every failed step uses `n`.

**Skipped on purpose:**
- Crit #1 (mitigation) → enforced at inference time, not in training labels.
- Crit #3 (S/R confluence) → disabled by default to keep training viable (467 positives vs 29 with it on).

Each click auto-saves to `data/gold_labels.csv`. Close tab anytime, reopen to resume.

---

## The pattern in one sentence

A Valid Fair Value Gap (FVG) is a 3-candle price gap where the reaction candle (N+2) closes on the correct side of the zone, the gap sits in the correct half of the current price range, and a recent break of structure preceded the gap.

---

## What you see on the chart

Each annotation window shows a **25-bar candlestick chart** (17 bars left, 7 bars right of target). The target candle (N) is marked by a **red dashed vertical line**. The reaction candle (N+2) is marked by a **green dotted vertical line**.

**Index convention:** the `candle_index` column in `data/gold_labels.csv` is the LABEL position = N+2. The chart's "N (target)" red line is at `candle_index − 2` (FVG middle). The title bar shows both: `Label idx X → FVG middle N=Y`. Don't be confused — annotate based on the red dashed line (N), not the title's "Label idx".

Overlays:

| Overlay | Colour | Meaning |
|---------|--------|---------|
| FVG zone | Shaded green (bull) / red (bear) | The gap zone: from `high[N-1]` to `low[N+1]` for bull, `low[N-1]` to `high[N+1]` for bear |
| Swing markers | Purple triangles | 50-bar pivot highs (▼) and pivot lows (▲) — informational only |
| **BOS stars (Q4 signal)** | Orange ★ ↑BOS / Purple ★ ↓BOS | Bar where close broke the prior 50-bar swing high/low. **This is the authoritative BOS signal — use this for Q4, not triangles.** |
| BOS broken level | Dash-dot horizontal line (matches star color) | The exact swing level that got broken at that bar |
| S/R lines | Orange dotted horizontal | 2 closest pivot levels — reference only, NOT a criterion (Crit#3 disabled) |
| Gann midpoint | Blue dashed horizontal | Midpoint of the 50-bar swing range |

---

## Step 1: Is there a geometric gap? (Criterion #1 — baseline)

A **bullish FVG** exists if:
- Bar N (the target) has a **bullish body** (close > open)
- There is a **gap** between bar N-1 high and bar N+1 low (`high[N-1] < low[N+1]`)

A **bearish FVG** exists if:
- Bar N (the target) has a **bearish body** (close < open)
- There is a **gap** between bar N-1 low and bar N+1 high (`low[N-1] > high[N+1]`)

**Reject immediately (label = None) if:**
- No visible gap (the candles overlap between N-1 and N+1)
- Bar N's body is flat or ambiguous

The FVG zone is highlighted as a shaded box on the chart. If you cannot see the box, there is no geometric FVG.

---

## Step 2: Does bar N+2 close on the correct side of the gap? (Criterion #2 — loose)

Look at the **second bar to the right of the red dashed line** — the green dotted line marks it (bar N+2).

The labeller uses the **loose** version of this criterion (per TradingLab video — "close inside OR in direction of zone"):

**Bullish FVG — passes if:** bar N+2 **close ≥ gap bottom** (`close[N+2] >= high[N-1]`). Price did not collapse back through the gap.

**Bearish FVG — passes if:** bar N+2 **close ≤ gap top** (`close[N+2] <= low[N-1]`). Price did not rip back through the gap.

ELI5: "Did price respect the gap by staying on the bullish/bearish side of it?"

**Why loose, not strict:** the strict version (close *inside* the zone) passes only ~10% of geometric FVGs on SPY H1 — gaps are too tight (~1 pt wide), price almost always overshoots. Loose keeps the directional intent and gives a trainable label rate.

**Fail cases:**
- Bullish FVG, N+2 closes BELOW gap bottom → reversal, label None.
- Bearish FVG, N+2 closes ABOVE gap top → reversal, label None.

---

## Step 3: SKIPPED — S/R confluence (Criterion #3 — disabled by default)

Crit #3 is **disabled** in the active labeller (`require_crit3=False`). The orange dotted S/R lines on the chart are shown for context only — do not use them as a pass/fail gate.

**Why disabled:** combining all 6 strict criteria yielded only 6–29 positives on the full SPY H1 dataset (need ≥ 150 for training). Dropping #3 and relaxing #6 gave 467 positives — enough to train.

If you really want to enable it later, set `require_crit3=True` in `ValidFVGLabeller(...)` and re-run the sparsity gate.

---

## Step 4: Is this an extreme FVG in the swing? (Criterion #4 — Priority)

For annotation purposes, **label all valid FVGs** — do not skip a FVG just because another nearby one seems "more extreme." The model learns priority from the relative label density.

You only need to apply this step if multiple FVG zones are visible in the same chart window:
- **Bullish FVGs:** prefer the one with the **lowest bottom** (deepest in the range).
- **Bearish FVGs:** prefer the one with the **highest top** (deepest in the range from above).

If in doubt, label both as Confirm — annotation priority is conservative.

---

## Step 5: Is the gap in the correct half of the range? (Criterion #5 — Gann box)

Look at the **blue dashed horizontal line** (Gann midpoint). This line divides the 50-bar swing range in half.

**Bullish FVG — passes if:** the bottom of the shaded FVG zone is **at or below the blue dashed line**.
ELI5: "Is the bullish gap zone in the lower half of the current price range?"

**Bearish FVG — passes if:** the top of the shaded FVG zone is **at or above the blue dashed line**.
ELI5: "Is the bearish gap zone in the upper half of the current price range?"

**Fail cases:**
- Bull FVG whose bottom is clearly above the midpoint line: **FAILS** (gap is too high in the range to be a valid demand zone).
- Bear FVG whose top is clearly below the midpoint line: **FAILS** (gap is too low in the range to be a valid supply zone).

---

## Step 6: Was there a structure break recently to the left? (Criterion #6)

Look left on the chart — up to **20 bars** to the left of the red dashed line.

A **bullish BOS** = a candle CLOSED above a prior swing high (a purple ▼ triangle). Look for a candle to the left whose close was visibly above the nearest purple downward triangle.

A **bearish BOS** = a candle CLOSED below a prior swing low (a purple ▲ triangle). Look for a candle to the left whose close was visibly below the nearest purple upward triangle.

ELI5: "Did the market break structure before creating this gap?"

**Passes if:** you can identify at least one such close (above a prior high for bull FVG, below a prior low for bear FVG) within the visible left portion of the chart.

**Fail cases:**
- Price has been ranging without breaking any visible swing high/low in the past 20 bars.
- The nearest swing markers are all further left than 20 bars.

**Mark Ambiguous** if it is unclear whether a BOS occurred (e.g., marginal close at the exact level of a prior high).

---

## Decision table

| Step | Criterion | Action if FAILS |
|------|-----------|-----------------|
| 1 | Geometric FVG exists | → Label: **None** immediately |
| 2 | N+2 closes inside gap | → Label: **None** |
| 3 | S/R level nearby | → Label: **None** |
| 4 | Extreme in swing | → Label all passing, skip if clearly secondary |
| 5 | Gann box position | → Label: **None** |
| 6 | Recent BOS | → Label: **None** |
| All pass | — | → Label: **Confirm (bull or bear)** |

---

## Keyboard shortcuts in the notebook

| Key | Label |
|-----|-------|
| `b` | Bullish FVG (Confirm — bull) |
| `e` | Bearish FVG (Confirm — bear) |
| `n` | None (any criterion fails) |
| `a` | Ambiguous (unsure — skip and come back) |

---

## What to mark Ambiguous

Mark **Ambiguous** when:
- Steps 1–5 are clear but Step 6 (BOS) is a close call (candle nearly reached but didn't clearly close above/below).
- The FVG zone is very narrow and it is hard to tell if N+2 is inside or just outside.
- Two criteria are borderline at the same time.

Ambiguous candles contribute `label=0` to the gold set but are flagged for review. They do NOT increase kappa noise — they are excluded from the kappa computation.

---

## Reference definitions

**Fair Value Gap (FVG):** A 3-candle pattern where there is a price gap between the outer candles. Bullish FVG: `high[N-1] < low[N+1]`. Bearish FVG: `low[N-1] > high[N+1]`.

**Reaction candle (N+2):** The candle two bars after the FVG middle candle N. Its close confirms whether price accepted the zone.

**Swing high/low:** The highest high or lowest low within a 50-bar rolling window. Shown as purple triangles.

**Break of Structure (BOS):** A candle close that exceeds a prior swing high (bullish BOS) or falls below a prior swing low (bearish BOS). Indicates directional intent before the FVG formed.

**Gann midpoint:** The price level at the midpoint of the 50-bar swing range: `swing_low + 0.5 × (swing_high - swing_low)`. Shown as the blue dashed line.

**S/R (Support/Resistance):** Pivot high/low levels computed from the prior 5 bars of price history. Shown as orange horizontal lines.

**Unmitigated:** The FVG has not yet been "filled" by price returning through the zone after formation. The annotation tool shows FVGs at time of formation — mitigation is checked at inference time, not during annotation.

---

## Notes for this annotation session

- You are re-annotating the same 75 candle indices from the previous session.
- The programmatic label is **not shown** to prevent anchoring bias.
- Aim for ~2 minutes per candle.
- If you see a candle that is obviously "none" (no FVG, flat market), use `n` immediately without working through all 6 steps.
- The annotation resumes automatically from where you left off if interrupted.
