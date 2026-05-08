# smart-money-concepts Library — Lookahead Audit

> **Ready for /nb:plan.**
> Question: Does the `smart-money-concepts` Python library use future candle data when computing FVG labels? If yes, what is the exact lookahead depth and what mitigation is required?
> Verdict: **CONFIRMED lookahead — 1 bar on every FVG label.** The library is NOT safe to use as-is for DL training. Mitigation path: shift labels +1 bar in post-processing (not causal=True, which does not exist in v0.0.27). A custom 30-line implementation is the cleanest fix if label semantics matter precisely.

**Confidence:** High
**Why this confidence:** (a) Read actual source at `/tmp/smc.py` (688 lines of production code). (b) Reproduced falsification test on a 9-candle synthetic fixture — confirmed label at index 4 uses `low[5]` (bar N+1). (c) Verified against open GitHub issues #95, #101, #103 that independently confirm the same bugs. (d) PR #95 is unmerged — v0.0.27 (latest, installed) does NOT have the fix.
**Depth used:** Deep

---

## Project context

Master plan (`master-phasing.md`) flags lookahead as the highest-probability failure mode. This stream was spawned to confirm or deny before Phase 3 labelling code is written. No existing `src/` code yet — this research directly shapes `src/data/label.py` design.

---

## Findings

### 1. Library scope — what smc v0.0.27 implements

**Source:** https://github.com/joshyattridge/smart-money-concepts/blob/master/smartmoneyconcepts/smc.py (read in full)

Functions available (all `@classmethod`):
- `smc.fvg(ohlc, join_consecutive=False)` — Fair Value Gap
- `smc.swing_highs_lows(ohlc, swing_length=50)` — Swing Highs/Lows (used by OB, BOS, CHoCH)
- `smc.bos_choch(ohlc, swing_highs_lows, close_break=True)` — Break of Structure / Change of Character
- `smc.ob(ohlc, swing_highs_lows, close_mitigation=False)` — Order Blocks
- `smc.liquidity(ohlc, swing_highs_lows, range_percent=0.01)` — Liquidity
- `smc.previous_high_low(ohlc, time_frame='1D')` — Previous session H/L
- `smc.retracements(ohlc, swing_highs_lows)` — Fibonacci retracements
- `smc.sessions(ohlc, session, ...)` — Session marking

MVP scope (FVG only) means only `smc.fvg()` is directly relevant. `smc.swing_highs_lows()` is a dependency of OB/BOS/CHoCH — irrelevant for MVP but audited below for completeness.

**Version in pypi:** 0.0.27 (latest). Installed successfully in clean venv (`numpy<2`, `pandas`).

---

### 2. FVG implementation — exact algorithm

**Source:** smc.py lines 56–134

```python
# Bullish FVG condition at row i:
(ohlc["high"].shift(1) < ohlc["low"].shift(-1))  # high[i-1] < low[i+1]
& (ohlc["close"] > ohlc["open"])                  # candle i is bullish

# Bearish FVG condition at row i:
(ohlc["low"].shift(1) > ohlc["high"].shift(-1))   # low[i-1] > high[i+1]
& (ohlc["close"] < ohlc["open"])                  # candle i is bearish
```

The `top` and `bottom` bounds also use `shift(-1)`:
```python
top    = ohlc["low"].shift(-1)   # for bullish: top of gap = low[i+1]
bottom = ohlc["high"].shift(1)   # for bullish: bottom of gap = high[i-1]
```

**Lookahead depth:** exactly 1 bar (`shift(-1)`) for every FVG label. The label is placed at row `i` but uses `ohlc` row `i+1`.

**ICT textbook definition match:** The 3-candle pattern (N-1, N, N+1) is correctly identified. The issue is not the pattern definition — it's the index assignment. Labelling at N means claiming knowledge of N+1 at time N. The library's docstring says "the previous high is lower than the next low" but does not acknowledge that "next" requires a future bar. This matches ICT definition but the label index convention is wrong for ML.

**MitigatedIndex:** Computed forward-looking (finds when price later enters the FVG zone). This is correct conceptually — mitigation is always a future event — but the raw `MitigatedIndex` column should not be used as a feature in DL training. It is inherently future-data.

---

### 3. Lookahead audit — per label type

| Label | Index assigned at | Future data used | Lookahead bars |
|-------|-------------------|------------------|----------------|
| `FVG` direction (1/-1) | row N | `low[N+1]` or `high[N+1]` | **1 bar** |
| `Top` (FVG top bound) | row N | `low[N+1]` (bullish) | **1 bar** |
| `Bottom` (FVG bottom bound) | row N | `high[N+1]` (bearish) | **1 bar** |
| `MitigatedIndex` | row N | far-future row where price re-enters | **K bars (unbounded)** |

`swing_highs_lows()` (not MVP, but documented):
```python
ohlc["high"].shift(-(swing_length // 2)).rolling(swing_length).max()
```
Uses `swing_length // 2` bars of lookahead (default `swing_length=50` → 25 bars forward). Issue #101 tested this and found PF drops from 7.32 to 1.82 when bias removed — confirming the lookahead was doing meaningful work and inflating results.

---

### 4. Falsification test — synthetic fixture

**Fixture (9 candles, 0-indexed):**
```
i=3: high=100 (N-1 candle)
i=4: open=102, close=103 (bullish, N candle)
i=5: low=105  (N+1 candle — GAP EXISTS: 100 < 105)
```

**Theory says:** One FVG at bar 4 (between high[3]=100 and low[5]=105).

**Library returned:** FVG at indices **[3, 4, 5]** — three labels.

Explanation: the library applies the condition rolling across all rows, so any bar that is bullish and whose N-1/N+1 pair forms a gap gets labelled. In this fixture, bars 3, 4, 5 all satisfy the condition because the entire price run is ascending with consecutive gaps. This is technically correct per ICT (each bar sees a gap to its neighbours), but it means the label density is higher than the single "the FVG" concept implies. For a DL classifier, this is fine — the label means "this bar is part of a fair value gap zone" — but the lookahead is still structurally present at every labelled index.

**Key verification:**
```
Library at i=4: uses shift(-1)[4] = low[5] = 105  (future bar)
Library at i=3: uses shift(-1)[3] = low[4] = 101  (future bar)
At time i=3, bar 4 hasn't closed yet. Label is contaminated.
```

Falsification result: **Lookahead confirmed at every FVG label.**

---

### 5. PR #95 — causal parameter (NOT yet merged, NOT in v0.0.27)

**Source:** https://github.com/joshyattridge/smart-money-concepts/pulls/95

PR adds `causal=False` parameter. When `causal=True`:
```python
fvg = np.roll(fvg, 1); fvg[0] = np.nan
top = np.roll(top, 1); top[0] = np.nan
bottom = np.roll(bottom, 1); bottom[0] = np.nan
```

This shifts all output arrays forward by 1 position. Effect: the label that was at row N moves to row N+1, so it only appears once bar N+1 has closed. This correctly removes the forward reference from the signal index.

**Critical caveat (from issue #101 discussion):** The `causal=True` approach for `fvg()` is sound — the shift by 1 is the correct fix for FVG. However, for `swing_highs_lows()` (which PR #95 also patches), the causal fix only shifts outputs without fixing the underlying centered-window detection. A deeper algorithm rewrite is needed for swing detection to be truly bias-free.

**Status:** PR open, unreviewed, unmerged. **v0.0.27 has no causal mode.** Calling `smc.fvg(df, causal=True)` raises `TypeError` in the installed package.

---

### 6. Known bugs and compatibility issues

| Issue | Status | Impact |
|-------|--------|--------|
| #65 — Remove LookAhead Bias (PR, merged then reverted by #73) | Reverted — bias is BACK | High — confirms maintainer awareness but no fix shipped |
| #95 — Add causal parameter | Open, unmerged | High — fix exists but not available |
| #101 — swing_highs_lows inflated backtest PF 7.32→1.82 | Open | MVP scope exclusion: swing not used for FVG-only |
| #103 — Fix lookahead in swing_highs_lows | Open PR | MVP scope exclusion |
| #94 — read-only buffer error in np.where | Open | Medium — can cause runtime error if modifying FVG arrays |
| #77 — numpy 1.24.3 install error Python 3.12 | Closed (fixed) | Resolved; numpy<2 required |
| #78 — setup.py updated for latest libraries | Closed | Compatible with current stack |

**Pandas/numpy version requirement:** Library requires `numpy<2` (numpy 2.x breaks it — AttributeError on `np.__version__`). Compatible with pandas current via clean venv. This creates a potential conflict with PyTorch 2.x stack — **verify numpy<2 is compatible with your PyTorch 2.x install before Phase 3**.

---

### 7. Comparison to ICT textbook definition

ICT defines a Bullish FVG as: "a 3-candle formation where candle 1's high is lower than candle 3's low, and candle 2 is bullish." The gap zone is `[high[1], low[3]]`.

The library **correctly implements the ICT pattern**. The definition itself requires candle 3 (N+1) to be seen — so any real-time detection of this pattern inherently requires waiting for N+1 to close. The library does not acknowledge this constraint in its API and places the label at N instead of N+1.

**Canonical "detection time" per ICT:** A FVG is knowable at the CLOSE of candle N+1. The label should therefore be timestamped at N+1 for any causal use.

---

## Recommendation

**Do NOT use `smc.fvg()` as-is for DL training.** The 1-bar lookahead contaminates every label.

**Mitigation path A (preferred — simple, auditable):** Write a 30-line custom FVG detector in `src/data/label.py` that places the label at bar N+1 (the close of the confirmation bar). This is:
- ~30 lines of pandas vectorised code
- Zero dependencies on `smartmoneyconcepts`
- Auditable, testable, deterministic
- Matches ICT canonical detection time exactly

**Mitigation path B (tolerable if A is too much work):** Use `smc.fvg()` then shift all output arrays forward by 1 (`.shift(1)` in pandas). This replicates what PR #95 `causal=True` would do. Drop first row (NaN from shift). This is safe but relies on the library's correctness for the pattern itself — and the library has other known bugs (#94 read-only buffer).

**Mitigation path C (not recommended):** Wait for PR #95 to merge and use `causal=True`. This is uncontrolled — PR is open with no merge timeline. Do not block Phase 3 on upstream.

**Recommended path: A.** Custom implementation. Reasons:
1. No numpy<2 dependency conflict risk with PyTorch.
2. No hidden bugs from the library (#94 read-only error).
3. Label logic is ~30 lines — not complex enough to justify a dependency.
4. Easier to test: write a pytest fixture with known FVGs, assert label indices are at N+1.

**Custom FVG implementation (reference — confirm in Phase 3 sub-plan):**
```python
def label_fvg(df: pd.DataFrame) -> pd.Series:
    """
    Bullish FVG: high[i-1] < low[i+1] AND close[i] > open[i]
    Bearish FVG: low[i-1]  > high[i+1] AND close[i] < open[i]
    Label placed at i+1 (first bar where pattern is fully confirmed).
    Returns Series: 1=bullish FVG confirmed, -1=bearish FVG confirmed, 0=none.
    """
    bull = (df["high"].shift(1) < df["low"].shift(-1)) & (df["close"] > df["open"])
    bear = (df["low"].shift(1) > df["high"].shift(-1)) & (df["close"] < df["open"])
    fvg_at_N = np.where(bull, 1, np.where(bear, -1, 0))
    # Shift to N+1: label appears when N+1 closes (pattern confirmed)
    label = pd.Series(fvg_at_N, index=df.index).shift(1).fillna(0).astype(int)
    return label
```

This is Path A in 10 lines. The `shift(1)` here shifts the label forward by 1 so it lives at the bar after the confirming candle closes.

---

## What /nb:plan needs to know

- `smc.fvg()` v0.0.27 has **1-bar lookahead on every label** — do not use directly.
- Custom FVG implementation is ~30 lines; `src/data/label.py` should NOT import `smartmoneyconcepts` for FVG labelling.
- `numpy<2` is required by the library — verify this does not conflict with PyTorch 2.x in Phase 3 env setup. If conflict exists, the custom implementation removes this risk entirely.
- `MitigatedIndex` column from `smc.fvg()` is inherently future-data (unbounded lookahead) — never use as a DL feature, only as a post-hoc analysis tool.
- The label at candle N means "a FVG was confirmed at the close of bar N" (i.e., bar N is bar N+1 in the library's internal notation post-shift). State this explicitly in the notebook and label documentation.
- Phase 3 test `tests/test_data_label.py` must include a synthetic fixture that asserts: given a known FVG pattern at bars 3–5, the label appears at index 5 (not 4 or 3).
- `swing_highs_lows()` is out of MVP scope but also has deep lookahead (`swing_length // 2` bars). If OB/BOS/CHoCH is ever added as an extension, it needs the same audit + fix.

---

## Open questions

- Does `numpy<2` conflict with PyTorch 2.x on the project's M4 Pro environment? (Resolve in Phase 3 env setup — smoke test before writing any label code.)
- For the ternary label (bullish FVG / bearish FVG / none): should "none" candles inside an active FVG zone be relabelled as "in-zone"? This is a label-strategy question for R3, not for this stream.

---

## What I couldn't verify

- Whether `numpy<2` is compatible with the `xlstm` package or PyTorch 2.x under MPS on Apple Silicon. This must be smoke-tested in Phase 3 environment setup.
- Whether `join_consecutive=True` in `smc.fvg()` changes the lookahead structure — from source inspection it does not (it only merges adjacent FVG labels in post-processing after the shift(-1) detection), but it was not tested in the synthetic fixture.
- Whether PR #95 will be merged before Phase 3 build starts. Assumed: no. Custom implementation assumed.
- Whether the library's FVG definition exactly matches the variant of ICT FVG used in the project's reference (some practitioners use a "gap filled" condition or require specific wick structure — ICT has multiple versions). This is a definition choice for R3/Phase 3 sub-plan.

---

## Falsification attempts (Deep)

**Attempt 1: "Maybe the label at index N is correct because it refers to detection, not signal emission."**

Counterargument: In a live or backtesting scenario, at bar N's close, bar N+1 has not printed. The label at N claiming "this is a FVG" requires `low[N+1]`, which is unknowable. Even if one argues the label is a retrospective tag, any DL model trained to predict "is this a FVG candle" using bar N's OHLCV as input will have been trained on labels that were assigned using N+1's data. The model cannot reproduce this computation at inference time when N+1 hasn't printed. This is textbook temporal leakage.

Verdict: argument fails. Lookahead IS harmful for this use case.

**Attempt 2: "The lookahead is only 1 bar — on H1 data that's 1 hour. Is it really consequential?"**

Yes. At inference time you would need N+1's low to confirm a bullish FVG at N. If you're trading on H1 bars, that's 60 minutes of future price data. Any model trained on these labels would look like it works in backtest but fail live. The magnitude of the temporal leakage doesn't change the severity — leakage of any size invalidates causal claims.

Verdict: argument fails.

**Attempt 3: "Use smc.fvg() with a post-hoc shift — is this really equivalent to a custom implementation?"**

Nearly. The custom implementation (Path A) and post-hoc shift of library output (Path B) produce identical FVG signals if the library's pattern detection is correct. The difference is: Path B still depends on the library for the correct computation of `Top` and `Bottom` FVG bounds, which also use `shift(-1)`. After `.shift(1)` on the output, the bounds would be correct. However, the library's #94 read-only buffer bug can cause silent errors when the output arrays are modified. Path A avoids this entirely.

Verdict: Path A is cleaner; Path B is acceptable fallback if bounds are needed quickly.
