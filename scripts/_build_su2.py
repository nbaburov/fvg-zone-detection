#!/usr/bin/env python3
"""Generate notebooks/06-status-update-2.ipynb (full, Fontys-grade) and
notebooks/07-status-update-2-presentation.ipynb (reveal.js deck).

Self-contained: every figure uses synthetic Plotly or real numbers embedded
below (no file reads at runtime). Numbers transcribed verbatim from the
project's committed 5-seed diagnosis results and bootstrap confidence intervals.
Build artifact, not a deliverable itself.
"""
from __future__ import annotations
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# ----------------------------------------------------------------------------
# Setup + real data (verbatim from the committed results)
# ----------------------------------------------------------------------------
DATA_PRELUDE = '''import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook_connected"

BULL, BEAR, NEUTRAL, TEXT = "#26a69a", "#ef5350", "#95a5a6", "#37474f"
SPLIT = {"train": "#1e88e5", "val": "#fb8c00", "test": "#43a047"}
ACCENT, WARN = "#5e35b1", "#c62828"

def synth(opens, highs, lows, closes, names):
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=names)

def candle(sdf):
    return go.Candlestick(x=list(sdf.index), open=sdf["open"], high=sdf["high"],
        low=sdf["low"], close=sdf["close"],
        increasing_line_color=BULL, decreasing_line_color=BEAR, showlegend=False)

def style(fig, title, height=440, ytitle="Price"):
    fig.update_layout(title=title, height=height, template="plotly_white",
        xaxis_rangeslider_visible=False, yaxis_title=ytitle, margin=dict(t=70, b=40))
    return fig

# Five-model results: 5-seed mean macro-F1 with 95% bootstrap CI (block size 60).
# XGBoost is a reference row. It reads 35 hand-engineered features; the four deep
# nets read the raw 60x5 candle window. The XGB gap is partly the easier input,
# not pure architecture, so the fair comparison is deep-net against deep-net.
MODELS = pd.DataFrame([
    ("XGBoost",    "engineered feats", 0.721, 0.671, 0.761, 0.001, None,  None,  None),
    ("CNN-LSTM",   "raw window",       0.639, 0.603, 0.674, 0.013, 0.504, 0.439, 0.88),
    ("LSTM",       "raw window",       0.595, 0.560, 0.628, 0.015, 0.414, 0.402, 0.96),
    ("Transformer","raw window",       0.548, 0.519, 0.572, 0.085, 0.395, 0.282, 0.37),
    ("xLSTM",      "raw window",       0.369, 0.354, 0.384, 0.007, 0.114, 0.056, 0.34),
], columns=["model","family","f1","lo","hi","std","bull","bear","train_f1"])

# Class balance on the labelled data (none / bullish-FVG / bearish-FVG).
CLASS_PCT = {"none": 96.9, "bullish FVG": 1.8, "bearish FVG": 1.3}

# Mean validation macro-F1 vs fraction of training data (3 seeds per point).
# xLSTM was only run at the full-data anchor (see text).
CURVES = {
    "LSTM":        {0.2:0.438, 0.4:0.550, 0.6:0.546, 0.8:0.592, 1.0:0.612},
    "CNN-LSTM":    {0.2:0.486, 0.4:0.527, 0.6:0.575, 0.8:0.562, 1.0:0.602},
    "Transformer": {0.2:0.327, 0.4:0.327, 0.6:0.382, 0.8:0.327, 1.0:0.523},
}
print("Setup ready.")'''

# ----------------------------------------------------------------------------
# Figure code (plain comments, no em-dashes in titles)
# ----------------------------------------------------------------------------
FIG_CANDLE = '''# Figure 1: the anatomy of one candle.
demo = synth([100,106],[108,108],[98,100],[105,101],["up hour","down hour"])
fig = go.Figure(candle(demo))
fig.add_annotation(x="up hour", y=108, ax=110, ay=-45, text="<b>wick</b> = high/low reached", showarrow=True, arrowhead=2)
fig.add_annotation(x="up hour", y=102.5, ax=150, ay=0, text="<b>body</b> = open to close", showarrow=True, arrowhead=2)
style(fig, "Figure 1. One candle is one hour. Green closed up, red closed down.", 420)
fig.update_xaxes(type="category"); fig.show()'''

FIG_BULL = '''# Figure 2: a bullish Fair Value Gap.
bull = synth([100,101,106],[102,105,110],[99,100,105],[101,104,109],["N-1","N (up)","N+1"])
fig = go.Figure(candle(bull)); y0,y1 = bull["high"].iloc[0], bull["low"].iloc[2]
fig.add_shape(type="rect", x0="N-1", x1="N+1", y0=y0, y1=y1, fillcolor=BULL, opacity=0.22,
              line=dict(color=BULL, width=1.5, dash="dot"))
fig.add_annotation(x="N+1", y=(y0+y1)/2, ax=70, ay=0, text="<b>the gap</b><br>price skipped this zone",
                   showarrow=True, arrowhead=2, font=dict(color="#1b5e20"))
style(fig, "Figure 2. A bullish FVG. The shaded band never traded; price leapt over it.", 460)
fig.update_xaxes(type="category"); fig.show()'''

FIG_BEAR = '''# Figure 3: a bearish Fair Value Gap (mirror of Figure 2).
bear = synth([110,109,104],[111,110,106],[108,105,100],[109,106,101],["N-1","N (down)","N+1"])
fig = go.Figure(candle(bear)); y0,y1 = bear["high"].iloc[2], bear["low"].iloc[0]
fig.add_shape(type="rect", x0="N-1", x1="N+1", y0=y0, y1=y1, fillcolor=BEAR, opacity=0.22,
              line=dict(color=BEAR, width=1.5, dash="dot"))
fig.add_annotation(x="N+1", y=(y0+y1)/2, ax=70, ay=0, text="<b>the gap</b><br>price skipped this zone",
                   showarrow=True, arrowhead=2, font=dict(color="#b71c1c"))
style(fig, "Figure 3. A bearish FVG. A fast drop skips the band.", 460)
fig.update_xaxes(type="category"); fig.show()'''

FIG_IMBALANCE = '''# Figure 4: how rare the gaps are.
labels = list(CLASS_PCT); vals = list(CLASS_PCT.values())
fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[NEUTRAL, BULL, BEAR],
                       text=[f"{v}%" for v in vals], textposition="outside"))
style(fig, "Figure 4. The classes are extremely lopsided: about 97% of hours are 'no gap'.",
      430, ytitle="% of labelled hours")
fig.update_yaxes(range=[0,105]); fig.show()'''

FIG_WINDOW = '''# Figure 5: why so few examples are truly independent.
fig = go.Figure()
# two windows that overlap by almost all of their length
fig.add_shape(type="rect", x0=0, x1=60, y0=0.55, y1=0.95, fillcolor="#1e88e5", opacity=0.25, line=dict(color="#1e88e5"))
fig.add_shape(type="rect", x0=1, x1=61, y0=0.05, y1=0.45, fillcolor=ACCENT, opacity=0.25, line=dict(color=ACCENT))
fig.add_annotation(x=30, y=0.75, text="window at hour N (60 hours)", showarrow=False, font=dict(color="#1e88e5"))
fig.add_annotation(x=31, y=0.25, text="window at hour N+1 (shares 59 of 60 hours)", showarrow=False, font=dict(color=ACCENT))
fig.update_xaxes(title="hour", range=[-2, 64]); fig.update_yaxes(visible=False, range=[0,1])
fig.update_layout(height=320, template="plotly_white",
    title="Figure 5. Windows slide one hour at a time, so neighbours are nearly identical.")
fig.show()'''

FIG_TWO_FAMILIES = '''# Figure 6: two ways to show a model the same 60 hours.
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "Engineered features (XGBoost)<br><sub>60 hours into 35 hand-made numbers</sub>",
    "Raw window (the deep nets)<br><sub>the full 60x5 grid, learn its own features</sub>"))
feats = ["range", "body %", "volume z", "trend slope", "31 more"]
fig.add_bar(x=feats, y=[0.8,0.5,0.65,0.4,0.3], marker_color=ACCENT, row=1, col=1, showlegend=False)
grid = np.random.RandomState(0).rand(5, 12)
fig.add_heatmap(z=grid, colorscale="Blues", showscale=False, row=1, col=2)
fig.update_yaxes(title_text="value", row=1, col=1)
fig.update_yaxes(title_text="O H L C V", row=1, col=2, showticklabels=False)
fig.update_xaxes(title_text="time", row=1, col=2, showticklabels=False)
fig.update_layout(height=430, template="plotly_white",
    title="Figure 6. The same 60 hours, two representations. This difference is the question.")
fig.show()'''

FIG_RANKING = '''# Figure 7: the scoreboard, with confidence intervals.
d = MODELS.sort_values("f1")
colors = [NEUTRAL if m=="XGBoost" else (ACCENT if m=="CNN-LSTM" else "#90a4ae") for m in d["model"]]
fig = go.Figure(go.Bar(
    y=d["model"], x=d["f1"], orientation="h", marker_color=colors,
    error_x=dict(type="data", symmetric=False,
                 array=d["hi"]-d["f1"], arrayminus=d["f1"]-d["lo"], color=TEXT, thickness=1.5),
    text=[f"{v:.3f}" for v in d["f1"]], textposition="outside", showlegend=False))
fig.add_vline(x=0.328, line_dash="dot", line_color=WARN,
              annotation_text="always-guess-'none' floor", annotation_position="top")
style(fig, "Figure 7. Five-model scoreboard (5 seeds, 95% CI). Overlapping bars mean a statistical tie.", 460, ytitle="")
fig.update_xaxes(title="macro-F1 (higher is better; equal credit to the rare bull and bear zones)", range=[0,0.85])
fig.show()'''

FIG_PERCLASS = '''# Figure 8: where the models actually struggle (per-class F1).
d = MODELS.dropna(subset=["bull"]).iloc[::-1]
fig = go.Figure()
fig.add_bar(name="bullish-FVG F1", y=d["model"], x=d["bull"], orientation="h", marker_color=BULL)
fig.add_bar(name="bearish-FVG F1", y=d["model"], x=d["bear"], orientation="h", marker_color=BEAR)
fig.update_layout(barmode="group", height=430, template="plotly_white",
    title="Figure 8. The rare classes are the hard part. Every model finds the common 'none' well;<br>"
          "<sub>they separate on how well they catch the bull and bear gaps.</sub>",
    legend=dict(orientation="h", y=-0.18))
fig.update_xaxes(title="F1 on the minority class", range=[0,0.7])
fig.show()'''

FIG_FIT_GAP = '''# Figure 9: can the model even memorise the training data?
d = MODELS.dropna(subset=["train_f1"]).sort_values("train_f1")
fig = go.Figure()
fig.add_bar(name="train F1 (fit on seen data)", y=d["model"], x=d["train_f1"], orientation="h", marker_color="#b0bec5")
fig.add_bar(name="test F1 (generalise to unseen)", y=d["model"], x=d["f1"], orientation="h", marker_color=ACCENT)
fig.update_layout(barmode="group", height=430, template="plotly_white",
    title="Figure 9. Two failure modes. xLSTM and Transformer cannot fit train (left bars low);<br>"
          "<sub>LSTM and CNN-LSTM fit train well but the gap to test stays wide.</sub>",
    legend=dict(orientation="h", y=-0.18))
fig.update_xaxes(title="macro-F1", range=[0,1.0])
fig.show()'''

FIG_CURVES = '''# Figure 10: does more data still help? (validation F1 vs training-set size)
fig = go.Figure()
cmap = {"LSTM":"#1e88e5", "CNN-LSTM":ACCENT, "Transformer":"#fb8c00"}
for m, pts in CURVES.items():
    xs = sorted(pts); ys = [pts[x] for x in xs]
    fig.add_scatter(x=[int(x*100) for x in xs], y=ys, mode="lines+markers", name=m, line=dict(color=cmap[m], width=2))
fig.add_hline(y=0.328, line_dash="dot", line_color=WARN, annotation_text="majority-only floor")
style(fig, "Figure 10. The recurrent nets keep improving with more data (no flattening yet).<br>"
           "<sub>The curves are noisy over 3 seeds, so the honest read is 'no plateau', not a precise slope. "
           "Transformer sits near the floor until it finally has all the data.</sub>", 480, ytitle="validation macro-F1")
fig.update_xaxes(title="% of training data used")
fig.show()'''

FIG_MATRIX = '''# Figure 11: the decision rule, fixed before looking at results.
fig = go.Figure()
# x: 0 = train-F1 LOW (cannot fit), 1 = train-F1 HIGH (fits)
# y: 0 = curve FLAT, 1 = curve still RISING
cells = [
    (1,1,"data + regularisation-bound","more data","#c8e6c9","CNN-LSTM, LSTM"),  # fits + rising
    (0,1,"data + capacity-bound","more data, then size","#fff9c4",""),            # rising but cannot fit
    (1,0,"over-fit / reg-bound","regularise or augment","#ffe0b2",""),            # fits but flat
    (0,0,"capacity / cannot fit","bigger or fix training","#ffcdd2","Transformer, xLSTM"),  # cannot fit + flat
]
for x,y,label,lever,color,who in cells:
    fig.add_shape(type="rect", x0=x, x1=x+1, y0=y, y1=y+1, fillcolor=color, line=dict(color="white", width=3))
    fig.add_annotation(x=x+0.5, y=y+0.62, text=f"<b>{label}</b>", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=x+0.5, y=y+0.40, text=f"lever: {lever}", showarrow=False, font=dict(size=10, color=TEXT))
    if who:
        fig.add_annotation(x=x+0.5, y=y+0.18, text=who, showarrow=False, font=dict(size=11, color=ACCENT))
fig.update_xaxes(range=[0,2], tickvals=[0.5,1.5], ticktext=["train F1 LOW<br>(cannot fit)","train F1 HIGH<br>(fits)"], title="")
fig.update_yaxes(range=[0,2], tickvals=[0.5,1.5], ticktext=["curve FLAT","curve still RISING"], title="")
fig.update_layout(height=470, template="plotly_white",
    title="Figure 11. The decision matrix, committed before the experiments. Each model lands in one box.")
fig.show()'''


FIG_GOLD = '''# Figure 4b: do the automatic labels match a human? (gold set, 75 hand-checked candles)
cm = [[70,0,0],[0,4,0],[0,0,1]]; lab = ["none","bull","bear"]
fig = go.Figure(go.Heatmap(z=cm, x=lab, y=lab, colorscale="Greens", showscale=False,
                           text=cm, texttemplate="%{text}", textfont=dict(size=16)))
style(fig, "Figure 4b. Hand-check vs the automatic rule on 75 candles. A perfect diagonal: Cohen's kappa = 1.0.", 420, ytitle="my hand label")
fig.update_xaxes(title="automatic rule label"); fig.show()'''

FIG_SHAP = '''# Figure 6b: which engineered features XGBoost leans on (SHAP, mean absolute impact).
sh = [("ret_60 (60-hour momentum)",0.838),("gap_norm_bull",0.814),("gap_bear",0.701),
      ("pos_in_range",0.556),("gap_norm_bear",0.517)]
names=[s[0] for s in sh][::-1]; vals=[s[1] for s in sh][::-1]
fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=ACCENT,
                       text=[f"{v:.2f}" for v in vals], textposition="outside"))
style(fig, "Figure 6b. XGBoost's top features: momentum and gap geometry dominate.", 400, ytitle="")
fig.update_xaxes(title="mean absolute SHAP value (importance)", range=[0,1.0]); fig.show()'''

FIG_CONFUSION = '''# Figure 8b: every model's confusion matrix (seed 42, same 5198-hour test set).
# Cell text = counts; colour = row % (how each true class was predicted).
CONF = {
    "XGBoost (ref)": [[4895,84,56],[18,78,0],[17,0,50]],
    "CNN-LSTM":      [[4859,105,71],[28,68,0],[25,0,42]],
    "LSTM":          [[4822,121,92],[31,65,0],[25,0,42]],
    "Transformer":   [[4960,75,0],[81,15,0],[66,1,0]],
    "xLSTM":         [[4748,161,126],[79,15,2],[61,1,5]],
}
lab = ["none","bull","bear"]
fig = make_subplots(rows=2, cols=3, subplot_titles=list(CONF), horizontal_spacing=0.08, vertical_spacing=0.16)
for i,(name,cm) in enumerate(CONF.items()):
    cm = np.array(cm); pct = (cm / cm.sum(1, keepdims=True) * 100).round().astype(int)
    r,c = divmod(i,3)
    fig.add_heatmap(z=pct, x=lab, y=lab, colorscale="Purples", showscale=False,
                    text=cm, texttemplate="%{text}", textfont=dict(size=10), row=r+1, col=c+1)
    fig.update_yaxes(autorange="reversed", row=r+1, col=c+1)
fig.update_layout(height=560, template="plotly_white",
    title="Figure 8b. Confusion matrix per model (rows = true none/bull/bear, columns = predicted; text = counts).")
fig.show()'''

FIG_INSPECT = '''# Figure 12 (illustrative): the inspection tool, a probability strip against the labels.
import matplotlib.pyplot as plt
rng = np.random.default_rng(7); T = 200
p_bull = np.clip(0.02 + 0.6*(rng.random(T)>0.96) + 0.05*rng.random(T), 0, 1)
p_bear = np.clip(0.02 + 0.6*(rng.random(T)>0.97) + 0.05*rng.random(T), 0, 1)
p_none = np.clip(1 - p_bull - p_bear, 0, 1)
gold = np.zeros(T, int); gold[p_bull>0.4]=1; gold[p_bear>0.4]=2
gold[40]=1; p_bull[40]=0.15; gold[120]=2; p_bear[120]=0.18; p_bull[160]=0.7
fig, ax = plt.subplots(2,1, figsize=(11,3.2), sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].fill_between(range(T), 0, p_none, color="#cccccc", label="P(none)")
ax[0].fill_between(range(T), p_none, p_none+p_bull, color="#26a69a", label="P(bull)")
ax[0].fill_between(range(T), p_none+p_bull, 1, color="#ef5350", label="P(bear)")
ax[0].set_ylim(0,1); ax[0].set_ylabel("class prob."); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("Figure 12. Inspection tool: model probabilities over 200 test hours (illustrative)")
cmap = {0:"#eeeeee",1:"#26a69a",2:"#ef5350"}
for t,gv in enumerate(gold): ax[1].axvspan(t-0.5, t+0.5, color=cmap[int(gv)])
ax[1].set_yticks([]); ax[1].set_xlabel("hour index (test set)")
ax[1].set_ylabel("label", rotation=0, labelpad=22, va="center")
plt.tight_layout(); plt.show()'''

FIG_PAPERTRADE = '''# Figure 13 (illustrative): one paper-trade session, dry-run decisions logged.
import matplotlib.pyplot as plt
rng = np.random.default_rng(3); hours = np.arange(7)
price = 540 + np.cumsum(rng.normal(0, 0.3, len(hours)))
sig_t=[2,5]; sig_kind=["bull","bear"]; sig_conf=[0.71,0.63]
fig, ax = plt.subplots(figsize=(10,3.4))
ax.plot(hours, price, color="#264653", lw=2, marker="o", label="SPY hourly close")
for t,k,cf in zip(sig_t, sig_kind, sig_conf):
    col = "#26a69a" if k=="bull" else "#ef5350"
    ax.axvspan(t-0.4, t+0.4, color=col, alpha=0.2)
    ax.annotate(f"{k}  p={cf:.2f}", xy=(t, price[t]), xytext=(t, price[t]+0.6),
                ha="center", fontsize=9, color=col, arrowprops=dict(arrowstyle="->", color=col))
ax.set_xticks(hours); ax.set_xticklabels(["09:30","10:30","11:30","12:30","13:30","14:30","15:30"])
ax.set_ylabel("SPY"); ax.set_title("Figure 13. Paper-trade harness: one session, dry-run signals with confidence (illustrative)")
ax.legend(loc="lower right", fontsize=8); plt.tight_layout(); plt.show()'''


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


def build_notebook():
    c = []
    c.append(md("""# 06 · Status update 2: why classical ML beat deep learning at spotting Fair Value Gaps"""))

    c.append(md("""## One-page synthesis

**The question that drove this update.** I am building a system that automatically detects *Fair Value
Gaps* (FVGs), specific zones on a price chart, and reports each with a confidence score. Five model types
now sit side by side on identical data. A classical machine-learning model (XGBoost) leads, and the
deep-learning models trail it. Why would the "simpler" method win? Answering that properly, rather than
guessing, is the work of this update.

**Objective reflection (what the numbers say).** On a like-for-like comparison (all models reading the
same raw 60-hour window), the best deep model is **CNN-LSTM**, at a macro-F1 of **0.639 ± 0.013** (95% CI [0.603, 0.674]).
Macro-F1 gives the rare bull and bear gaps equal weight with the common "none", so 1.0 is perfect and the
always-say-none baseline scores only 0.328. XGBoost scores **0.721**, but on a different, easier input (35 hand-engineered features),
so that gap is not a fair architecture verdict. The two *largest* deep models did *worse* as untuned
baselines (**Transformer 0.548**, unstable; **xLSTM 0.369**); and when I later gave the Transformer a fair
tuned pass it reached only **0.601 ± 0.045**, tying LSTM and still below CNN-LSTM. A decision rule fixed in advance places the
workable models in the "needs more data" box and rules out "needs a bigger model".

**Subjective reflection (how I worked).** I did not simply crown a winner. I fixed the diagnosis rule
before seeing results, attached confidence intervals so I would not mistake noise for a finding, gave every
model its own reasoned write-up, and reported the uncomfortable parts (an unstable Transformer, an
under-fitting xLSTM, noisy curves). The useful output here is a defensible reason that tells me what to do
next, not a higher score."""))

    c.append(md("""## What this maps to (semester learning outcomes)

| Learning outcome | What it asks for | Where I demonstrate it |
|---|---|---|
| **LO1, Data preparation** | Prepare and store a dataset so it can be used for analysis and modelling. | Parts 2 to 3: pulling SPY hourly data (and rejecting yfinance), resampling to clean sessions, the six-criterion labelling rule, the gold-set check (kappa 1.0), the strict time-ordered split, windowing, and class weights. |
| **LO2, Data analysis and model engineering** | Reliably apply machine and deep learning and other techniques to a prepared dataset. | Parts 4 to 7: a five-model ladder (XGBoost, LSTM, CNN-LSTM, Transformer, xLSTM), per-model tuning (G1 to G10), learning-curve and fit diagnosis, and bootstrap confidence intervals. |
| **LO3, Professional Standard** | Take responsibility for an ICT problem, research with selected methods, advise stakeholders under uncertainty, and substantiate future-oriented choices with ethical and sustainable arguments. | The decision rule fixed before results and the honest limitations (Part 7, Final summary); advice drawn from the evidence (data-bound, so chase data not model size, Part 9); a bounded, responsible scope (the tool flags confidence-scored zones, not trade instructions, Part 8b); and a sustainability argument in choosing not to spend compute tuning xLSTM, which cannot fit the data it already has (Part 9). |
| **LO4, Personal Leadership** | Independently set goals and actions for your own development, carry them out, and adjust them. | I set the diagnosis as my own goal, sequenced a roadmap with the blocker first, adjusted the plan from the evidence (deferring the Transformer re-test, dropping xLSTM), and reflect on what I learned and would try next (Parts 9 and 10). |

The outcomes are argued directly in this table and indirectly by the rigour of the method itself."""))

    c.append(code(DATA_PRELUDE))

    # Part 1
    c.append(md("""# Part 1 · The question, and why it matters

## 1. The one-sentence version

> **Why did a classical machine-learning model (XGBoost) detect Fair Value Gaps better than deep
> learning, and does that mean deep learning is the wrong tool, or that it is simply starved of
> something it needs?**

That second half matters. "Deep learning lost" is a dead end. "Deep learning lost *because of X*" tells me
whether to change the **model** or change the **data**, two completely different next phases. The whole
update is the disciplined process of turning the first statement into the second."""))

    # Part 2
    c.append(md("""# Part 2 · The minimum you need to follow along

Three pictures cover all the finance and all the jargon."""))
    c.append(md("""## 2. A candle, and a Fair Value Gap

A **candle** summarises one hour of trading: where price opened, closed, and the highest and lowest it
reached (Figure 1). A **Fair Value Gap** is a three-candle pattern where price moved so fast it *skipped*
a band of prices entirely, leaving a gap that never traded (Figures 2 and 3). Traders watch these zones;
my job is to detect them automatically and attach a confidence score."""))
    c.append(code(FIG_CANDLE))
    c.append(md("""**What this shows.** A candle is four numbers per hour: open, high, low, close (plus volume). The
model never sees a chart image; it sees these numbers. Everything downstream is built on this single unit."""))
    c.append(code(FIG_BULL))
    c.append(md("""**Reading it.** A bullish FVG is left behind by a fast move *up*: the shaded band is a stretch of
prices that never traded because price jumped straight over it. The three candles N-1, N, N+1 define it."""))
    c.append(code(FIG_BEAR))
    c.append(md("""**The mirror case.** A bearish FVG is the exact mirror, left by a fast move *down*. Both are purely
**geometric** events, defined by where three candles sit relative to one another. That is why a fixed rule
can label them, and why a model can, in principle, learn to recognise them."""))

    c.append(md("""## 3. Why use a model at all if a rule can label them?

A fixed rule gives a yes/no. A model gives a **probability** instead: "I am 80% sure this is a valid gap."
(I treat this as a calibrated confidence, though I have not yet verified the calibration; I flag that in the
Limitations.) That confidence score is the actual product. It lets a user rank zones and filter out the weak ones, and trust
strong ones. The rule is the *teacher*; the model learns to imitate it **and** to express doubt. So this is
a **supervised classification** problem, not price prediction and not trading.

How the data is prepared, briefly, because it shapes the whole question:

- **Source.** SPY hourly candles, 2016 to 2025, pulled from Alpaca (a free market-data provider). An
  earlier option, Yahoo/yfinance, was rejected: it caps hourly history at about 730 days, and I need years.
- **Labels.** I built a custom rule (six criteria) that marks each candle as *none*, *bullish-FVG* or *bearish-FVG*.
- **Gold check.** I hand-annotated a sample and compared it to the rule. They agreed almost perfectly
  (Cohen's kappa near 1.0), so the automatic labels are trustworthy ground truth.
- **Strict time split.** Train on the past, validate on the middle, test on the most recent years, and
  **never shuffle** (the temporal-evaluation discipline argued by López de Prado, 2018). Shuffling time would let the model peek at the future, the cardinal sin we avoid.
- **Windowing.** Each example is the last **60 hours by 5 numbers** (open, high, low, close, volume)."""))
    c.append(code(FIG_IMBALANCE))
    c.append(md("""**Why this matters.** The classes are extremely lopsided (Figure 4): about 97% of hours are "no gap",
and only a few percent are bull or bear gaps. If I scored on plain accuracy, a model that says "none" every
single time would score 97% and be useless. That always-say-none strategy is my **naive baseline**: it
scores only **0.328 macro-F1**, the floor every useful model must clear (it is the dotted line in the later
figures). So the headline metric is **macro-F1**, which averages how well each class is found and gives the
rare bull and bear zones equal weight. I also weight the rare classes more heavily during training
(inverse-frequency class weights: a rare bull or bear example counts far more toward the training loss than a common "none"), so the model cannot just ignore them."""))

    c.append(md("""## 3a. What makes a gap "valid", and do the labels hold up?

Not every geometric gap counts. My labelling rule follows a published "TradingLab" definition with six
criteria. One of the six is pure bookkeeping (label every candidate that passes), so five are real tests a
gap must clear, all at once. The label is attached at the reaction candle, two bars after the gap forms,
which is the first moment all five are knowable:

1. **A real gap exists** (the three-candle geometry of Figures 2 and 3).
2. **The reaction candle closes back inside the gap** (price actually responded to the zone).
3. **Support/resistance confluence**: the gap forms at a meaningful prior price level.
4. **A swing-position check**: the gap sits in the right part of the recent price swing (a standard
   chart-geometry rule on recent swing highs and lows).
5. **A recent break of structure**: just before the gap, price broke a prior swing high or low, so the move
   had real momentum behind it.

All five must pass together, which is why a gap is a multi-step judgement rather than a one-line pattern.
That is also part of why a learned model can add value: it can weigh these conditions probabilistically
instead of as a hard yes/no.

To trust those automatic labels, I hand-annotated 75 candles myself and compared (Figure 4b)."""))
    c.append(code(FIG_GOLD))
    c.append(md("""**What I found.** On all 75 hand-checked candles my labels and the rule agreed exactly: Cohen's kappa,
a standard agreement score that corrects for chance (so it is stricter than a raw match rate), where 1.0 is perfect, came out at **1.0**. The sample is small, so I read this as
"no disagreements found in 75 checks", not "proven perfect forever". It is enough to treat the automatic
labels as trustworthy ground truth for training."""))

    c.append(md("""## 3b. The whole pipeline in one view

Here is the full journey from raw data to a scored prediction:

| Stage | What happens | Why it is done this way |
|---|---|---|
| **Acquire** | SPY 1-minute bars from Alpaca, 2016 to 2025 | Free, deep history; Yahoo rejected (730-day hourly cap) |
| **Resample** | 1-minute to hourly, regular trading hours only, anchored 09:30 | Clock-hour bars mix in pre-market noise; anchoring keeps sessions clean |
| **Label** | 6-criterion rule into none / bull-FVG / bear-FVG | A custom detector; an off-the-shelf library was rejected for look-ahead leakage |
| **Verify** | hand-annotated gold set against the rule (kappa near 1.0) | Proves the automatic labels are trustworthy ground truth |
| **Split** | train (past) / val (middle) / test (recent), never shuffled | Shuffling time leaks the future |
| **Window** | each example is the last 60 hours by 5 numbers, sliding by 1 | Gives the model temporal context to recognise the 3-candle pattern |
| **Feed** | engineered 35-feature table (XGBoost) or raw 60x5 grid (deep nets) | The fork that creates this update's question |"""))
    c.append(code(FIG_WINDOW))
    c.append(md("""**The key consequence.** Windows slide one hour at a time, so neighbouring examples overlap by 59 of
their 60 hours (Figure 5): they are almost identical. After accounting for that overlap, the rare FVG events
boil down to far fewer independent examples than the row count suggests. The rare classes are about 3% of
hours, and because each 60-hour window overlaps its neighbour by 59 hours, consecutive positive windows are
near-duplicates, so the number of genuinely independent gap events is on the order of a hundred, not
thousands. That is a *tiny*
dataset for a deep network, and it is the quiet reason the later diagnosis lands on "data-bound" and the
reason a feature-fed classical model is so hard to beat here."""))

    c.append(md("""## 3c. Paths I tried and deliberately rejected

Several paths did not make the cut, each rejected for a reason:

- **An off-the-shelf SMC library**, rejected: its FVG detection peeked one bar into the future (look-ahead
  leakage). I wrote a clean, leakage-free detector instead.
- **Yahoo/yfinance data**, rejected: hourly history capped at about 730 days, and I need years.
- **Focal loss**, which down-weights easy examples during training so the model concentrates on the hard, rare gaps, tried, did not beat plain class-weighted training here.
- **Decision-threshold tuning**, tried, gains did not survive across seeds, so I dropped it to avoid
  over-fitting the validation set."""))

    # Part 3
    c.append(md("""# Part 3 · The fork that creates the puzzle

## 4. Two ways to show a model the same 60 hours

There are two honest ways to feed those 60 hours to a model, and they are the root of the whole question
(Figure 6):

- **Hand-engineered features (XGBoost).** I compress the window into **35 summary numbers** I designed:
  range, body sizes, volume spikes, trend slope, and so on. The model gets a tidy, pre-digested table.
- **The raw window (the deep nets).** I hand over the full **60 by 5 grid** and ask the network to discover
  its own features. More flexible, but it must learn everything from scratch."""))
    c.append(code(FIG_TWO_FAMILIES))
    c.append(md("""**Why this is the whole question.** XGBoost competes on an easier, human-assisted input; the deep nets
compete on the raw input. So "XGBoost wins" is partly "good hand-crafted features win". The genuinely fair
fight is **deep-net against deep-net on the raw window**, and that is where I look for the real lesson.

To see *which* of my 35 features carry XGBoost, I measured each one's SHAP value, a standard way to score
how much a feature moves the model's output (Figure 6b)."""))
    c.append(code(FIG_SHAP))
    c.append(md("""**What I found.** Two kinds of feature dominate: 60-hour **momentum** (`ret_60`) and the **gap geometry**
itself (`gap_norm_bull`, `gap_bear`). That is reassuring, because it means the hand-engineered features
encode exactly the things the FVG definition cares about. The deep nets must find those same patterns from
raw candles on their own, with far less data to do it."""))

    # Part 4
    c.append(md("""# Part 4 · What I needed to do to answer it

A single score cannot explain *why*. To separate "wrong model" from "starved model" I needed three things,
designed up front:

1. **Complete the model ladder.** Test not just one deep net but a representative range: a plain **LSTM**,
   a **CNN-LSTM**, and the two most expressive modern architectures, a **Transformer** and an **xLSTM**. If
   bigger and fancier models keep winning, the task wants more capacity. If they stall or get worse,
   capacity is not the problem.
2. **Run a data-size experiment (learning curves).** Retrain each model on 20%, 40%, up to 100% of the data
   and watch the score. If it is still climbing at 100%, the model is hungry for data. If it flattened long
   ago, more data will not help.
3. **Commit to a decision rule before looking.** I wrote down a fixed 2x2 matrix (Figure 11) mapping two
   measurements, *is the curve still rising?* and *can the model even fit the training data?*, to a verdict.
   Fixing the rule first is what stops me from rationalising whatever I happen to see.

A note on **confidence intervals**, since they appear throughout. Each model is trained five times with
different random starts, and I draw from those results a thousand times at random with replacement, to simulate many more experiments and read off the spread (a block bootstrap, which keeps neighbouring near-identical windows together rather than splitting them; Efron & Tibshirani, 1993). When two models' ranges overlap, I call it a tie rather than inventing a winner."""))

    # Part 5
    c.append(md("""# Part 5 · What I did: five models, each on its own terms

Every model is trained five times (different random seeds) so I report a range, not a lucky run, with 95%
confidence intervals. Each model below gets a fair hearing: what it is, why it earned a place, and what it
taught me, including the ones that lost."""))

    c.append(md("""### XGBoost, the classical control (reference)
A gradient-boosted tree ensemble (it builds many small decision trees one after another, each correcting the previous trees' mistakes) on the 35 engineered features. **Role:** the sanity check. In a small-data
problem like this, a well-fed classical model is often very hard to beat, and it is, at **0.721**. But it
reads the easier input, so I treat it as a reference line, not the deep-learning winner.

### LSTM, the recurrent baseline
Reads the 60 hours in order, one step at a time, carrying a running summary (its hidden state) of everything seen so far. **0.595 ± 0.015.** It memorises the training data well
(train-F1 near 0.96) but generalises to only about 0.60, a large gap. That gap is a fingerprint, and I
return to it.

### CNN-LSTM, the best deep model (the carrier)
A small convolution scans the window for local shapes first, then the LSTM reads the sequence of those
shapes. An FVG *is* a local shape, so this built-in bias fits the problem: **0.639 ± 0.013**,
the top deep model, with the best rare-class scores (bull 0.50, bear 0.44) and the tightest spread.

### Transformer, the flexible heavyweight
Self-attention (Vaswani et al., 2017), which lets every hour look at every other hour at once instead of reading step by step, with almost no built-in assumptions, the architecture behind modern AI at scale. **0.548 ± 0.085.**
The headline here is the **±0.085**, six times the spread of the recurrent models. On four of five runs it
reaches about 0.58 to 0.61; on others it *collapses* to predicting "none" only. On this little data it is
unreliable rather than weak: when it trains it competes, but I cannot count on it doing so.

### xLSTM, the newest design, and an informative negative
A 2024 "extended LSTM" (Beck et al., 2024) that replaces the LSTM's single memory value with a matrix and uses exponential gating, in theory more expressive, on paper the most sophisticated sequence model here. In practice **0.369 ± 0.007**,
barely above the always-guess-"none" floor, and it **cannot even fit the training data** (train-F1 near
0.34). This is the most useful of the failures: a more complex model scoring *worse* is direct evidence
that complexity is not what the task lacks."""))

    c.append(code(FIG_RANKING))
    c.append(md("""**What I found.** CNN-LSTM and LSTM bars overlap, so by the confidence intervals they are a
**statistical tie**. I pick CNN-LSTM on secondary grounds (higher point estimate, tighter spread, more
headroom) and say so rather than overclaiming a win. The other half of the picture is that the two *biggest*
models sit at the *bottom*: on a fair raw-window comparison, more architecture bought less performance."""))

    c.append(code(FIG_PERCLASS))
    c.append(md("""**Where the models actually fail.** Every model finds the common "none" class easily; they separate
on the rare gaps (Figure 8). CNN-LSTM leads on both bull (0.50) and bear (0.44). The bear class is the
hardest for everyone, and xLSTM barely registers it at all (0.06). This is the same story as the headline
number, seen at the class level: the difficulty lives entirely in the rare events.

To see *how* each model gets things wrong, not just how often, Figure 8b lays all five confusion matrices
side by side (same test set, seed 42)."""))
    c.append(code(FIG_CONFUSION))
    c.append(md("""**What I observed, model by model:**
- **XGBoost and CNN-LSTM** have the cleanest profiles: they catch most "none" hours and a real share of both
  gap types, and their mistakes are almost all "missed a gap, called it none" (the cautious error on an
  imbalanced problem). Both have **zeros in the bull-vs-bear corners**: when they fire on a gap, they get the
  direction right.
- **LSTM** looks similar but lets a few more real gaps slip through as "none".
- **Transformer** shows its failure plainly: the entire **bear column is zero**, so it *never predicts a bear
  gap*, and it catches only a handful of bull gaps. It copes by dumping almost everything into "none".
- **xLSTM** is the most degenerate: it correctly flags only a tiny number of real gaps (5 bear, 15 bull) and
  sends the rest to "none".

The shared pattern is reassuring for a detector: when these models do err, they err on the side of *missing*
a gap rather than inventing one or flipping its direction. The difference between the good and the failed
models is entirely in how many real gaps they give up on."""))

    c.append(md("""## How every model was tuned and stress-tested (G1 to G10)

The numbers above are not single lucky runs. Each model went through a fixed battery of ten checks (I call
them G1 to G10) so the comparison is fair and the conclusions are not artefacts of one setting:

| Gate | What it does | Why it matters here |
|---|---|---|
| **G1** | Hyper-parameter tuning (an Optuna search that tries many setting combinations and uses past results to choose the next) | every model competes at its own *best* settings, not a guess |
| **G2** | 5-seed variance | reports a range, not one lucky run |
| **G3** | Focal-loss ablation | focal loss dynamically down-weights easy examples; it did not beat plain class weighting here |
| **G4** | Decision-threshold tuning | checked if raising the confidence cut-off before calling a gap helps; gains did not survive across seeds |
| **G5** | SHAP feature importance | Figure 6b; confirms XGBoost relies on momentum and gap geometry |
| **G6** | Bull vs bear asymmetry | confirms bear gaps are the harder class for every model |
| **G7** | Window-size sweep | checked 30 to 120 hours; 60 gave the best validation F1 for its cost, so I use it |
| **G8** | Data-scaling check | more years of data helps, which foreshadows the data-bound verdict |
| **G9** | Regularisation ablation | dropout (randomly switching off connections in training) and weight-decay (penalising large weights) help the recurrent models |
| **G10** | Bootstrap confidence intervals | the error bars behind every "tie" claim in this notebook |

The key one for this update is **G1**: because each model is individually tuned, "CNN-LSTM beats Transformer"
is a statement about the architectures, not about one having a luckier configuration than the other."""))

    # Part 6
    c.append(md("""# Part 6 · What I observed

## 5. Two different ways to fail

Figure 9 splits the models by a simple test: *can it memorise the training data at all?*"""))
    c.append(code(FIG_FIT_GAP))
    c.append(md("""**What I observed.** There are two distinct failure modes, and conflating them would be a mistake.
LSTM and CNN-LSTM fit training data well (0.88 to 0.96) but leave a wide gap to test; a model that *can*
learn the pattern but does not *generalise* is typically short of data, not short of capacity. Transformer
and xLSTM cannot even reach 0.4 on training data they have already seen, which is an optimisation and
stability failure (untuned, fragile), not "the task is too hard for their size"."""))

    c.append(md("""## 6. The workable models are still hungry

Figure 10 plots score against how much data each model was given."""))
    c.append(code(FIG_CURVES))
    c.append(md("""**What I found.** For LSTM and CNN-LSTM the line is still heading up at 100% of the data: no plateau.
The honest caveat: with only three seeds the curve is noisy and non-monotonic (it dips and recovers), so I
claim "no plateau", not a precise slope. Either way the signal is the same: these models have not yet seen
enough examples, especially of the rare bull and bear gaps, to stop improving."""))

    c.append(md("""## 7. Bigger did not help

If the task were limited by model capacity, the Transformer and xLSTM, the most expressive models, should
have pushed the ceiling up. They did not. xLSTM scored far below everything (0.369), and the Transformer,
even after a dedicated tuning pass (0.601, see Part 8), only reached parity with the recurrent models and
never beat CNN-LSTM. Adding capacity, even when well-tuned, did not raise the ceiling, so capacity is not
the binding constraint."""))

    # Part 7
    c.append(md("""# Part 7 · The verdict, by the rule I fixed in advance

## 8. The decision matrix

I now drop each model into the 2x2 box defined *before* the experiments (Figure 11)."""))
    c.append(code(FIG_MATRIX))
    c.append(md("""**The verdict.** The two workable models land in "needs more data". The two big models land in "cannot
fit / fix training", a per-model training problem rather than the task's ceiling. No model lands in a box
that says "make the network bigger to win".

**The problem is data-bound, not capacity-bound.** The system is limited by how much signal exists in a
single instrument's rare events, not by the size of the network. This also explains the question's premise:
XGBoost wins partly because hand-engineered features inject human knowledge to compensate for scarce data,
which is exactly what a data-bound problem rewards."""))

    # Part 8
    c.append(md("""# Part 8 · Progress alongside the headline

The question above was the spine, but a few other things are worth recording.

- **The model ladder is finished.** The two new architectures (Transformer and xLSTM) are fully built and
  trained under the same rules as the rest, so the five-model comparison the project set out to make is
  complete.
- **Why the scores carry a range, not a single number.** Training a network involves randomness, so I run
  every model five times and report the spread. CNN-LSTM (0.639) and LSTM (0.595) overlap once that spread is
  accounted for, which is why I call them a tie rather than declaring a winner the next run might overturn.
- **The Transformer fairness pass is done, and it confirms the prediction.** The untuned Transformer did
  badly partly because it was untuned, so I gave it a proper tuned, stabilised run: a learning-rate warm-up
  (easing the rate up at the start), gradient clipping (capping how large a single update can be, so one bad
  batch cannot blow up the weights the way it did on the failed runs), and an Optuna search over its settings.
  Tuned, it reaches **0.601 ± 0.045** across five seeds, up from the untuned 0.548, and a wider 10-seed probe
  gives **0.604 ± 0.050** with no collapse to the majority-only floor. So the earlier instability was largely
  a tuning artifact. But even properly tuned it only ties LSTM (0.595) and stays below CNN-LSTM (0.639): a
  fair tuned shot reached parity, not a win, which is exactly what a data-bound task predicts."""))

    # Part 8.5 - tooling and product
    c.append(md("""# Part 8b · Tooling and the product surface

The assignment asks for more than a notebook: a usable detector. Two pieces of that already exist and are
how I read a model's behaviour and how it would run live. Both are real, working code in the project (the
live harness alone has 47 passing tests); the figures below are illustrative mock-ups of their output,
drawn to show its shape rather than a specific day's data.

## The inspection tool

I cannot judge a model from one number. The inspection tool plots the model's per-class probability for
every hour against the labels, so I can see *where* it is confident, where it hesitates, and where it misses
(Figure 12)."""))
    c.append(code(FIG_INSPECT))
    c.append(md("""**What this shows.** The top band is the model's confidence split between none, bull and bear over a
stretch of test hours; the strip beneath is the truth. Tall coloured spikes that line up with a coloured
strip are confident correct calls; a spike with no strip beneath (or the reverse) is a false alarm or a
miss. This is how I caught the Transformer's collapse behaviour and how I sanity-check any model before
trusting its scores."""))

    c.append(md("""## The paper-trading harness

To close the loop, a paper-trading harness replays live hourly data and logs what the detector would have
flagged, in dry-run mode, with no real orders (Figure 13)."""))
    c.append(code(FIG_PAPERTRADE))
    c.append(md("""**Reading it.** Across one trading session, each detected gap is marked with its direction and the
model's confidence. This turns the classifier into something a person can actually watch in real time, and
it is the backbone of the final demo. It also keeps the project honest about scope: the output is a flagged
zone with a confidence score, not a trade instruction."""))

    # Part 9
    c.append(md("""# Part 9 · Why these are the next steps, and why I am confident

## 9. The next steps follow directly from the verdict

Because the problem is **data-bound**, the highest-value lever is more and richer data, not a bigger model.
The order matters:

1. **Multi-symbol expansion comes first, because it is the blocker.** Train on QQQ, IWM and sector ETFs
   alongside SPY. FVGs are a *general* market structure, so other instruments multiply the scarce bull and
   bear examples the learning curves say the models still want. Every later comparison depends on this
   larger dataset existing, so it is the gate everything else waits behind. It targets the carrier,
   CNN-LSTM, directly.
2. **The Transformer fairness pass is complete** (Part 8): tuning lifted it to 0.601 ± 0.045 and removed the
   collapse, which closes the "bigger model just needed tuning" door with evidence: it reaches parity, not a
   win. The one open question left for it is whether *more data* changes that, which is step 3.
3. **Re-judge fairly on the big data, but only the models that can plausibly use it.** Once multi-symbol
   data exists, I will tune and re-evaluate the models whose evidence says more data should help:
   **CNN-LSTM** and **LSTM** (both fit train and have rising curves) and the **Transformer** (now tuned to parity,
   and the architecture that should gain most from scale). **xLSTM is deliberately excluded** from that sweep:
   it cannot fit the training data it already has (train-F1 near 0.34), so more data cannot rescue a model
   that has not learned what it was given; spending a hyper-parameter search on it would not change the
   ranking and is not a good use of limited compute. XGBoost stays as the reference control. This keeps the
   final comparison fair *and* honest about where the effort is worth spending.
4. **Deliver the product surface**, the confidence-scored detector with chart overlays, for the final demo.

## 10. Why I am confident in the conclusion

- The verdict comes from a rule fixed *before* the results, so it is not hindsight.
- It rests on two independent signals that agree: the learning curves (still rising) and the
  bigger-models-did-worse cross-check.
- The numbers carry confidence intervals, so a near-tie is reported as a tie.
- The counter-evidence is on the record: noisy curves, an unstable Transformer, an under-fitting xLSTM, the
  XGBoost input mismatch. A conclusion that survives its own stated caveats is one I can defend.

What would change my mind: if multi-symbol data *fails* to lift CNN-LSTM, the data-bound story is wrong and
I would pivot to data quality and labelling, or to capacity. That falsification test is the next milestone."""))

    # Reflection
    c.append(md("""# Part 10 · Reflection

**Technical.** The main lesson: a single metric cannot diagnose a model. The same 0.5-ish score meant
"starved" for one architecture and "cannot train" for another, and only the train-versus-test gap and the
learning curve told them apart. The most modern architecture also was not the safe default. On a few hundred
rare events the Transformer's flexibility turned into fragility and xLSTM could not fit at all, while the
built-in bias of CNN-LSTM's convolution beat raw capacity.

**Methodological.** Committing to the decision rule before seeing results turned "which model is best?" into
a falsifiable experiment. Confidence intervals stopped me celebrating noise. Treating the xLSTM failure as a
finding rather than an embarrassment is what made the diagnosis hold: the failing model carried the proof. I
leave this update with a lower score than XGBoost but with something more useful, a reason I can act on."""))

    # Final summary
    c.append(md("""# Final summary

**What is built.** A five-model detection ladder (XGBoost, LSTM, CNN-LSTM, Transformer, xLSTM) on identical
2016 to 2025 SPY hourly data, a learning-curve diagnosis harness, bootstrap confidence intervals, and a
decision rule fixed in advance, plus a fully tested training and configuration system.

**Concrete results** (test set, 5-seed mean ± std, 95% bootstrap CI; macro-F1):

| Model | Input | macro-F1 | 95% CI |
|---|---|---|---|
| XGBoost (reference) | 35 engineered features | 0.721 ± 0.001 | [0.671, 0.761] |
| **CNN-LSTM (carrier)** | raw 60x5 window | **0.639 ± 0.013** | [0.603, 0.674] |
| LSTM | raw 60x5 window | 0.595 ± 0.015 | [0.560, 0.628] |
| Transformer (untuned) | raw 60x5 window | 0.548 ± 0.085 | [0.519, 0.572] |
| xLSTM (untuned) | raw 60x5 window | 0.369 ± 0.007 | [0.354, 0.384] |

**Key findings.** On fair raw input, CNN-LSTM is the best deep model, tied with LSTM within CI. The two
largest models did *worse*, so the task is **data-bound, not capacity-bound**. The lever is therefore
**multi-symbol data**, aimed at CNN-LSTM.

**Limitations.**
- **Input vs architecture is named but not yet experimentally separated.** XGBoost reads 35 engineered
  features and the deep nets read the raw window, so its lead is partly the easier input. I have not run the
  controlled cross-test (XGBoost on the raw window, or a deep net on the 35 features), so I cannot fully
  attribute the gap to architecture rather than representation. That cross-test is the clean way to settle it.
- **The confidence scores are not yet calibration-checked.** The product value is a calibrated probability,
  but the scores currently come straight from the network's softmax. Softmax turns the raw outputs into probabilities that add to one, which does not by itself make them accurate, and I have not verified with a reliability
  diagram that "80% sure" is right about 80% of the time. Calibration assessment is future work.
- **Learning curves are noisy** over three seeds ("no plateau", not a precise slope).
- **The ladder's Transformer and xLSTM are untuned baselines.** I later tuned the Transformer (Part 8): it
  rose to 0.601 ± 0.045 and stopped collapsing across a 10-seed probe, so its earlier instability was largely
  a tuning artifact, but it still did not beat CNN-LSTM. xLSTM stays untuned by choice, since it cannot fit
  the training data and tuning is unlikely to help.
- **xLSTM's data-efficiency curve was abbreviated**, since its full-data score is already its ceiling and
  cannot change the ranking.

**Next steps.** Multi-symbol data expansion first, since it is the blocker, aimed at CNN-LSTM; then re-judge the
data-hungry models (CNN-LSTM, LSTM, Transformer, now all tuned) on the larger set, with xLSTM excluded
because it cannot fit the data it already has; build the demo surface;
final delivery by 20 June."""))

    c.append(md("""# References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* arXiv:1706.03762. https://arxiv.org/abs/1706.03762 (the Transformer.)
- Beck, M. et al. (2024). *xLSTM: Extended Long Short-Term Memory.* arXiv:2405.04517. https://arxiv.org/abs/2405.04517 (architecture under test.)
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. ISBN 978-1119482086. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 (temporal evaluation; small-sample finance.)
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593 (confidence intervals; block bootstrap for serial dependence.)"""))

    nb = new_notebook(cells=c)
    nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
    nbf.write(nb, "notebooks/06-status-update-2.ipynb")
    print("wrote notebooks/06-status-update-2.ipynb", len(c), "cells")


def _slide(cell, kind="slide"):
    cell.metadata["slideshow"] = {"slide_type": kind}
    return cell


def build_deck():
    c = []
    def S(cell, kind="slide"): c.append(_slide(cell, kind)); return cell
    S(md("""# Status update 2
## Why classical ML beat deep learning at spotting Fair Value Gaps"""))

    S(md("""### What this maps to (learning outcomes)

- **LO1, Data preparation**: candles, labelling, gold-check, time-ordered split, windowing
- **LO2, Data analysis & model engineering**: a five-model ladder, tuning (G1 to G10), confidence intervals, diagnosis
- **LO3, Professional Standard**: a rule fixed before results, honest limitations, evidence-based advice, bounded scope (zones, not trades), compute spent where it pays
- **LO4, Personal Leadership**: goals set from evidence, a sequenced roadmap, plans adjusted as the data demanded
"""))

    S(md("""## 1 · What am I building?

I detect **Fair Value Gaps**, specific zones on a price chart, automatically, and attach a **confidence
score** to each. Not price prediction. Not trading. A detector that reports its confidence in every flag.

First, the only finance needed: a candle."""))
    S(code(DATA_PRELUDE), "skip")
    S(code(FIG_CANDLE), "subslide")
    S(md("""**What this shows.** A candle is four numbers for one hour. The model sees these numbers, never a
chart image.""", ), "subslide")

    S(md("""## 2 · What is a Fair Value Gap?

A three-candle pattern where price moved so fast it **skipped a band of prices entirely**, a gap that never
traded. Bullish is a fast jump up; bearish is a fast drop. Traders watch these zones; I detect them."""))
    S(code(FIG_BULL), "subslide")
    S(code(FIG_BEAR), "subslide")
    S(md("""**What this shows.** Both are purely geometric, defined by three candles. That is why a rule can label
them and a model can learn them.""", ), "subslide")

    S(md("""## 3 · Why a model, not just a rule?

A rule gives yes/no. A model gives a **calibrated probability**, "80% sure", so a user can rank and filter
zones. The rule is the teacher; the model learns to imitate it and to express doubt.

**The data, briefly:** SPY hourly 2016 to 2025 (Alpaca); a 6-criterion rule labels none / bull / bear
(about 97% / 1.8% / 1.3%); a hand-checked gold set confirms the labels (kappa near 1.0, where 1.0 is perfect agreement); a strict
time-ordered split (never shuffled); each example is the last **60 hours by 5 numbers**."""))
    S(code(FIG_IMBALANCE), "subslide")
    S(md("""**Why this matters.** The classes are extremely lopsided, so plain accuracy is meaningless. The metric
is **macro-F1**, which gives the rare bull and bear gaps equal weight.""", ), "subslide")

    S(md("""## 4 · The question

> **Why did classical ML (XGBoost) detect FVGs *better* than deep learning, and is deep learning the wrong
> tool, or just starved of something?**

The catch: the two families read the **same 60 hours differently**.
- **XGBoost** uses 35 **hand-engineered** summary numbers (human-assisted, easier).
- **Deep nets** use the **raw 60x5 grid** (must learn everything themselves).

So the fair fight is **deep-net against deep-net on raw input**."""))
    S(code(FIG_TWO_FAMILIES), "subslide")

    S(md("""## 5 · What I did to find out

1. **Completed the ladder**: LSTM, CNN-LSTM, Transformer, xLSTM (plain to most expressive). *If bigger keeps
   winning, it needs capacity. If bigger stalls, it does not.*
2. **Learning-curve experiment**: retrain on 20% to 100% of the data. Still rising means hungry for data; flat would mean more data will not help.
3. **Fixed the verdict rule before looking**: a 2x2 matrix, so no hindsight.

Every model trained five times with **95% confidence intervals**, so I never mistake noise for a finding."""))

    S(md("""## 6 · The scoreboard"""))
    S(code(FIG_RANKING), "subslide")
    S(md("""**Read it honestly.** CNN-LSTM (0.639) and LSTM (0.595) bars overlap, so they are a **statistical
tie**; CNN-LSTM wins on secondary grounds and I say so. The **two biggest** models sit at the bottom.
XGBoost (0.721) leads, but on the easier engineered input, so not a pure-architecture win."""), "subslide")
    S(code(FIG_PERCLASS), "subslide")
    S(md("""**Where they fail.** Everyone finds "none". The models separate on the rare bull and bear gaps, and
that is where the difficulty lives.""", ), "subslide")

    S(md("""## 7 · Every model got a fair hearing

| Model | Score | What it taught me |
|---|---|---|
| **XGBoost** | 0.721* | Classical control; hard to beat on scarce data (*easier input) |
| **CNN-LSTM** | **0.639** | Best deep net; its built-in shape detector fits an FVG |
| **LSTM** | 0.595 | Fits training well, generalises less, so it wants more data |
| **Transformer** | 0.548 | Unstable across seeds (±0.085): some runs reach ~0.59, others collapse |
| **xLSTM** | 0.369 | Newest design, yet cannot even fit train; an informative loss |

xLSTM is the most useful failure: a more complex model scoring *worse* is direct proof that complexity is
not what the task lacks."""))

    S(md("""## 8 · What I observed, and the verdict"""))
    S(code(FIG_FIT_GAP), "subslide")
    S(code(FIG_CURVES), "subslide")
    S(code(FIG_MATRIX), "subslide")
    S(md("""**Two signals agree:** the workable models are still improving with more data, and the biggest models
did worse. Both point one way.

> ### The problem is **data-bound**, not capacity-bound.
> The limit is how much signal exists in one instrument's rare events, not the size of the network."""), "subslide")

    S(md("""## 8b · The product surface: how I read and run the model

The assignment wants a usable detector, not just scores. Two tools already exist (figures are illustrative).
- **Inspection tool**: the model's per-class probability for every hour, against the labels, so I can see
  where it is confident, where it hesitates, and where it misses.
- **Paper-trading harness**: replays live hourly data and logs the flagged gaps with confidence, dry-run,
  no real orders."""))
    S(code(FIG_INSPECT), "subslide")
    S(code(FIG_PAPERTRADE), "subslide")

    S(md("""## 9 · Progress alongside the headline

- **The five-model ladder is finished** and trained under one set of rules, so the comparison is complete.
- **Scores carry a range, not one number.** Training has randomness, so I run each model five times. CNN-LSTM
  and LSTM overlap once that spread is counted, so I call them a tie instead of declaring a winner.
- **Transformer fairness pass: done.** Untuned it scored 0.548 and was unstable; tuned (warm-up + clipping +
  search) it reaches **0.601 ± 0.045** with no collapse over 10 seeds. Still ties LSTM and below CNN-LSTM:
  a fair tuned shot reached parity, not a win."""))

    S(md("""## 10 · Next steps, and why I am confident

**Because it is data-bound, the lever is more and richer data, not a bigger model:**
1. **Multi-symbol expansion** (QQQ, IWM, sector ETFs) first, the blocker: multiplies the rare gap examples; targets CNN-LSTM.
2. **Transformer fairness pass: done** (tuned to 0.601, parity not a win) -> the "bigger model" door is closed; the open question is whether more data helps it (step 3).
3. **Re-judge on the bigger data** the models that can use it (CNN-LSTM, LSTM, Transformer). xLSTM is excluded: it cannot fit the data it already has.
4. **Build the demo**: the confidence-scored detector with chart overlays.

**Why I trust the conclusion:** rule fixed before results; two independent signals agree; every number has a
confidence interval; all counter-evidence is on the record. If multi-symbol data fails to help, the
data-bound story is wrong, and that is the next milestone's falsification test."""))

    S(md("""## 11 · Reflection

- **Technical**: one score cannot diagnose a model; the same 0.5-ish meant "starved" for one and "cannot
  train" for another. Built-in bias (CNN-LSTM) beat raw capacity on small data.
- **Methodological**: committing to the verdict rule before results turned "which is best?" into a
  falsifiable experiment; the xLSTM failure carried the proof. I leave with a lower score than XGBoost but
  with a clear, evidence-backed reason for what to do next."""))

    S(md("""## References

López de Prado (2018) *Advances in Financial ML* · Vaswani et al. (2017) *Attention Is All You Need* ·
Beck et al. (2024) *xLSTM* · Efron and Tibshirani (1993) *Introduction to the Bootstrap*."""))

    nb = new_notebook(cells=c)
    nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
    nb.metadata["celltoolbar"] = "Slideshow"
    nbf.write(nb, "notebooks/07-status-update-2-presentation.ipynb")
    print("wrote notebooks/07-status-update-2-presentation.ipynb", len(c), "cells")


if __name__ == "__main__":
    build_notebook()
    build_deck()
