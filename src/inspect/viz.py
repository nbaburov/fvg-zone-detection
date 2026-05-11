"""viz.py — Plotly visualisations for inspection (per-window + per-model timeline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.inspect.outcomes import Trade

# Colour scheme for classes
_CLASS_COLOURS = {
    0: "#888888",   # none — grey
    1: "#00cc44",   # bullish — green
    2: "#ff3333",   # bearish — red
}
_CLASS_LABELS = {0: "none", 1: "bullish", 2: "bearish"}


def plot_window(
    raw_ohlcv: np.ndarray,
    timestamp: pd.Timestamp,
    gold_label: int,
    model_preds: dict[str, tuple[int, float]],
    output_path: Path,
) -> None:
    """Render a single 60-bar OHLCV window to a standalone HTML file.

    Parameters
    ----------
    raw_ohlcv : np.ndarray, shape (60, 5)
        Raw un-normalised OHLCV bars. Columns: [open, high, low, close, volume].
    timestamp : pd.Timestamp
        Timestamp of the window's last bar (used in chart title).
    gold_label : int
        Ground-truth encoded label {0, 1, 2}.
    model_preds : dict[str, tuple[int, float]]
        model_name -> (predicted_class, confidence).
        Confidence = max probability from predict_proba.
    output_path : Path
        Destination HTML file path. Parent directory is created if absent.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required for visualisation.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opens = raw_ohlcv[:, 0]
    highs = raw_ohlcv[:, 1]
    lows = raw_ohlcv[:, 2]
    closes = raw_ohlcv[:, 3]
    volumes = raw_ohlcv[:, 4]

    bar_indices = list(range(60))

    n_subplots = 2  # candlestick + volume
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=bar_indices,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="OHLC",
            increasing_line_color="#00cc44",
            decreasing_line_color="#ff3333",
        ),
        row=1,
        col=1,
    )

    # Gold label annotation on bar 59 (last bar)
    gold_colour = _CLASS_COLOURS[gold_label]
    gold_name = _CLASS_LABELS[gold_label]
    fig.add_annotation(
        x=59,
        y=float(highs[59]),
        text=f"GT: {gold_name}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=gold_colour,
        font={"color": gold_colour, "size": 11, "family": "monospace"},
        ax=20,
        ay=-30,
        row=1,
        col=1,
    )

    # Per-model prediction annotations (staggered slightly below gold)
    for offset, (model_name, (pred_cls, confidence)) in enumerate(model_preds.items()):
        pred_colour = _CLASS_COLOURS[pred_cls]
        pred_name = _CLASS_LABELS[pred_cls]
        fig.add_annotation(
            x=59,
            y=float(lows[59]),
            text=f"{model_name}: {pred_name} ({confidence:.2f})",
            showarrow=True,
            arrowhead=1,
            arrowcolor=pred_colour,
            font={"color": pred_colour, "size": 10, "family": "monospace"},
            ax=-(30 + offset * 15),
            ay=30 + offset * 20,
            row=1,
            col=1,
        )

    # Volume bar chart
    vol_colours = [
        "#00cc44" if closes[i] >= opens[i] else "#ff3333"
        for i in range(60)
    ]
    fig.add_trace(
        go.Bar(
            x=bar_indices,
            y=volumes,
            marker_color=vol_colours,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Title
    models_str = " | ".join(
        f"{m}: {_CLASS_LABELS[p[0]]} ({p[1]:.2f})"
        for m, p in model_preds.items()
    )
    title = (
        f"Window ending {timestamp.strftime('%Y-%m-%d %H:%M')} — "
        f"GT: {gold_name} | {models_str}"
    )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=520,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    fig.write_html(str(output_path), include_plotlyjs="cdn")


def plot_model_timeline(
    df: pd.DataFrame,
    trades: list[Trade],
    model_name: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Render the full data slice as one candlestick chart with this model's trades overlaid.

    Markers per trade:
    - Triangle-up (green) at entry for bull / triangle-down (red) for bear.
    - Circle at exit, coloured by outcome: green=TP, red=SL, grey=undecided.
    - Dashed line from entry → exit (faint).

    Hover shows entry/SL/TP/outcome/R-multiple.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for visualisation.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="SPY H1",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            opacity=0.6,
        )
    )

    OUTCOME_COLOUR = {"tp": "#00cc44", "sl": "#ff3333", "undecided": "#aaaaaa", "no_future": "#666666"}

    bull_x, bull_y, bull_hover = [], [], []
    bear_x, bear_y, bear_hover = [], [], []
    exit_x, exit_y, exit_colour, exit_hover = [], [], [], []
    line_shapes = []

    for t in trades:
        if pd.isna(t.entry_ts) or pd.isna(t.entry):
            continue
        if t.direction == 1:
            bull_x.append(t.entry_ts); bull_y.append(t.entry)
            bull_hover.append(_trade_hover(t))
        else:
            bear_x.append(t.entry_ts); bear_y.append(t.entry)
            bear_hover.append(_trade_hover(t))

        if t.exit_ts is not None and not pd.isna(t.exit_ts):
            exit_x.append(t.exit_ts); exit_y.append(t.exit_price)
            exit_colour.append(OUTCOME_COLOUR.get(t.outcome, "#888"))
            exit_hover.append(f"Exit: {t.outcome.upper()} @ {t.exit_price:.2f} | R={t.r_multiple:+.2f}")
            line_shapes.append({
                "type": "line", "xref": "x", "yref": "y",
                "x0": t.entry_ts, "x1": t.exit_ts,
                "y0": t.entry, "y1": t.exit_price,
                "line": {"color": OUTCOME_COLOUR.get(t.outcome, "#888"), "width": 1, "dash": "dot"},
                "opacity": 0.5,
            })

    if bull_x:
        fig.add_trace(go.Scatter(
            x=bull_x, y=bull_y, mode="markers",
            marker={"symbol": "triangle-up", "size": 12, "color": "#00cc44",
                    "line": {"color": "#003311", "width": 1}},
            name="Bull entry", hovertext=bull_hover, hoverinfo="text+x",
        ))
    if bear_x:
        fig.add_trace(go.Scatter(
            x=bear_x, y=bear_y, mode="markers",
            marker={"symbol": "triangle-down", "size": 12, "color": "#ff3333",
                    "line": {"color": "#330000", "width": 1}},
            name="Bear entry", hovertext=bear_hover, hoverinfo="text+x",
        ))
    if exit_x:
        fig.add_trace(go.Scatter(
            x=exit_x, y=exit_y, mode="markers",
            marker={"symbol": "circle", "size": 10, "color": exit_colour,
                    "line": {"color": "#000", "width": 1}},
            name="Exit", hovertext=exit_hover, hoverinfo="text+x",
        ))

    fig.update_layout(
        title=f"{model_name} — trades over full slice {title_suffix}".strip(),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=720,
        shapes=line_shapes,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def _trade_hover(t: Trade) -> str:
    direction_str = "BULL" if t.direction == 1 else "BEAR"
    return (
        f"{direction_str} entry @ {t.entry:.2f}<br>"
        f"SL {t.sl:.2f} | TP {t.tp:.2f}<br>"
        f"Outcome: {t.outcome.upper()} | R={t.r_multiple:+.2f}"
    )
