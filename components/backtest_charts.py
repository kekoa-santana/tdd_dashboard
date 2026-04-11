"""Backtest and in-season accuracy charts for the Model Performance page.

All charts use plotly with transparent backgrounds to match the dashboard theme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import GOLD, EMBER, SAGE, SLATE, CREAM

# Shared plotly config passed to st.plotly_chart()
PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False}

# Common layout defaults
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    font=dict(color=CREAM),
    dragmode=False,
)

_AXIS_STYLE = dict(
    tickfont=dict(color=SLATE, size=9),
    showgrid=False,
    zeroline=False,
    showline=False,
)

_GRID_AXIS = dict(
    tickfont=dict(color=SLATE, size=9),
    showgrid=True,
    gridcolor="rgba(123,143,166,0.12)",
    gridwidth=0.5,
    zeroline=False,
    showline=False,
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Backtest charts
# ---------------------------------------------------------------------------


def create_coverage_chart(
    df: pd.DataFrame,
    coverage_cols: list[str],
    labels: list[str],
) -> go.Figure:
    """Horizontal bars showing CI coverage vs expected targets."""
    avgs = [df[col].mean() for col in coverage_cols]
    targets = []
    for lbl in labels:
        if "95" in lbl:
            targets.append(0.95)
        elif "80" in lbl:
            targets.append(0.80)
        elif "90" in lbl:
            targets.append(0.90)
        elif "50" in lbl:
            targets.append(0.50)
        else:
            targets.append(0.95)

    colors = [SAGE if avg >= tgt * 0.95 else EMBER for avg, tgt in zip(avgs, targets)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[a * 100 for a in avgs],
        orientation="h",
        marker_color=colors, opacity=0.85,
        text=[f"{a*100:.1f}%" for a in avgs],
        textposition="outside", textfont=dict(color=CREAM, size=10),
    ))

    # Target lines
    for i, tgt in enumerate(targets):
        fig.add_shape(
            type="line",
            x0=tgt * 100, x1=tgt * 100,
            y0=i - 0.4, y1=i + 0.4,
            line=dict(color=GOLD, width=2, dash="dash"),
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=max(180, len(labels) * 55),
        title=dict(text="Interval Coverage (dashed = target)",
                   font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=dict(**_AXIS_STYLE, range=[0, 108],
                   title=dict(text="Coverage %", font=dict(color=SLATE, size=10))),
        yaxis=dict(**_AXIS_STYLE, autorange="reversed"),
    )
    return fig


def create_game_k_model_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped bars comparing 4 model tiers by Brier score."""
    tier_cols = [
        ("naive_avg_brier", "Naive"),
        ("poisson_avg_brier", "Poisson"),
        ("model_no_matchup_avg_brier", "No Matchup"),
        ("full_model_avg_brier", "Full Model"),
    ]
    tier_colors = [SLATE, EMBER, GOLD, SAGE]
    seasons = df["test_season"].astype(str).tolist()

    fig = go.Figure()
    for (col, label), color in zip(tier_cols, tier_colors):
        if col not in df.columns:
            continue
        vals = df[col].tolist()
        fig.add_trace(go.Bar(
            name=label, x=seasons, y=vals,
            marker_color=color, opacity=0.85,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(color=color, size=9),
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="group",
        height=300,
        showlegend=True,
        legend=dict(font=dict(color=CREAM, size=9), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Game K Model Comparison by Brier Score",
                   font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=_AXIS_STYLE,
        yaxis=dict(**_GRID_AXIS,
                   title=dict(text="Avg Brier Score", font=dict(color=SLATE, size=10))),
    )
    return fig


def create_movers_chart(
    names: list[str],
    deltas: list[float],
    title: str,
    positive_color: str = SAGE,
    negative_color: str = EMBER,
) -> go.Figure:
    """Horizontal bar chart showing biggest projection movers."""
    colors = [positive_color if d >= 0 else negative_color for d in deltas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=deltas,
        orientation="h",
        marker_color=colors, opacity=0.85,
        text=[f"{d:+.1f}pp" for d in deltas],
        textposition="outside", textfont=dict(color=CREAM, size=10),
    ))

    fig.add_vline(x=0, line_color=SLATE, line_width=0.5, opacity=0.5)

    fig.update_layout(
        **_BASE_LAYOUT,
        height=max(200, len(names) * 32),
        title=dict(text=title, font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=dict(**_AXIS_STYLE,
                   title=dict(text="Change (pp)", font=dict(color=SLATE, size=10))),
        yaxis=dict(**_AXIS_STYLE, autorange="reversed"),
    )
    return fig


def create_projection_timeline(
    snapshots: dict[str, pd.DataFrame],
    player_id: int,
    id_col: str,
    name_col: str,
    stat_cols: list[str],
) -> go.Figure | None:
    """Line chart of projected rate over time with CI band."""
    dates = sorted(snapshots.keys())
    if not dates:
        return None

    stat_data: dict[str, list[float]] = {s: [] for s in stat_cols}
    valid_dates: list[str] = []
    player_name = ""

    for d in dates:
        df = snapshots[d]
        row = df[df[id_col] == player_id]
        if row.empty:
            continue
        valid_dates.append(d)
        if not player_name and name_col in row.columns:
            player_name = str(row.iloc[0][name_col])
        for s in stat_cols:
            stat_data[s].append(float(row.iloc[0].get(s, 0)))

    if not valid_dates:
        return None

    colors = [SAGE, GOLD, EMBER, SLATE]
    fig = go.Figure()

    for i, s in enumerate(stat_cols):
        color = colors[i % len(colors)]
        vals = [v * 100 for v in stat_data[s]]
        label = s.replace("projected_", "")

        fig.add_trace(go.Scatter(
            x=valid_dates, y=vals,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            name=label,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))

        # CI band
        lo_col = s + "_2_5"
        hi_col = s + "_97_5"
        lo_vals, hi_vals = [], []
        for d in valid_dates:
            row = snapshots[d][snapshots[d][id_col] == player_id]
            if not row.empty and lo_col in row.columns and hi_col in row.columns:
                lo_vals.append(float(row.iloc[0][lo_col]) * 100)
                hi_vals.append(float(row.iloc[0][hi_col]) * 100)

        if len(lo_vals) == len(valid_dates):
            fig.add_trace(go.Scatter(
                x=valid_dates + valid_dates[::-1],
                y=hi_vals + lo_vals[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.12),
                line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ))

    title = f"{player_name} | Projection Timeline" if player_name else "Projection Timeline"
    fig.update_layout(
        **_BASE_LAYOUT,
        height=280,
        showlegend=len(stat_cols) > 1,
        legend=dict(font=dict(color=CREAM, size=9), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=title, font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=dict(**_AXIS_STYLE, tickangle=-45),
        yaxis=dict(**_GRID_AXIS,
                   title=dict(text="Rate (%)", font=dict(color=SLATE, size=10))),
    )
    return fig


# ---------------------------------------------------------------------------
# In-season accuracy charts
# ---------------------------------------------------------------------------


def create_pred_vs_actual_scatter(
    predicted: pd.Series,
    actual: pd.Series,
    stat_label: str,
    has_lineup: pd.Series | None = None,
) -> go.Figure:
    """Scatter of predicted vs actual game-level stats with diagonal."""
    fig = go.Figure()

    if has_lineup is not None:
        mask_lu = has_lineup.astype(bool)
        if mask_lu.any():
            fig.add_trace(go.Scatter(
                x=predicted[mask_lu], y=actual[mask_lu],
                mode="markers",
                marker=dict(size=6, color=SAGE, opacity=0.5),
                name="With lineup",
                hovertemplate="Pred: %{x:.1f}, Actual: %{y:.0f}<extra></extra>",
            ))
        if (~mask_lu).any():
            fig.add_trace(go.Scatter(
                x=predicted[~mask_lu], y=actual[~mask_lu],
                mode="markers",
                marker=dict(size=6, color=SLATE, opacity=0.4),
                name="No lineup",
                hovertemplate="Pred: %{x:.1f}, Actual: %{y:.0f}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=predicted, y=actual,
            mode="markers",
            marker=dict(size=6, color=SAGE, opacity=0.5),
            hovertemplate="Pred: %{x:.1f}, Actual: %{y:.0f}<extra></extra>",
            showlegend=False,
        ))

    lo = min(predicted.min(), actual.min(), 0)
    hi = max(predicted.max(), actual.max()) + 1
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color=GOLD, width=1.5, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        height=300,
        showlegend=has_lineup is not None,
        legend=dict(font=dict(color=CREAM, size=9), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"Predicted vs Actual {stat_label}",
                   font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=dict(**_GRID_AXIS, range=[lo - 0.3, hi],
                   title=dict(text=f"Predicted {stat_label}", font=dict(color=SLATE, size=10))),
        yaxis=dict(**_GRID_AXIS, range=[lo - 0.3, hi],
                   title=dict(text=f"Actual {stat_label}", font=dict(color=SLATE, size=10)),
                   scaleanchor="x"),
    )
    return fig


def create_error_histogram(
    errors: pd.Series,
    stat_label: str,
) -> go.Figure:
    """Histogram of prediction errors (predicted - actual)."""
    fig = go.Figure()

    # Split into positive (over-predict) and negative (under-predict)
    pos = errors[errors >= 0]
    neg = errors[errors < 0]

    bin_size = max(0.5, (errors.max() - errors.min()) / 25)

    if not neg.empty:
        fig.add_trace(go.Histogram(
            x=neg, marker_color=EMBER, opacity=0.8,
            xbins=dict(size=bin_size),
            hovertemplate="Error: %{x:.1f}, Count: %{y}<extra></extra>",
            showlegend=False,
        ))
    if not pos.empty:
        fig.add_trace(go.Histogram(
            x=pos, marker_color=SAGE, opacity=0.8,
            xbins=dict(size=bin_size),
            hovertemplate="Error: %{x:.1f}, Count: %{y}<extra></extra>",
            showlegend=False,
        ))

    fig.add_vline(x=0, line_color=GOLD, line_width=1.5, line_dash="dash", opacity=0.8)

    mean_err = errors.mean()
    fig.add_vline(x=mean_err, line_color=CREAM, line_width=1, line_dash="dot", opacity=0.6)
    fig.add_annotation(
        x=mean_err, y=1.0, yref="paper",
        text=f"bias: {mean_err:+.2f}",
        showarrow=False, font=dict(color=CREAM, size=10),
        xanchor="left", xshift=6, yanchor="top",
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        height=300,
        title=dict(text=f"{stat_label} Error Distribution",
                   font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=dict(**_AXIS_STYLE,
                   title=dict(text=f"{stat_label} Error (predicted - actual)",
                              font=dict(color=SLATE, size=10))),
        yaxis=dict(**_GRID_AXIS,
                   title=dict(text="Count", font=dict(color=SLATE, size=10))),
    )
    return fig


def create_rolling_mae_chart(
    df: pd.DataFrame,
    date_col: str,
    stat_configs: list[tuple[str, str, str]],
    window: int = 7,
) -> go.Figure:
    """Rolling MAE line chart over time.

    stat_configs: list of (predicted_col, actual_col, label) tuples.
    """
    colors = [SAGE, GOLD, EMBER, SLATE]
    df = df.sort_values(date_col).copy()

    fig = go.Figure()
    for i, (pred_col, act_col, label) in enumerate(stat_configs):
        df[f"_ae_{label}"] = (df[pred_col] - df[act_col]).abs()
        rolling = df.groupby(date_col)[f"_ae_{label}"].mean().rolling(
            window, min_periods=1,
        ).mean()
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=rolling.index.astype(str), y=rolling.values,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{label} ({window}d)",
            hovertemplate="%{x}: MAE %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        height=280,
        showlegend=True,
        legend=dict(font=dict(color=CREAM, size=9), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"Prediction Accuracy Over Time ({window}-day rolling)",
                   font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=dict(**_AXIS_STYLE, tickangle=-45),
        yaxis=dict(**_GRID_AXIS,
                   title=dict(text="Rolling MAE", font=dict(color=SLATE, size=10))),
    )
    return fig


def create_prop_calibration_chart(
    predicted_probs: list[pd.Series],
    actual_overs: list[pd.Series],
    line_labels: list[str],
) -> go.Figure:
    """Calibration dots: predicted probability vs actual hit rate per prop line."""
    fig = go.Figure()

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=SLATE, width=1, dash="dash"),
        opacity=0.5, hoverinfo="skip", showlegend=False,
    ))

    colors = [SAGE, GOLD, EMBER, SLATE]
    for i, (probs, overs, label) in enumerate(
        zip(predicted_probs, actual_overs, line_labels)
    ):
        if len(probs) < 10:
            continue
        n_bins = min(10, max(3, len(probs) // 20))
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_actuals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() >= 3:
                bin_centers.append((lo + hi) / 2)
                bin_actuals.append(overs[mask].mean())

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=bin_centers, y=bin_actuals,
            mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(size=8, color=color),
            name=f"Over {label} (n={len(probs)})",
            hovertemplate="Predicted: %{x:.0%}, Actual: %{y:.0%}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        height=300,
        showlegend=True,
        legend=dict(font=dict(color=CREAM, size=9), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Prop Line Calibration",
                   font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        xaxis=dict(**_GRID_AXIS, range=[-0.02, 1.02],
                   title=dict(text="Predicted Probability", font=dict(color=SLATE, size=10))),
        yaxis=dict(**_GRID_AXIS, range=[-0.02, 1.02],
                   title=dict(text="Actual Frequency", font=dict(color=SLATE, size=10))),
    )
    return fig
