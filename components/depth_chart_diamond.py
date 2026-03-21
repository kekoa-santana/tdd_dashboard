"""Baseball diamond depth chart using Plotly with headshot images."""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from config import GOLD, CREAM, SLATE, DARK

# Field coordinates for each position (x, y on a 0-100 scale)
_POSITION_COORDS = {
    "C":  (50, 8),
    "1B": (72, 32),
    "2B": (62, 48),
    "SS": (38, 48),
    "3B": (28, 32),
    "LF": (18, 72),
    "CF": (50, 82),
    "RF": (82, 72),
    "DH": (6, 8),
}

_HEADSHOT_SIZE = 45


def _headshot_url(player_id: int, size: int = 60) -> str:
    return (
        f"https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/"
        f"w_{size},q_auto:best/v1/people/{player_id}/headshot/67/current"
    )


def render_diamond_chart(
    starters: pd.DataFrame,
    team_abbr: str,
    *,
    height: int = 480,
) -> None:
    """Render a baseball diamond with player headshots at each position.

    Parameters
    ----------
    starters : pd.DataFrame
        Probable starters for one team.  Expected columns:
        position, player_id, player_name, batting_order.
    team_abbr : str
        Team abbreviation for the title.
    height : int
        Chart height in pixels.
    """
    if starters.empty:
        st.info("No probable starters data available.")
        return

    fig = go.Figure()

    # ── Draw the field ────────────────────────────────────────────
    # Infield diamond
    diamond_x = [50, 72, 50, 28, 50]
    diamond_y = [12, 32, 52, 32, 12]
    fig.add_trace(go.Scatter(
        x=diamond_x, y=diamond_y,
        mode="lines",
        line=dict(color=f"rgba(123,143,166,0.33)", width=1.5),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Outfield arc
    import numpy as np
    arc_angles = np.linspace(-0.15, np.pi + 0.15, 60)
    arc_x = 50 + 42 * np.cos(arc_angles)
    arc_y = 32 + 42 * np.sin(arc_angles)
    fig.add_trace(go.Scatter(
        x=arc_x.tolist(), y=arc_y.tolist(),
        mode="lines",
        line=dict(color=f"rgba(123,143,166,0.2)", width=1, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Foul lines
    fig.add_trace(go.Scatter(
        x=[50, 5], y=[12, 75],
        mode="lines",
        line=dict(color=f"rgba(123,143,166,0.2)", width=1),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[50, 95], y=[12, 75],
        mode="lines",
        line=dict(color=f"rgba(123,143,166,0.2)", width=1),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Place players ─────────────────────────────────────────────
    starter_lookup = {}
    for _, row in starters.iterrows():
        starter_lookup[row["position"]] = row

    # Position markers + name labels
    marker_x, marker_y, texts, hovers = [], [], [], []
    for pos, (px, py) in _POSITION_COORDS.items():
        if pos in starter_lookup:
            row = starter_lookup[pos]
            name = row["player_name"]
            # Shorten to last name for display
            last_name = name.split()[-1] if " " in name else name
            bo = int(row["batting_order"])
            label = f"{pos}"
            display = f"{last_name}"
            hover = (
                f"<b>{name}</b><br>"
                f"Position: {pos}<br>"
                f"Batting Order: {bo}<br>"
                f"2025 Starts: {int(row.get('starts_at_position', 0))}"
            )
        else:
            label = pos
            display = "—"
            hover = f"{pos}: No starter assigned"

        marker_x.append(px)
        marker_y.append(py)
        texts.append(f"<b>{label}</b><br>{display}")
        hovers.append(hover)

    # Add player markers
    fig.add_trace(go.Scatter(
        x=marker_x, y=marker_y,
        mode="markers+text",
        marker=dict(
            size=50,
            color=f"{DARK}",
            line=dict(color=f"rgba(200,169,110,0.53)", width=2),
            symbol="circle",
        ),
        text=texts,
        textfont=dict(color=CREAM, size=10),
        textposition="bottom center",
        hovertext=hovers,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor=DARK,
            bordercolor=GOLD,
            font=dict(color=CREAM, size=12),
        ),
        showlegend=False,
    ))

    # ── Add headshot images ───────────────────────────────────────
    for pos, (px, py) in _POSITION_COORDS.items():
        if pos not in starter_lookup:
            continue
        row = starter_lookup[pos]
        pid = int(row["player_id"])
        url = _headshot_url(pid, _HEADSHOT_SIZE)

        # Convert data coords to fraction of plot area
        # x: data range ~0-100, y: data range ~0-100
        x_frac = (px - (-5)) / 110  # account for axis range
        y_frac = (py - (-5)) / 105

        # Image size as fraction of plot
        img_size = 0.08

        fig.add_layout_image(
            dict(
                source=url,
                x=x_frac,
                y=y_frac,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="middle",
                sizex=img_size,
                sizey=img_size,
                layer="above",
            )
        )

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            range=[-5, 105],
            visible=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[-5, 100],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
        ),
        dragmode=False,
    )

    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
