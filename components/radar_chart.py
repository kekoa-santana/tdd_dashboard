"""Radar chart component — 6-axis team profile visualization.

Renders a matplotlib polar plot as a base64 PNG for HTML embedding.
"""
from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from config import GOLD, SLATE, CREAM, DARK


def _fig_to_b64_img(fig: Figure, width_css: str = "100%") -> str:
    """Convert a matplotlib Figure to a base64 <img> tag."""
    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=100, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f'<img src="data:image/png;base64,{b64}" style="width:{width_css}; height:auto;" />'


def radar_chart_html(
    scores: dict[str, float],
    size: int = 200,
) -> str:
    """Build a 6-axis radar chart and return it as a base64 ``<img>`` tag.

    Parameters
    ----------
    scores : dict[str, float]
        Axis label -> 0-1 score.  Order of keys determines axis order.
    size : int
        Target display width in pixels.

    Returns
    -------
    str
        HTML ``<img>`` tag with embedded base64 PNG.
    """
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    n = len(labels)

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(3, 3), facecolor=DARK)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(DARK)

    # Draw filled area + outline
    ax.fill(angles_closed, values_closed, color=GOLD, alpha=0.25)
    ax.plot(angles_closed, values_closed, color=GOLD, linewidth=1.5, alpha=0.85)
    ax.scatter(angles, values, color=GOLD, s=18, zorder=5, alpha=0.9)

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color=CREAM, fontsize=7, fontweight=600)

    # Radial grid
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=SLATE, alpha=0.2, linewidth=0.5)
    ax.xaxis.grid(True, color=SLATE, alpha=0.15, linewidth=0.5)
    ax.spines["polar"].set_visible(False)

    # Tight layout
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    html = _fig_to_b64_img(fig, width_css=f"{size}px")
    plt.close(fig)
    return html
