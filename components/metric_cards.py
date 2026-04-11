"""Metric card and percentile bar HTML components."""
from __future__ import annotations

import pandas as pd

from config import GOLD, EMBER, SAGE, SLATE


def pctile_color(pctile: float) -> str:
    """Color for percentile bar fill."""
    if pctile >= 80:
        return SAGE
    elif pctile >= 60:
        return GOLD
    elif pctile >= 40:
        return SLATE
    elif pctile >= 20:
        return EMBER
    else:
        return EMBER


def percentile_rank(
    series: pd.Series, value: float, higher_is_better: bool,
) -> float:
    """Compute percentile rank (0-100) of value within series."""
    valid = series.dropna()
    if len(valid) == 0:
        return 50.0
    if higher_is_better:
        return float((valid < value).sum() / len(valid) * 100)
    else:
        return float((valid > value).sum() / len(valid) * 100)


def metric_card(label: str, value: str, delta_html: str = "", pctile: float | None = None) -> str:
    """Render a styled metric card with optional delta and percentile badge."""
    delta_div = f'<div class="metric-delta">{delta_html}</div>' if delta_html else ""
    if pctile is not None:
        pct_color = pctile_color(pctile)
        pctile_div = f'<div class="metric-pctile" style="color:{pct_color};">{pctile:.0f}th pctile</div>'
    else:
        pctile_div = ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{pctile_div}'
        f'{delta_div}'
        f'</div>'
    )

