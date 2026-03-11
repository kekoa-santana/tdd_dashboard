"""Metric card and percentile bar HTML components."""
from __future__ import annotations

import pandas as pd

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK, DARK_CARD, DARK_BORDER, PRIOR_SEASON
from utils.formatters import fmt_stat


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
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {pctile_div}
        {delta_div}
    </div>
    """


def pctile_bar_html(
    label: str,
    pctile: float,
    ci_lo: float,
    ci_hi: float,
    key: str,
    pctile_prev: float | None = None,
) -> str:
    """Render a percentile bar row with optional prior-season dashed reference line."""
    color = pctile_color(pctile)
    ci_str = f"{fmt_stat(ci_lo, key)} - {fmt_stat(ci_hi, key)}"

    prev_line = ""
    prev_label = ""
    if pctile_prev is not None:
        diff = pctile - pctile_prev
        if abs(diff) >= 1:
            arrow = "&#9650;" if diff > 0 else "&#9660;"
            arrow_color = SAGE if diff > 0 else EMBER
        else:
            arrow = ""
            arrow_color = SLATE
        prev_line = (
            f'<div style="position:absolute; left:{pctile_prev:.0f}%; top:0; '
            f'height:100%; width:2px; border-left:2px dashed {SLATE}; opacity:0.7; z-index:2;"></div>'
        )
        prev_label = (
            f'<span style="color:{arrow_color}; font-size:0.75rem; margin-left:8px;">'
            f'{arrow} {PRIOR_SEASON}: {pctile_prev:.0f}th</span>'
        )

    return (
        f'<div style="margin:12px 0;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">'
        f'<span style="color:{SLATE}; font-size:0.85rem; font-weight:600;">{label}{prev_label}</span>'
        f'<span style="color:{SLATE}; font-size:0.8rem;">{pctile:.0f}th percentile | Range: {ci_str}</span>'
        f'</div>'
        f'<div style="position:relative; width:100%; background:{DARK}; border-radius:6px; height:22px; '
        f'overflow:hidden; border:1px solid {DARK_BORDER};">'
        f'<div style="height:100%; width:{pctile:.0f}%; background:{color}; border-radius:5px;"></div>'
        f'{prev_line}'
        f'</div>'
        f'</div>'
    )


def observed_pctile_bar_html(
    label: str,
    pctile: float,
    value: float,
    key: str,
) -> str:
    """Render a percentile bar for an observed stat (no CI)."""
    color = pctile_color(pctile)
    val_str = fmt_stat(value, key)

    return (
        f'<div style="margin:12px 0;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">'
        f'<span style="color:{SLATE}; font-size:0.85rem; font-weight:600;">{label}</span>'
        f'<span style="color:{SLATE}; font-size:0.8rem;">{pctile:.0f}th percentile | {val_str}</span>'
        f'</div>'
        f'<div style="position:relative; width:100%; background:{DARK}; border-radius:6px; height:22px; '
        f'overflow:hidden; border:1px solid {DARK_BORDER};">'
        f'<div style="height:100%; width:{pctile:.0f}%; background:{color}; border-radius:5px;"></div>'
        f'</div>'
        f'</div>'
    )
