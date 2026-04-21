"""Metric card and percentile bar HTML components."""
from __future__ import annotations

import pandas as pd

from config import GOLD, EMBER, SAGE, SLATE, CREAM

# Percentile population thresholds
QUALIFIED_PA = 150   # rate stats (scales naturally as season progresses)
MIN_PA = 25          # advanced / counting stats (early-season friendly)


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


def hybrid_percentile_rank(
    population_df: pd.DataFrame,
    value: float,
    col: str,
    higher_is_better: bool,
    min_pa: int = MIN_PA,
    pa_col: str = "pa",
) -> float | None:
    """Percentile rank with PA-filtered population.

    Returns None if the population is too small (<20 qualifying players)
    or the column is missing.
    """
    if col not in population_df.columns or pa_col not in population_df.columns:
        return None
    qualified = population_df.loc[population_df[pa_col] >= min_pa, col].dropna()
    if len(qualified) < 20:
        return None
    return percentile_rank(qualified, value, higher_is_better)


def stat_chip(
    value: str, label: str, tooltip: str = "", color: str = CREAM,
) -> str:
    """Single stat chip: large colored value over small uppercase label."""
    title = f' title="{tooltip}"' if tooltip else ""
    cursor = " cursor:help;" if tooltip else ""
    return (
        f'<div style="text-align:center; padding:4px 10px;"{title}>'
        f'<div style="color:{color}; font-size:1.4rem; font-weight:700;{cursor}">{value}</div>'
        f'<div style="color:{SLATE}; font-size:0.65rem; text-transform:uppercase; '
        f'letter-spacing:1px;">{label}</div></div>'
    )


def stat_chip_row(chips: list[str], margin: str = "4px 0 8px") -> str:
    """Wrap stat chips in a centered flex row."""
    if not chips:
        return ""
    return (
        f'<div style="display:flex; flex-wrap:wrap; gap:8px; '
        f'justify-content:center; margin:{margin};">'
        + "".join(chips) + '</div>'
    )


def grade_bar(label: str, grade: float, ci_lo: float | None = None, ci_hi: float | None = None) -> str:
    """Horizontal skill bar for a 20-80 scouting grade."""
    # Map 20-80 to 0-100% width
    pct = max(0, min(100, (grade - 20) / 60 * 100))
    color = pctile_color(pct)
    grade_int = int(round(grade))

    ci_html = ""
    if ci_lo is not None and ci_hi is not None:
        lo_pct = max(0, min(100, (ci_lo - 20) / 60 * 100))
        hi_pct = max(0, min(100, (ci_hi - 20) / 60 * 100))
        ci_html = (
            f'<div style="position:absolute; left:{lo_pct:.0f}%; width:{hi_pct - lo_pct:.0f}%; '
            f'top:0; height:100%; background:{color}; opacity:0.15; border-radius:3px;"></div>'
        )

    return (
        f'<div style="display:flex; align-items:center; gap:8px; margin:3px 0;">'
        f'<div style="width:80px; flex-shrink:0; color:{SLATE}; font-size:0.75rem; text-align:right;">{label}</div>'
        f'<div style="flex:1; position:relative; height:14px; background:rgba(255,255,255,0.06); '
        f'border-radius:3px; overflow:hidden;">'
        f'{ci_html}'
        f'<div style="width:{pct:.0f}%; height:100%; background:{color}; border-radius:3px; '
        f'transition:width 0.3s;"></div>'
        f'</div>'
        f'<div style="width:24px; color:{color}; font-size:0.75rem; font-weight:700;">{grade_int}</div>'
        f'</div>'
    )


def grade_bar_block(grades: list[tuple[str, float, float | None, float | None]]) -> str:
    """Render a block of grade bars."""
    bars = "".join(grade_bar(lbl, g, lo, hi) for lbl, g, lo, hi in grades if g is not None)
    if not bars:
        return ""
    return f'<div style="margin:4px 0 8px; max-width:320px;">{bars}</div>'


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

