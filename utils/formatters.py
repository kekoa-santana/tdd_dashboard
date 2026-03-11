"""Formatting and color utility functions for the TDD Dashboard."""
from __future__ import annotations

import pandas as pd

from config import GOLD, EMBER, SAGE, SLATE, CREAM, POSITIVE, NEGATIVE


def fmt_pct(val: float, decimals: int = 1) -> str:
    """Format a 0-1 rate as percentage string."""
    return f"{val * 100:.{decimals}f}%"


def fmt_xwoba(val: float) -> str:
    """Format xwOBA as .XXX."""
    return f".{val * 1000:.0f}"


def fmt_stat(val: float, key: str, decimals: int = 1) -> str:
    """Format a stat value based on its key."""
    if key == "xwoba":
        return fmt_xwoba(val)
    if key in ("avg_exit_velo", "avg_velo"):
        return f"{val:.1f}"
    if key == "sprint_speed":
        return f"{val:.1f}"
    if key == "release_extension":
        return f"{val:.1f}"
    return fmt_pct(val, decimals)


def fmt_trad(val: float, fmt: str) -> str:
    """Format a traditional stat value."""
    if pd.isna(val):
        return "--"
    if fmt == ".000":
        return f"{val:.3f}"
    if fmt == "0.00":
        return f"{val:.2f}"
    return str(val)


def delta_html(val: float, higher_is_better: bool = True) -> str:
    """Format a delta as colored HTML span."""
    pct = val * 100
    improving = (pct > 0 and higher_is_better) or (pct < 0 and not higher_is_better)
    if abs(pct) < 0.05:
        return f'<span style="color:{SLATE}; font-weight:600;">0.0pp</span>'
    elif improving:
        return f'<span style="color:{POSITIVE}; font-weight:600;">{pct:+.1f}pp</span>'
    else:
        return f'<span style="color:{NEGATIVE}; font-weight:600;">{pct:+.1f}pp</span>'


def whiff_quality_color(whiff_rate: float) -> str:
    """Color-code a whiff rate: green=elite, gold=avg, red=poor."""
    if whiff_rate >= 0.35:
        return SAGE
    elif whiff_rate >= 0.25:
        return GOLD
    elif whiff_rate >= 0.15:
        return SLATE
    else:
        return EMBER


def xwoba_quality_color(xwoba: float) -> str:
    """Color-code xwOBA against: green=suppresses contact, red=gets hit."""
    if xwoba <= 0.280:
        return SAGE
    elif xwoba <= 0.340:
        return GOLD
    elif xwoba <= 0.400:
        return SLATE
    else:
        return EMBER


def spark_color_rate(value: float, league_avg: float, higher_is_worse: bool) -> str:
    """Color a rate stat relative to league average."""
    ratio = value / league_avg if league_avg > 0 else 1.0
    if higher_is_worse:
        if ratio >= 1.25:
            return EMBER
        elif ratio >= 1.05:
            return GOLD
        elif ratio >= 0.85:
            return SLATE
        else:
            return SAGE
    else:
        if ratio >= 1.25:
            return SAGE
        elif ratio >= 1.05:
            return GOLD
        elif ratio >= 0.85:
            return SLATE
        else:
            return EMBER


def spark_color_xwoba(value: float, for_pitcher: bool = False) -> str:
    """Color xwOBA — green = dangerous for hitters, green = suppresses for pitchers."""
    if for_pitcher:
        if value <= 0.280:
            return SAGE
        elif value <= 0.340:
            return GOLD
        elif value <= 0.400:
            return SLATE
        else:
            return EMBER
    else:
        if value >= 0.420:
            return SAGE
        elif value >= 0.350:
            return GOLD
        elif value >= 0.300:
            return SLATE
        else:
            return EMBER


def spark_html(value: float, max_val: float, color: str,
               alpha: float = 0.85) -> str:
    """Render an inline pill-shaped sparkbar."""
    width_pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div class="spark-bar" style="width:{width_pct:.0f}%; '
        f'background:{color}; opacity:{alpha:.2f};"></div>'
    )
