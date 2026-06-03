"""Projection attribution panel: "Why this number".

Renders the per-driver decomposition emitted by the projection engine
(`game_prop_attribution.parquet`) as a waterfall: league baseline plus the
signed contribution of each validated driver (talent, opponent, TTO, umpire,
catcher, park, weather) and an "other" residual, summing to the projection.

This is the honest successor to the removed pitch-type matchup tab: it only
shows levers the model actually conditions on, and the contributions reconcile
exactly to the Monte-Carlo projection (guaranteed upstream).
"""
from __future__ import annotations

from html import escape

import pandas as pd

# Negligible-contribution threshold (stat units). Drivers below this are
# collapsed into a "negligible factors" note rather than drawn as bars.
_THRESHOLD = 0.05


def _driver_labels(player_type: str) -> tuple[str, str]:
    """Labels for the two structurally-different middle terms."""
    if player_type == "batter":
        return "Batter talent", "Opposing pitcher"
    return "Talent", "Opponent lineup"


_TIER_COLOR = {
    "HIGH": "var(--tdd-sage)",
    "MEDIUM": "var(--tdd-gold)",
    "LOW": "var(--tdd-ember)",
}


def build_attribution_panel(
    row: pd.Series | dict, *, name: str | None = None, key_suffix: str = "",
) -> str:
    """Build the HTML for one prop's attribution waterfall.

    Parameters
    ----------
    row : mapping
        One row of ``game_prop_attribution.parquet`` (baseline, driver_self,
        driver_opp, tto, umpire, catcher, park, weather, residual, expected,
        volume, player_type, stat, plus p10/p90/confidence_tier when present).
    name : str, optional
        Player name to title the panel so it is unambiguous which player the
        projection describes. Falls back to "Pitcher"/"Batter".
    key_suffix : str
        Unused placeholder for caller-side uniqueness; kept for signature
        stability.

    Returns
    -------
    str
        HTML string (render with ``st.markdown(..., unsafe_allow_html=True)``).
        Empty string if the row is missing.
    """
    if row is None or (isinstance(row, pd.Series) and row.empty):
        return ""

    g = (lambda k, d=0.0: float(row.get(k, d)))
    ptype = str(row.get("player_type", "pitcher"))
    stat = str(row.get("stat", "K"))
    expected = g("expected")
    baseline = g("baseline")
    volume = g("volume")
    p10 = g("p10")
    p90 = g("p90")
    tier = str(row.get("confidence_tier", "") or "").upper()
    vol_unit = "BF" if ptype == "pitcher" else "PA"
    self_label, opp_label = _driver_labels(ptype)
    title = name if name else ("Pitcher" if ptype == "pitcher" else "Batter")

    # Headline drivers (the "who" of the projection) always render, even when
    # small; context levers collapse below the negligible threshold.
    headline = [
        (self_label, g("driver_self")),
        (opp_label, g("driver_opp")),
    ]
    context = [
        ("Times through order", g("tto")),
        ("Umpire", g("umpire")),
        ("Catcher framing", g("catcher")),
        ("Park", g("park")),
        ("Weather", g("weather")),
        ("Other (fatigue, sequencing)", g("residual")),
    ]

    visible = headline + [(l, v) for l, v in context if abs(v) >= _THRESHOLD]
    hidden = [l for l, v in context if abs(v) < _THRESHOLD]
    visible.sort(key=lambda lv: abs(lv[1]), reverse=True)
    max_abs = max([abs(v) for _, v in visible] + [1e-6])

    def _bar_row(label: str, value: float) -> str:
        pos = value >= 0
        color = "var(--tdd-sage)" if pos else "var(--tdd-ember)"
        width = min(abs(value) / max_abs * 100.0, 100.0)
        sign = "+" if pos else "−"  # minus sign
        return (
            '<div style="display:flex;align-items:center;gap:8px;'
            'padding:0.18rem 0;font-size:0.72rem">'
            f'<span style="flex:0 0 38%;color:var(--tdd-cream)">{escape(label)}</span>'
            f'<span style="flex:0 0 3.0rem;text-align:right;font-family:var(--tdd-font-mono);'
            f'color:{color};font-weight:700">{sign}{abs(value):.2f}</span>'
            '<span style="flex:1;height:6px;background:var(--tdd-dark-border);'
            'border-radius:3px;overflow:hidden">'
            f'<span style="display:block;height:100%;width:{width:.1f}%;'
            f'background:{color};border-radius:3px"></span></span>'
            '</div>'
        )

    rows_html = "".join(_bar_row(l, v) for l, v in visible)

    anchor_style = (
        'display:flex;justify-content:space-between;align-items:baseline;'
        'padding:0.2rem 0;font-size:0.72rem;color:var(--tdd-slate)'
    )
    baseline_row = (
        f'<div style="{anchor_style}">'
        f'<span>League baseline</span>'
        f'<span style="font-family:var(--tdd-font-mono)">{baseline:.2f}</span></div>'
    )
    proj_row = (
        '<div style="display:flex;justify-content:space-between;align-items:baseline;'
        'padding:0.3rem 0 0;margin-top:0.2rem;border-top:1px solid var(--tdd-dark-border);'
        'font-size:0.8rem;color:var(--tdd-cream);font-weight:700">'
        f'<span>Projected {escape(stat)}</span>'
        f'<span style="font-family:var(--tdd-font-mono);color:var(--tdd-gold)">{expected:.2f}</span></div>'
    )
    hidden_note = (
        f'<div style="font-size:0.62rem;color:var(--tdd-slate);margin-top:0.3rem">'
        f'+ {len(hidden)} negligible factor{"s" if len(hidden) != 1 else ""} '
        f'(under {_THRESHOLD:.2f})</div>'
        if hidden else ""
    )

    # 80% likely-range band (P10 to P90), drawn relative to a 0..p90 track.
    band_html = ""
    if p90 > 0:
        lo_pct = max(0.0, min(p10 / p90 * 100.0, 100.0))
        exp_pct = max(0.0, min(expected / p90 * 100.0, 100.0))
        band_html = (
            '<div style="margin-top:0.45rem">'
            '<div style="font-size:0.62rem;color:var(--tdd-slate);margin-bottom:2px">'
            f'80% likely range&#58; <span style="font-family:var(--tdd-font-mono);'
            f'color:var(--tdd-cream)">{p10:.1f} to {p90:.1f}</span> {escape(stat)}</div>'
            '<div style="position:relative;height:6px;background:var(--tdd-dark-border);'
            'border-radius:3px">'
            f'<div style="position:absolute;left:{lo_pct:.1f}%;right:0;top:0;bottom:0;'
            'background:var(--tdd-slate);opacity:0.45;border-radius:3px"></div>'
            f'<div style="position:absolute;left:{exp_pct:.1f}%;top:-2px;bottom:-2px;'
            'width:2px;background:var(--tdd-gold)"></div>'
            '</div></div>'
        )

    # Epistemic confidence chip (how much data backs the talent estimate).
    chip_html = ""
    if tier in _TIER_COLOR:
        c = _TIER_COLOR[tier]
        chip_html = (
            f'<span style="float:right;font-size:0.58rem;font-weight:700;'
            f'letter-spacing:0.5px;color:{c};border:1px solid {c};border-radius:3px;'
            f'padding:1px 5px;text-transform:none">{tier} confidence</span>'
        )

    return (
        '<div style="margin:0.4rem 0 0.8rem;padding:0.6rem 0.8rem;'
        'border-left:2px solid var(--tdd-gold);background:rgba(0,0,0,0.12)">'
        '<div style="margin-bottom:0.35rem">'
        f'{chip_html}'
        f'<div style="color:var(--tdd-cream);font-weight:700;font-size:0.82rem;'
        f'font-family:var(--tdd-font-heading)">{escape(title)}</div>'
        '<div style="color:var(--tdd-gold);font-weight:700;font-size:0.62rem;'
        'letter-spacing:1px;text-transform:uppercase">'
        f'Why this number&#58; {escape(stat)} projection '
        f'<span style="color:var(--tdd-slate);font-weight:400;text-transform:none;'
        f'letter-spacing:0">over ~{volume:.0f} {vol_unit}</span></div></div>'
        f'{baseline_row}{rows_html}{proj_row}{band_html}{hidden_note}'
        '</div>'
    )
