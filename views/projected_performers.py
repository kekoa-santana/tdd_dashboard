"""Projected Performers -- props with strong projected edge over market lines."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, EMBER, SLATE, CREAM
from services.data_loader import load_projections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STAT_LABELS = {
    "K": "Strikeouts",
    "H": "Hits",
    "HR": "Home Runs",
    "TB": "Total Bases",
    "BB": "Walks",
    "R": "Runs",
    "RBI": "RBIs",
    "Outs": "Outs",
}


def _american_to_implied(american: str | int | float | None) -> float | None:
    """Convert American odds to implied probability."""
    if american is None or (isinstance(american, float) and pd.isna(american)):
        return None
    try:
        cleaned = str(american).replace("\u2212", "-").replace("\u2013", "-")
        odds = int(cleaned)
    except (ValueError, TypeError):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _build_name_lookup() -> dict[int, str]:
    """Build player_id -> name from projections."""
    lookup: dict[int, str] = {}
    hp = load_projections("hitter")
    if not hp.empty and "batter_name" in hp.columns:
        for _, r in hp.iterrows():
            lookup[int(r["batter_id"])] = r["batter_name"]
    pp = load_projections("pitcher")
    if not pp.empty and "pitcher_name" in pp.columns:
        for _, r in pp.iterrows():
            lookup[int(r["pitcher_id"])] = r["pitcher_name"]
    return lookup


def _edge_color(edge: float) -> str:
    if edge >= 10:
        return SAGE
    if edge >= 5:
        return GOLD
    return SLATE


def _recommendation(edge: float) -> tuple[str, str]:
    """Return (label, color) based on edge size."""
    if edge >= 15:
        return "Strong", SAGE
    if edge >= 8:
        return "Lean", GOLD
    if edge >= 3:
        return "Slight", SLATE
    return "Pass", EMBER


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def page_projected_performers() -> None:
    """Render the Projected Performers page."""
    from services.data_loader import load_game_props

    st.markdown(
        '<div class="section-header">Projected Performers</div>',
        unsafe_allow_html=True,
    )

    all_props = load_game_props()
    if all_props.empty:
        st.warning("No game props data found.")
        return

    # Compat: coalesce old and new column names
    if "line_mid" in all_props.columns:
        if "line" not in all_props.columns:
            all_props.rename(
                columns={"line_mid": "line", "p_over_mid": "p_over"},
                inplace=True,
            )
        else:
            all_props["line"] = all_props["line"].fillna(all_props["line_mid"])
            all_props["p_over"] = all_props["p_over"].fillna(
                all_props["p_over_mid"]
            )

    # Resolve player names
    name_lookup = _build_name_lookup()

    # Upcoming picks only
    props = all_props[
        (all_props["game_status"].isin(["scheduled", "in_progress", ""]))
        | all_props["game_status"].isna()
    ].copy()

    if props.empty:
        st.info("No upcoming games with props data.")
        return

    # Resolve player names
    name_lookup = _build_name_lookup()
    props["player_name"] = props["player_id"].map(
        lambda pid: name_lookup.get(int(pid), str(pid))
    )

    # Compute implied probability from DK odds
    props["dk_implied"] = props["vegas_odds"].apply(_american_to_implied)

    # Model edge = model p_over - DK implied
    props["edge"] = props.apply(
        lambda r: (r["p_over"] - r["dk_implied"]) * 100
        if pd.notna(r.get("dk_implied")) and pd.notna(r.get("p_over"))
        else None,
        axis=1,
    )

    # Split: props with DK lines vs without
    has_dk = props[props["vegas_line"].notna()].copy()
    no_dk = props[props["vegas_line"].isna()].copy()

    # Filter: P(over) >= 63%
    MIN_P_OVER = 0.63
    has_dk_strong = has_dk[has_dk["p_over"] >= MIN_P_OVER].copy()
    no_dk_strong = no_dk[no_dk["p_over"] >= MIN_P_OVER].copy()

    # ---------------------------------------------------------------------------
    # Section 1: Props with DK lines -- show edge
    # ---------------------------------------------------------------------------
    st.markdown(
        '<div style="color:var(--tdd-gold); font-size:1.0rem; font-weight:700; '
        'margin:1rem 0 0.5rem; letter-spacing:0.3px;">With Market Lines</div>',
        unsafe_allow_html=True,
    )

    if has_dk_strong.empty:
        st.markdown(
            '<div class="tdd-meta">No props above 63% with market lines available. '
            "Lines may not be posted yet for upcoming games.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Sort by edge descending
        has_dk_strong = has_dk_strong.sort_values("edge", ascending=False)

        rows_html = ""
        for _, row in has_dk_strong.iterrows():
            rows_html += _performer_row(row, has_dk_line=True)

        st.markdown(
            f'<div style="display:flex; flex-direction:column; gap:4px;">'
            f"{rows_html}</div>",
            unsafe_allow_html=True,
        )



def _performer_row(row: pd.Series, *, has_dk_line: bool) -> str:
    """Render a single prop recommendation row."""
    name = row["player_name"]
    stat = row["stat"]
    stat_label = _STAT_LABELS.get(stat, stat)
    team = row["team"]
    opp = row["opponent"]
    expected = row["expected"]
    p_over = row["p_over"]
    line = row["line"]
    ptype = row["player_type"]
    type_badge = "P" if ptype == "pitcher" else "H"

    line_str = f"{line:.0f}" if line == int(line) else f"{line:.1f}"
    pct = p_over * 100

    if has_dk_line:
        edge = float(row["edge"])
        dk_odds = str(row["vegas_odds"]) if pd.notna(row["vegas_odds"]) else ""
        dk_implied = row["dk_implied"]
        dk_pct = f"{dk_implied * 100:.0f}%" if pd.notna(dk_implied) else ""
        rec_label, rec_color = _recommendation(edge)
        e_color = _edge_color(edge)

        edge_html = (
            f'<span style="color:{e_color}; font-size:0.8rem; font-weight:700; '
            f'white-space:nowrap; min-width:60px; text-align:right;">'
            f'{edge:+.0f}%</span>'
        )
        rec_html = (
            f'<span style="background:{rec_color}22; color:{rec_color}; '
            f'font-size:0.7rem; font-weight:700; padding:2px 8px; '
            f'border-radius:4px; white-space:nowrap;">{rec_label}</span>'
        )
        odds_html = (
            f'<span style="color:{SLATE}; font-size:0.75rem; white-space:nowrap;">'
            f'{dk_odds} ({dk_pct})</span>'
        )
    else:
        edge_html = ""
        rec_html = (
            f'<span style="background:{SLATE}22; color:{SLATE}; '
            f'font-size:0.7rem; font-weight:700; padding:2px 8px; '
            f'border-radius:4px; white-space:nowrap;">No Line</span>'
        )
        odds_html = ""

    return (
        f'<div style="display:flex; align-items:center; gap:0.5rem; '
        f'padding:6px 10px; background:var(--tdd-dark-card); '
        f'border:1px solid var(--tdd-dark-border); border-radius:6px; '
        f'flex-wrap:wrap;">'
        # Rec badge
        f'{rec_html}'
        # Type badge
        f'<span style="font-size:0.65rem; color:{SLATE}; '
        f'border:1px solid var(--tdd-dark-border); border-radius:3px; '
        f'padding:0 0.25rem;">{type_badge}</span>'
        # Name + matchup
        f'<span style="color:{CREAM}; font-size:0.85rem; font-weight:600; '
        f'min-width:0; flex:1;">{name}'
        f'<span style="color:{SLATE}; font-size:0.75rem; margin-left:0.4rem;">'
        f'{team} vs {opp}</span></span>'
        # Stat + line
        f'<span style="color:{CREAM}; font-size:0.8rem; white-space:nowrap;">'
        f'{stat_label} O {line_str}</span>'
        # Model P(over)
        f'<span style="color:{SAGE}; font-size:0.8rem; font-weight:600; '
        f'white-space:nowrap;">{pct:.0f}%</span>'
        # DK odds
        f'{odds_html}'
        # Edge
        f'{edge_html}'
        f'</div>'
    )
