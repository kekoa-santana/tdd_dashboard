"""Breakout Candidates page — GMM-derived breakout models for hitters and pitchers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.alerts import tdd_info, tdd_warn
from config import GOLD, SAGE, SLATE, EMBER, DARK_CARD
from services.data_loader import (
    load_hitter_breakout_candidates,
    load_pitcher_breakout_candidates,
    load_player_teams,
)
from components.headshot import headshot_html
from components.diamond_rating import diamond_rating_html
from utils.team_names import team_short
from views.player_rankings import _render_ranking_card, _CSS as _RANKING_CSS
from components.expandable_card import EXPANDABLE_CARD_CSS


# ── Archetype configuration ─────────────────────────────────────────────

_ARCHETYPE_COLORS: dict[str, str] = {
    "Diamond in the Rough": SAGE,
    "Power Surge": EMBER,
    "Stuff Dominant": EMBER,
    "Command Leap": SAGE,
    "ERA Correction": GOLD,
}

_HITTER_PROB_COL: dict[str, str] = {
    "Diamond in the Rough": "prob_diamond_in_the_rough",
    "Power Surge": "prob_power_surge",
}

_PITCHER_PROB_COL: dict[str, str] = {
    "Stuff Dominant": "prob_stuff_dominant",
    "Command Leap": "prob_command_leap",
    "ERA Correction": "prob_era_correction",
}

# Stat badges per archetype: (label, column, format)
_HITTER_ARCHETYPE_STATS: dict[str, list[tuple[str, str, str]]] = {
    "Diamond in the Rough": [
        ("wOBA", "woba", "rate"), ("xwOBA", "xwoba", "rate"),
        ("Sprint", "sprint_speed", "f1"), ("OAA", "oaa", "d"),
        ("Z-Con%", "z_contact_pct", "pct"), ("Chase%", "chase_rate", "pct"),
    ],
    "Power Surge": [
        ("wOBA", "woba", "rate"), ("xwOBA", "xwoba", "rate"),
        ("Avg EV", "avg_exit_velo", "f1"), ("HH%", "hard_hit_pct", "pct"),
        ("Brl%", "barrel_pct", "pct"), ("K%", "k_pct", "pct"),
    ],
}

_PITCHER_ARCHETYPE_STATS: dict[str, list[tuple[str, str, str]]] = {
    "Stuff Dominant": [
        ("K%", "k_pct", "pct"), ("SwStr%", "swstr_pct", "pct"),
        ("Velo", "avg_velo", "f1"), ("BB%", "bb_pct", "pct"),
        ("ERA", "era", "f2"),
    ],
    "Command Leap": [
        ("BB%", "bb_pct", "pct"), ("Zone%", "zone_pct", "pct"),
        ("F-Str%", "first_strike_pct", "pct"), ("FIP", "fip", "f2"),
        ("ERA", "era", "f2"),
    ],
    "ERA Correction": [
        ("ERA", "era", "f2"), ("xFIP", "xfip", "f2"),
        ("Gap", "era_minus_xfip", "plus_f2"), ("HR/FB", "hr_per_fb", "pct"),
        ("FIP", "fip", "f2"),
    ],
}

_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]


# ── CSS ──────────────────────────────────────────────────────────────────

_BREAKOUT_CSS = """
<style>
.breakout-card {
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--tdd-dark-border);
    border-radius: 0;
    padding: 0.9rem 0;
    margin-bottom: 0.5rem;
    min-height: 200px;
}
.breakout-card-header {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
}
.breakout-card-name {
    color: var(--tdd-cream);
    font-size: 0.95rem;
    font-weight: 700;
    line-height: 1.2;
}
.breakout-card-meta {
    color: var(--tdd-slate);
    font-size: 0.78rem;
    font-weight: 400;
}
.breakout-card-badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.breakout-card-narrative {
    color: var(--tdd-slate);
    font-size: 0.76rem;
    line-height: 1.35;
    margin: 0.35rem 0;
}
.breakout-card-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.35rem;
}
.breakout-stat {
    background: var(--tdd-dark-border);
    border-radius: 3px;
    padding: 0.12rem 0.35rem;
    font-size: 0.68rem;
    white-space: nowrap;
}
.breakout-stat-label {
    color: var(--tdd-slate);
    margin-right: 0.15rem;
}
.breakout-stat-value {
    color: var(--tdd-cream);
    font-weight: 600;
}
.breakout-hole {
    color: var(--tdd-gold);
    font-size: 0.72rem;
    margin-top: 0.3rem;
    font-style: italic;
}
.breakout-compact {
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--tdd-dark-border);
    border-radius: 0;
    padding: 0.6rem 0;
    text-align: center;
    min-height: 130px;
}
.breakout-compact-pos {
    color: var(--tdd-gold);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
}
</style>
"""


# ── Helpers ──────────────────────────────────────────────────────────────

def _fmt(val: float, spec: str) -> str:
    """Format a stat value."""
    if pd.isna(val):
        return ""
    if spec == "pct":
        return f"{val:.1%}"
    if spec == "rate":
        return f"{val:.3f}"
    if spec == "f1":
        return f"{val:.1f}"
    if spec == "f2":
        return f"{val:.2f}"
    if spec == "d":
        return f"{val:.0f}"
    if spec == "plus_f2":
        return f"{val:+.2f}"
    return str(val)

def _merge_teams(
    df: pd.DataFrame, id_col: str,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Merge team info onto a breakout dataframe."""
    teams_df = load_player_teams()
    teams_lookup: dict[int, str] = {}
    if not teams_df.empty:
        df = df.merge(
            teams_df[["player_id", "team_abbr"]].rename(
                columns={"player_id": id_col}
            ),
            on=id_col,
            how="left",
        )
        df["team_abbr"] = df["team_abbr"].fillna("")
        teams_lookup = dict(
            zip(teams_df["player_id"].astype(int), teams_df["team_abbr"])
        )
    else:
        df["team_abbr"] = ""
    return df, teams_lookup


# ── Card builders ────────────────────────────────────────────────────────

def _build_compact_card_html(
    row: pd.Series,
    id_col: str,
    name_col: str,
    position: str,
    teams_lookup: dict[int, str],
) -> str:
    """Build compact card for position-best view."""
    pid = int(row[id_col])
    name = row[name_col]
    age = int(row["age"]) if pd.notna(row.get("age")) else ""
    team = teams_lookup.get(pid, "")
    archetype = row.get("breakout_type", "")
    score = row.get("breakout_score", 0)

    arch_color = _ARCHETYPE_COLORS.get(archetype, SLATE)
    headshot = headshot_html(pid, size=35)
    rating_html = diamond_rating_html(score, size="sm")

    meta = f'<span data-team="{team}">{team_short(team)}</span>' if team else ""
    if age:
        meta = f"{meta} \u00b7 {age}" if meta else str(age)

    return (
        f'<div class="breakout-compact">'
        f'<div class="breakout-compact-pos">{position}</div>'
        f'<div style="display:flex; align-items:center; gap:0.3rem; '
        f'justify-content:center;">'
        f'{headshot}'
        f'<div style="text-align:left;">'
        f'<div class="breakout-card-name" style="font-size:0.82rem;">{name}</div>'
        f'<div class="breakout-card-meta">{meta}</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:0.3rem;">{rating_html}</div>'
        f'<div style="margin-top:0.2rem;">'
        f'<span class="breakout-card-badge" '
        f'style="background:{arch_color}; color:{DARK_CARD}; font-size:0.62rem;">'
        f'{archetype}</span>'
        f'</div>'
        f'</div>'
    )


# ── Section renderers ────────────────────────────────────────────────────

_HITTER_DETAIL_STATS = [
    ("wOBA", "woba", ".000"), ("xwOBA", "xwoba", ".000"),
    ("K%", "k_pct", "pct"), ("BB%", "bb_pct", "pct"),
    ("Brl%", "barrel_pct", "pct"), ("Sprint", "sprint_speed", "dec1"),
]

_PITCHER_DETAIL_STATS = [
    ("K%", "k_pct", "pct"), ("BB%", "bb_pct", "pct"),
    ("SwStr%", "swstr_pct", "pct"), ("Velo", "avg_velo", "dec1"),
    ("ERA", "era", "0.00"), ("FIP", "fip", "0.00"),
]


def _render_position_bests(
    df: pd.DataFrame,
    teams_lookup: dict[int, str],
) -> None:
    """Render best breakout candidate per position (hitters only)."""
    st.markdown(
        '<div class="tdd-section-hdr" style="margin-top:1rem;">'
        'Best Breakout Candidate by Position</div>',
        unsafe_allow_html=True,
    )

    pos_best: dict[str, pd.Series] = {}
    for pos in _POSITIONS:
        pos_df = df[df["position"] == pos]
        if not pos_df.empty:
            pos_best[pos] = pos_df.nlargest(1, "breakout_score").iloc[0]

    if not pos_best:
        tdd_info("No position data available.")
        return

    positions = list(pos_best.keys())
    for i in range(0, len(positions), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(positions):
                pos = positions[idx]
                row = pos_best[pos]
                card_html = _build_compact_card_html(
                    row, "batter_id", "batter_name", pos, teams_lookup,
                )
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)


def _render_hitter_table(df: pd.DataFrame, teams_lookup: dict[int, str]) -> None:
    """Render filtered hitter breakout leaderboard."""
    col_pos, col_arch, col_search = st.columns([1, 1, 2])
    with col_pos:
        pos_opts = ["All"] + sorted(
            df["position"].dropna().unique().tolist()
        )
        pos_filter = st.selectbox("Position", pos_opts, key="brk_h_pos")
    with col_arch:
        arch_opts = ["All"] + sorted(
            df["breakout_type"].dropna().unique().tolist()
        )
        arch_filter = st.selectbox("Archetype", arch_opts, key="brk_h_arch")
    with col_search:
        search = st.text_input("Search batter", key="brk_h_search")

    filtered = df.copy()
    if pos_filter != "All":
        filtered = filtered[filtered["position"] == pos_filter]
    if arch_filter != "All":
        filtered = filtered[filtered["breakout_type"] == arch_filter]
    if search:
        filtered = filtered[
            filtered["batter_name"].str.contains(search, case=False, na=False)
        ]

    _render_ranking_card(
        filtered, "All Candidates", "breakout_rank",
        "batter_name", "batter_id", "breakout_score", teams_lookup,
        info_col="breakout_type", max_height=500,         wide=True, link_type="hitter",
        hover_stats=_HITTER_DETAIL_STATS,
    )


def _render_pitcher_table(df: pd.DataFrame, role_label: str, teams_lookup: dict[int, str]) -> None:
    """Render filtered pitcher breakout leaderboard."""
    col_arch, col_search = st.columns([1, 3])
    with col_arch:
        arch_opts = ["All"] + sorted(
            df["breakout_type"].dropna().unique().tolist()
        )
        arch_filter = st.selectbox(
            "Archetype", arch_opts, key=f"brk_p_{role_label}_arch",
        )
    with col_search:
        search = st.text_input(
            "Search pitcher", key=f"brk_p_{role_label}_search",
        )

    filtered = df.copy()
    if arch_filter != "All":
        filtered = filtered[filtered["breakout_type"] == arch_filter]
    if search:
        filtered = filtered[
            filtered["pitcher_name"].str.contains(
                search, case=False, na=False
            )
        ]

    _render_ranking_card(
        filtered, "All Candidates", "breakout_rank",
        "pitcher_name", "pitcher_id", "breakout_score", teams_lookup,
        info_col="breakout_type", max_height=500,         wide=True, link_type="pitcher",
        hover_stats=_PITCHER_DETAIL_STATS,
    )


# ── Role-specific renderers ──────────────────────────────────────────────

def _render_hitters() -> None:
    """Render hitter breakout section."""
    df = load_hitter_breakout_candidates()
    if df.empty:
        tdd_warn("No hitter breakout data. Run `precompute_dashboard_data.py`.")
        return

    df, teams_lookup = _merge_teams(df, "batter_id")

    _render_ranking_card(
        df.nsmallest(10, "breakout_rank"), "Hitter Breakout Candidates", "breakout_rank",
        "batter_name", "batter_id", "breakout_score", teams_lookup,
        info_col="breakout_type",         detail_stats=_HITTER_DETAIL_STATS, wide=True,
        link_type="hitter", expandable=True,
    )

    st.markdown("---")
    _render_position_bests(df, teams_lookup)

    st.markdown("---")
    _render_hitter_table(df, teams_lookup)


def _render_pitchers(is_starter: bool) -> None:
    """Render pitcher breakout section (SP or RP)."""
    df = load_pitcher_breakout_candidates()
    if df.empty:
        tdd_warn("No pitcher breakout data. Run `precompute_dashboard_data.py`.")
        return

    # Filter by role
    if "is_starter" in df.columns:
        if is_starter:
            df = df[df["is_starter"] == 1].copy()
        else:
            df = df[df["is_starter"] == 0].copy()

    if df.empty:
        role_label = "SP" if is_starter else "RP"
        tdd_info(f"No {role_label} breakout candidates found.")
        return

    df, teams_lookup = _merge_teams(df, "pitcher_id")

    role_label = "SP" if is_starter else "RP"
    _render_ranking_card(
        df.nsmallest(10, "breakout_rank"), f"{role_label} Breakout Candidates", "breakout_rank",
        "pitcher_name", "pitcher_id", "breakout_score", teams_lookup,
        info_col="breakout_type",         detail_stats=_PITCHER_DETAIL_STATS, wide=True,
        link_type="pitcher", expandable=True,
    )

    st.markdown("---")
    _render_pitcher_table(df, role_label, teams_lookup)


# ── Main page ────────────────────────────────────────────────────────────

def page_breakout() -> None:
    """Breakout Candidates page."""
    st.markdown(
        '<div class="brand-header">'
        '<div><div class="brand-title">Breakout Candidates</div>'
        '<div class="brand-subtitle">'
        'GMM-derived breakout archetypes identifying players poised for '
        'a performance leap in 2026</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(_BREAKOUT_CSS, unsafe_allow_html=True)
    st.markdown(_RANKING_CSS, unsafe_allow_html=True)
    st.markdown(EXPANDABLE_CARD_CSS, unsafe_allow_html=True)

    category = st.radio(
        "Category",
        ["Hitters", "Starting Pitchers", "Relief Pitchers"],
        horizontal=True,
        key="breakout_cat",
    )

    if category == "Hitters":
        _render_hitters()
    elif category == "Starting Pitchers":
        _render_pitchers(is_starter=True)
    else:
        _render_pitchers(is_starter=False)

    # Methodology
    with st.expander("Methodology"):
        st.markdown(
            "**Breakout Model**\n\n"
            "A Gaussian Mixture Model (GMM) identifies distinct breakout "
            "archetypes based on player statistical profiles. Each player is "
            "assigned to their most likely archetype and scored on breakout "
            "potential.\n\n"
            "**Hitter Archetypes (k=2)**\n"
            "- **Diamond in the Rough** \u2014 high contact/defense, low "
            "offensive output. Breakout trigger: improved contact quality\n"
            "- **Power Surge** \u2014 raw power tools, elevated strikeout "
            "rates. Breakout trigger: approach refinement\n\n"
            "**Pitcher Archetypes (k=3)**\n"
            "- **Stuff Dominant** \u2014 elite whiff/K rates, command gaps. "
            "Breakout trigger: command development\n"
            "- **Command Leap** \u2014 solid command metrics, moderate stuff. "
            "Breakout trigger: improved zone management\n"
            "- **ERA Correction** \u2014 ERA significantly above peripheral "
            "indicators (xFIP, FIP). Breakout trigger: regression to true "
            "talent\n\n"
            "**Scoring**\n\n"
            "`breakout_score` = GMM cluster probability \u00d7 room-to-grow "
            "(distance from archetype ceiling). Higher scores indicate "
            "stronger archetype fit with more upside remaining."
        )
