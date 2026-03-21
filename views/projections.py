"""Projections page — Leaderboard cards per projected metric."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER
from services.data_loader import load_counting_sim, load_player_teams


# ── Leaderboard definitions ───────────────────────────────────────────

# (title, column_prefix, format, higher_is_better, filter_fn_or_None)
HITTER_LEADERBOARDS = [
    ("wRC+", "projected_wrc_plus", "int", True, None),
    ("Home Runs", "total_hr", "int", True, None),
    ("Runs", "total_r", "int", True, None),
    ("RBI", "total_rbi", "int", True, None),
    ("Stolen Bases", "total_sb", "int", True, None),
    ("Walks", "total_bb", "int", True, None),
    ("Strikeouts (fewest)", "total_k", "int", False, None),
]

PITCHER_LEADERBOARDS = [
    ("FIP-ERA", "projected_fip_era", "dec2", False, lambda df: df[df["role"] == "SP"]),
    ("Strikeouts", "total_k", "int", True, lambda df: df[df["role"] == "SP"]),
    ("Innings Pitched", "projected_ip", "dec0", True, lambda df: df[df["role"] == "SP"]),
    ("Walks (fewest)", "total_bb", "int", False, lambda df: df[df["role"] == "SP"]),
    ("Saves", "total_sv", "int", True, lambda df: df[df["role"].isin(["CL", "SU", "MR"])]),
    ("Holds", "total_hld", "int", True, lambda df: df[df["role"].isin(["CL", "SU", "MR"])]),
]

# Per-stat CV thresholds for confidence filtering
_CV_THRESHOLD_HIGH = 0.35   # below = confident enough to show
_CV_THRESHOLD_MED = 0.50    # below = show with caveat
_MIN_CAREER_PA = 150        # minimum career PA/BF to appear by default


# ── CSS ────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
.lb-card {{
    background: {DARK_CARD};
    border: 1px solid {DARK_BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    min-height: 320px;
}}
.lb-title {{
    color: {GOLD};
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {DARK_BORDER};
}}
.lb-subtitle {{
    color: {SLATE};
    font-size: 0.72rem;
    font-weight: 400;
    margin-left: 0.4rem;
}}
.lb-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.30rem 0;
    border-bottom: 1px solid {DARK_BORDER}18;
}}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{
    color: {SLATE};
    font-size: 0.82rem;
    min-width: 1.6rem;
    text-align: right;
    margin-right: 0.5rem;
}}
.lb-rank-top {{ color: {GOLD}; font-weight: 700; }}
.lb-name {{
    color: {CREAM};
    font-size: 0.85rem;
    font-weight: 600;
    flex: 1;
}}
.lb-team {{
    color: {SLATE};
    font-size: 0.72rem;
    margin-left: 0.3rem;
    margin-right: 0.4rem;
}}
.lb-val {{
    color: {SAGE};
    font-size: 0.92rem;
    font-weight: 700;
    min-width: 3rem;
    text-align: right;
}}
.lb-range {{
    color: {SLATE};
    font-size: 0.68rem;
    min-width: 4.5rem;
    text-align: right;
    margin-left: 0.3rem;
}}
.lb-watch-header {{
    color: {SLATE};
    font-size: 0.88rem;
    font-weight: 600;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid {DARK_BORDER};
}}
.lb-watch-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
}}
.lb-watch-name {{
    color: {SLATE};
    font-size: 0.80rem;
    flex: 1;
}}
.lb-watch-val {{
    color: {SLATE};
    font-size: 0.80rem;
    font-weight: 600;
    min-width: 3rem;
    text-align: right;
}}
.lb-watch-range {{
    color: {SLATE};
    font-size: 0.65rem;
    min-width: 4.5rem;
    text-align: right;
    margin-left: 0.3rem;
    opacity: 0.7;
}}
</style>
"""


# ── Helpers ────────────────────────────────────────────────────────────

def _fmt(val: float, fmt: str) -> str:
    if pd.isna(val):
        return "--"
    if fmt == "int":
        return str(int(round(val)))
    if fmt == "dec0":
        return f"{val:.0f}"
    if fmt == "dec1":
        return f"{val:.1f}"
    if fmt == "dec2":
        return f"{val:.2f}"
    return str(val)


def _compute_stat_cv(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Compute coefficient of variation for a stat."""
    mean_col = f"{prefix}_mean"
    sd_col = f"{prefix}_sd"
    if mean_col in df.columns and sd_col in df.columns:
        mean = df[mean_col].fillna(0).clip(lower=0.01).abs()
        sd = df[sd_col].fillna(mean * 0.5)
        return sd / mean
    return pd.Series(1.0, index=df.index)


def _render_leaderboard(
    df: pd.DataFrame,
    title: str,
    prefix: str,
    fmt: str,
    higher_is_better: bool,
    teams_lookup: dict[int, str],
    id_col: str,
    name_col: str,
    show_all: bool = False,
    n_show: int = 10,
    role_filter=None,
) -> None:
    """Render a single leaderboard card."""
    work = df.copy()

    # Apply role filter (e.g. SP only for FIP-ERA, RP only for SV)
    if role_filter is not None:
        work = role_filter(work)

    mean_col = f"{prefix}_mean"
    if mean_col not in work.columns:
        return

    work = work.dropna(subset=[mean_col])

    # Compute per-stat CV for confidence filtering
    cv = _compute_stat_cv(work, prefix)

    if not show_all:
        # Filter to confident projections: low CV + enough MLB history
        confident = cv < _CV_THRESHOLD_MED
        if "career_pa" in work.columns:
            confident = confident & (work["career_pa"] >= _MIN_CAREER_PA)
        work_confident = work[confident]
        work_watch = work[~confident]
    else:
        work_confident = work
        work_watch = pd.DataFrame()

    # Sort
    ascending = not higher_is_better
    work_confident = work_confident.sort_values(mean_col, ascending=ascending)
    top = work_confident.head(n_show)

    # Range columns
    p10_col = f"{prefix}_p10"
    p90_col = f"{prefix}_p90"
    has_range = p10_col in work.columns and p90_col in work.columns

    # Subtitle
    subtitle = "SP" if role_filter and "SP" in str(role_filter) else ""
    if "SV" in title or "Hold" in title:
        subtitle = "RP"

    # Build HTML
    rows_html = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        team = teams_lookup.get(pid, "")
        team_html = f'<span class="lb-team">{team}</span>' if team else ""
        val = _fmt(row[mean_col], fmt)
        rank_class = "lb-rank-top" if i <= 3 else "lb-rank"

        range_html = ""
        if has_range and pd.notna(row.get(p10_col)) and pd.notna(row.get(p90_col)):
            lo = _fmt(row[p10_col], fmt)
            hi = _fmt(row[p90_col], fmt)
            range_html = f'<span class="lb-range">({lo}-{hi})</span>'

        rows_html.append(
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{i}.</span>'
            f'<span class="lb-name">{name}</span>'
            f'{team_html}'
            f'<span class="lb-val">{val}</span>'
            f'{range_html}'
            f'</div>'
        )

    subtitle_html = f'<span class="lb-subtitle">{subtitle}</span>' if subtitle else ""

    html = (
        f'<div class="lb-card">'
        f'<div class="lb-title">{title}{subtitle_html}</div>'
        + "".join(rows_html)
    )

    # "Players to Watch" section for low-confidence players
    if not work_watch.empty and not show_all:
        watch = work_watch.sort_values(mean_col, ascending=ascending).head(5)
        if not watch.empty:
            watch_rows = []
            for _, row in watch.iterrows():
                name = row[name_col]
                pid = int(row[id_col])
                team = teams_lookup.get(pid, "")
                val = _fmt(row[mean_col], fmt)
                range_html = ""
                if has_range and pd.notna(row.get(p10_col)) and pd.notna(row.get(p90_col)):
                    lo = _fmt(row[p10_col], fmt)
                    hi = _fmt(row[p90_col], fmt)
                    range_html = f'<span class="lb-watch-range">({lo}-{hi})</span>'
                watch_rows.append(
                    f'<div class="lb-watch-row">'
                    f'<span class="lb-watch-name">{name} ({team})</span>'
                    f'<span class="lb-watch-val">{val}</span>'
                    f'{range_html}'
                    f'</div>'
                )
            html += (
                '<div class="lb-watch-header">Players to Watch</div>'
                + "".join(watch_rows)
            )

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Main page ─────────────────────────────────────────────────────────

def page_projections() -> None:
    """Leaderboard-style projections page."""
    st.markdown(_CSS, unsafe_allow_html=True)

    player_type = st.radio(
        "Player type",
        ["Hitter", "Pitcher"],
        horizontal=True,
        key="proj_type",
    )

    pt_key = "hitter" if player_type == "Hitter" else "pitcher"
    df = load_counting_sim(pt_key)
    if df.empty:
        st.warning("No sim projection data found. Run precompute first.")
        return

    id_col = "batter_id" if player_type == "Hitter" else "pitcher_id"
    name_col = "batter_name" if player_type == "Hitter" else "pitcher_name"

    # Load teams
    teams_df = load_player_teams()
    teams_lookup: dict[int, str] = {}
    if not teams_df.empty:
        teams_lookup = dict(
            zip(teams_df["player_id"].astype(int), teams_df["team_abbr"])
        )

    # Get career PA/BF for confidence filtering
    try:
        from lib.db import read_sql
        role = "batter" if player_type == "Hitter" else "pitcher"
        pa_col = "bat_pa" if player_type == "Hitter" else "pit_bf"
        career = read_sql(f"""
            SELECT player_id as {id_col},
                   SUM({pa_col}) as career_pa
            FROM production.fact_player_game_mlb
            WHERE player_role = :role AND season BETWEEN 2020 AND 2025
            GROUP BY player_id
        """, {"role": role})
        if not career.empty:
            df = df.merge(career, on=id_col, how="left")
            df["career_pa"] = df["career_pa"].fillna(0)
    except Exception:
        df["career_pa"] = 999  # assume established if query fails

    # Controls
    ctrl_cols = st.columns(3)
    with ctrl_cols[0]:
        league_filter = st.radio(
            "League", ["ALL", "American League", "National League"],
            horizontal=True, key="proj_league",
        )
    with ctrl_cols[1]:
        show_all = st.checkbox("Show all players (incl. low confidence)", key="proj_show_all")
    with ctrl_cols[2]:
        n_show = st.selectbox("Show top", [5, 10, 15, 20], index=1, key="proj_n")

    # League filtering (if we have team data)
    # TODO: Add AL/NL team mapping when available

    # Page header
    season_label = "2026"
    st.markdown(
        f'<div style="text-align:center; margin-bottom:1rem;">'
        f'<span style="color:{CREAM}; font-size:1.6rem; font-weight:800; letter-spacing:1px;">'
        f'{season_label} {"BATTER" if player_type == "Hitter" else "PITCHER"} PROJECTIONS'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # Leaderboard cards
    leaderboards = HITTER_LEADERBOARDS if player_type == "Hitter" else PITCHER_LEADERBOARDS

    # Render in rows of 3
    for i in range(0, len(leaderboards), 3):
        batch = leaderboards[i:i+3]
        cols = st.columns(len(batch))
        for col, (title, prefix, fmt, hib, role_fn) in zip(cols, batch):
            with col:
                _render_leaderboard(
                    df, title, prefix, fmt, hib,
                    teams_lookup, id_col, name_col,
                    show_all=show_all, n_show=n_show,
                    role_filter=role_fn,
                )

    # Footer
    st.markdown("---")
    st.caption(
        "Projections from PA-by-PA game simulator with Bayesian hierarchical rate models. "
        "Ranges show 80% credible interval (p10-p90). "
        "Players to Watch have limited MLB track record — projections carry higher uncertainty. "
        f"{'wRC+ uses FanGraphs linear weights (100 = league average).' if player_type == 'Hitter' else 'FIP-ERA strips out BABIP/sequencing — more predictive than traditional ERA.'}"
    )
