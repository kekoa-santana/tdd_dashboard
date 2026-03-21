"""Projections page — Leaderboard cards per projected metric with headshots."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER
from services.data_loader import load_counting_sim, load_player_teams
from components.headshot import headshot_html


# ── Leaderboard definitions ───────────────────────────────────────────

BATTER_LEADERBOARDS = [
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

_CV_THRESHOLD_MED = 0.50
_MIN_CAREER_PA = 150

# AL / NL team mapping
_AL_TEAMS = {"BAL", "BOS", "NYY", "TB", "TOR",
             "CLE", "CWS", "DET", "KC", "MIN",
             "HOU", "LAA", "OAK", "SEA", "TEX"}
_NL_TEAMS = {"ATL", "MIA", "NYM", "PHI", "WSH",
             "CHC", "CIN", "MIL", "PIT", "STL",
             "ARI", "COL", "LAD", "SD", "SF"}


# ── CSS ────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
.proj-header {{
    text-align: center;
    margin-bottom: 0.8rem;
}}
.proj-title {{
    color: {CREAM};
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 1.5px;
}}
.proj-nav {{
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    margin: 0.6rem 0;
}}
.proj-nav-btn {{
    color: {SLATE};
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    text-decoration: none;
    transition: color 0.15s;
}}
.proj-nav-active {{
    color: {GOLD};
    border-bottom: 2px solid {GOLD};
}}
.lb-card {{
    background: {DARK_CARD};
    border: 1px solid {DARK_BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}}
.lb-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {DARK_BORDER};
}}
.lb-title {{
    color: {GOLD};
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}
.lb-subtitle {{
    color: {SLATE};
    font-size: 0.70rem;
    font-weight: 400;
    margin-left: 0.4rem;
}}
.lb-row {{
    display: flex;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid {DARK_BORDER}15;
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
.lb-headshot {{
    margin-right: 0.5rem;
}}
.lb-name {{
    color: {CREAM};
    font-size: 0.85rem;
    font-weight: 600;
    flex: 1;
}}
.lb-team {{
    color: {SLATE};
    font-size: 0.70rem;
    margin-right: 0.5rem;
}}
.lb-val {{
    color: {SAGE};
    font-size: 0.92rem;
    font-weight: 700;
    min-width: 2.5rem;
    text-align: right;
}}
.lb-range {{
    color: {SLATE};
    font-size: 0.65rem;
    min-width: 4.2rem;
    text-align: right;
    margin-left: 0.3rem;
}}
.lb-watch-header {{
    color: {SLATE};
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.6rem 0 0.3rem 0;
    padding-top: 0.4rem;
    border-top: 1px solid {DARK_BORDER};
    font-style: italic;
}}
.lb-watch-row {{
    display: flex;
    align-items: center;
    padding: 0.18rem 0;
    opacity: 0.75;
}}
.lb-watch-name {{
    color: {SLATE};
    font-size: 0.78rem;
    flex: 1;
}}
.lb-watch-val {{
    color: {SLATE};
    font-size: 0.78rem;
    font-weight: 600;
    min-width: 2.5rem;
    text-align: right;
}}
.lb-watch-range {{
    color: {SLATE};
    font-size: 0.62rem;
    min-width: 4.2rem;
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
    if fmt == "dec2":
        return f"{val:.2f}"
    return str(val)


def _render_leaderboard(
    df: pd.DataFrame,
    title: str,
    prefix: str,
    fmt: str,
    higher_is_better: bool,
    teams_lookup: dict[int, str],
    id_col: str,
    name_col: str,
    show_watch: bool = False,
    n_show: int = 10,
    role_filter=None,
) -> None:
    """Render a single leaderboard card with headshots for top 3."""
    work = df.copy()
    if role_filter is not None:
        work = role_filter(work)

    mean_col = f"{prefix}_mean"
    if mean_col not in work.columns:
        return

    work = work.dropna(subset=[mean_col])

    # Per-stat CV for confidence
    sd_col = f"{prefix}_sd"
    if sd_col in work.columns:
        cv = work[sd_col].fillna(0) / work[mean_col].clip(0.01).abs()
    else:
        cv = pd.Series(0.3, index=work.index)

    # Split confident vs watch
    confident_mask = cv < _CV_THRESHOLD_MED
    if "career_pa" in work.columns:
        confident_mask = confident_mask & (work["career_pa"] >= _MIN_CAREER_PA)

    work_main = work[confident_mask]
    work_watch_df = work[~confident_mask]

    ascending = not higher_is_better
    work_main = work_main.sort_values(mean_col, ascending=ascending)
    top = work_main.head(n_show)

    p10_col = f"{prefix}_p10"
    p90_col = f"{prefix}_p90"
    has_range = p10_col in work.columns and p90_col in work.columns

    # Subtitle for role
    subtitle = ""
    if role_filter:
        fn_str = str(role_filter)
        if "SP" in fn_str:
            subtitle = "SP"
        elif "CL" in fn_str or "SU" in fn_str:
            subtitle = "RP"

    # Build rows
    rows_html = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        team = teams_lookup.get(pid, "")
        val = _fmt(row[mean_col], fmt)
        rank_class = "lb-rank-top" if i <= 3 else "lb-rank"

        # Headshot for top 3
        hs = ""
        if i <= 3:
            hs = f'<span class="lb-headshot">{headshot_html(pid, size=32)}</span>'

        team_html = f'<span class="lb-team">{team}</span>' if team else ""

        range_html = ""
        if has_range and pd.notna(row.get(p10_col)) and pd.notna(row.get(p90_col)):
            lo = _fmt(row[p10_col], fmt)
            hi = _fmt(row[p90_col], fmt)
            range_html = f'<span class="lb-range">({lo}-{hi})</span>'

        rows_html.append(
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{i}.</span>'
            f'{hs}'
            f'<span class="lb-name">{name}</span>'
            f'{team_html}'
            f'<span class="lb-val">{val}</span>'
            f'{range_html}'
            f'</div>'
        )

    subtitle_html = f'<span class="lb-subtitle">{subtitle}</span>' if subtitle else ""

    html = (
        f'<div class="lb-card">'
        f'<div class="lb-title-row">'
        f'<span class="lb-title">Top {n_show} {title}{subtitle_html}</span>'
        f'</div>'
        + "".join(rows_html)
    )

    # Players to Watch (only when toggled)
    if show_watch and not work_watch_df.empty:
        watch = work_watch_df.sort_values(mean_col, ascending=ascending).head(5)
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

    # ── Player type toggle (clickable text) ───────────────────────
    if "proj_player_type" not in st.session_state:
        st.session_state.proj_player_type = "Batter"

    type_cols = st.columns([1, 1, 4])
    with type_cols[0]:
        if st.button("Batters", key="btn_batter",
                      type="primary" if st.session_state.proj_player_type == "Batter" else "secondary"):
            st.session_state.proj_player_type = "Batter"
            st.rerun()
    with type_cols[1]:
        if st.button("Pitchers", key="btn_pitcher",
                      type="primary" if st.session_state.proj_player_type == "Pitcher" else "secondary"):
            st.session_state.proj_player_type = "Pitcher"
            st.rerun()

    player_type = st.session_state.proj_player_type
    pt_key = "hitter" if player_type == "Batter" else "pitcher"

    # ── Load data ─────────────────────────────────────────────────
    df = load_counting_sim(pt_key)
    if df.empty:
        st.warning("No sim projection data found. Run precompute first.")
        return

    id_col = "batter_id" if player_type == "Batter" else "pitcher_id"
    name_col = "batter_name" if player_type == "Batter" else "pitcher_name"

    # Load teams + league
    teams_df = load_player_teams()
    teams_lookup: dict[int, str] = {}
    league_lookup: dict[int, str] = {}
    if not teams_df.empty:
        teams_lookup = dict(zip(teams_df["player_id"].astype(int), teams_df["team_abbr"]))
        if "league" in teams_df.columns:
            league_lookup = dict(zip(teams_df["player_id"].astype(int), teams_df["league"]))

    # career_pa from parquet
    if "career_pa" not in df.columns:
        df["career_pa"] = 999

    # ── Title ─────────────────────────────────────────────────────
    st.markdown(
        f'<div class="proj-header">'
        f'<div class="proj-title">2026 {"BATTER" if player_type == "Batter" else "PITCHER"} PROJECTIONS</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── League filter (clickable text) ────────────────────────────
    if "proj_league" not in st.session_state:
        st.session_state.proj_league = "ALL"

    lg_cols = st.columns([1, 1, 1, 3])
    for i, lg in enumerate(["ALL", "American League", "National League"]):
        with lg_cols[i]:
            label = lg if lg != "ALL" else "All"
            btn_type = "primary" if st.session_state.proj_league == lg else "secondary"
            if st.button(label, key=f"btn_lg_{lg}", type=btn_type):
                st.session_state.proj_league = lg
                st.rerun()

    # Apply league filter
    if st.session_state.proj_league != "ALL" and league_lookup:
        target_league = "AL" if "American" in st.session_state.proj_league else "NL"
        league_ids = {pid for pid, lg in league_lookup.items() if lg == target_league}
        df = df[df[id_col].isin(league_ids)]

    # ── Controls ──────────────────────────────────────────────────
    ctrl_cols = st.columns([1, 1])
    with ctrl_cols[0]:
        show_watch = st.checkbox("Show Players to Watch", value=False, key="proj_watch")
    with ctrl_cols[1]:
        n_show = st.selectbox("Show top", [5, 10, 15], index=1, key="proj_n")

    # ── Leaderboard cards ─────────────────────────────────────────
    leaderboards = BATTER_LEADERBOARDS if player_type == "Batter" else PITCHER_LEADERBOARDS

    for i in range(0, len(leaderboards), 3):
        batch = leaderboards[i:i+3]
        cols = st.columns(len(batch))
        for col, (title, prefix, fmt, hib, role_fn) in zip(cols, batch):
            with col:
                _render_leaderboard(
                    df, title, prefix, fmt, hib,
                    teams_lookup, id_col, name_col,
                    show_watch=show_watch, n_show=n_show,
                    role_filter=role_fn,
                )

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Projections from PA-by-PA game simulator with Bayesian hierarchical rate models. "
        "Ranges show 80% credible interval (p10-p90). "
        "Players to Watch have limited MLB track record — projections carry higher uncertainty. "
        f"{'wRC+ uses FanGraphs linear weights (100 = league average).' if player_type == 'Batter' else 'FIP-ERA strips out BABIP/sequencing noise — more predictive than traditional ERA.'}"
    )
