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
    color: var(--tdd-cream);
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
    color: var(--tdd-slate);
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    text-decoration: none;
    transition: color 0.15s;
}}
.proj-nav-active {{
    color: var(--tdd-gold);
    border-bottom: 2px solid var(--tdd-gold);
}}
.lb-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--tdd-dark-border);
}}
.lb-title {{
    color: var(--tdd-gold);
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}
.lb-subtitle {{
    color: var(--tdd-slate);
    font-size: 0.70rem;
    font-weight: 400;
    margin-left: 0.4rem;
}}
.lb-row {{
    display: flex;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--tdd-dark-border-faint);
}}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{
    color: var(--tdd-slate);
    font-size: 0.82rem;
    min-width: 1.6rem;
    text-align: right;
    margin-right: 0.5rem;
}}
.lb-rank-top {{ color: var(--tdd-gold); font-weight: 700; }}
.lb-headshot {{
    margin-left: 0.5rem;
    margin-right: 0.5rem;
}}
.lb-name {{
    color: var(--tdd-cream);
    font-size: 0.95rem;
    font-weight: 600;
    flex: 1;
}}
.lb-name a {{
    color: inherit;
    text-decoration: none;
}}
.lb-name a:hover {{
    color: var(--tdd-gold);
    text-decoration: underline;
}}
.lb-team {{
    color: var(--tdd-slate);
    font-size: 0.80rem;
    margin-right: 0.5rem;
}}
.lb-val {{
    color: var(--tdd-sage);
    font-size: 1rem;
    font-weight: 700;
    min-width: 2.5rem;
    text-align: right;
}}
.lb-range {{
    color: var(--tdd-slate);
    font-size: 0.7rem;
    min-width: 4.2rem;
    text-align: right;
    margin-left: 0.3rem;
}}
.lb-watch-header {{
    color: var(--tdd-slate);
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.6rem 0 0.3rem 0;
    padding-top: 0.4rem;
    border-top: 1px solid var(--tdd-dark-border);
    font-style: italic;
}}
.lb-watch-row {{
    display: flex;
    align-items: center;
    padding: 0.18rem 0;
    opacity: 0.75;
}}
.lb-watch-name {{
    color: var(--tdd-slate);
    font-size: 0.9rem;
    flex: 1;
}}
.lb-watch-val {{
    color: var(--tdd-slate);
    font-size: 0.78rem;
    font-weight: 800;
    min-width: 2.5rem;
    text-align: right;
}}
.lb-watch-range {{
    color: var(--tdd-slate);
    font-size: 0.62rem;
    min-width: 4.2rem;
    text-align: right;
    margin-left: 0.3rem;
    opacity: 0.8;
}}

.stSelectbox div[data-baseweb="select"] > div {{
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}}

.stSelectbox div[data-baseweb="select"] {{
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #C8A96E !important;
    cursor: pointer !important;
}}
.stSelectbox div[data-baseweb="select"] > div:hover {{
    background-color: transparent !important;
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
    link_type: str = "",
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
        rank_class = "lb-rank-top lb-rank" if i <= 5 else "lb-rank"

        hs = f'<span class="lb-headshot">{headshot_html(pid, size=50)}</span>'

        if link_type:
            profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"
            name_html = f'<a href="{profile_url}">{name}</a>'
        else:
            name_html = name

        team_html = f'<span class="lb-team" data-team="{team}">{team}</span>' if team else ""

        range_html = ""
        if has_range and pd.notna(row.get(p10_col)) and pd.notna(row.get(p90_col)):
            lo = _fmt(row[p10_col], fmt)
            hi = _fmt(row[p90_col], fmt)
            range_html = f'<span class="lb-range">({lo}-{hi})</span>'

        rows_html.append(
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{i}.</span>'
            f'{hs}'
            f'<span class="lb-name">{name_html}</span>'
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
                    f'<span class="lb-watch-name">{name}'
                    f' (<span data-team="{team}">{team}</span>)</span>'
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

    # ── Title ─────────────────────────────────────────────────────
    st.markdown(
        f'<div class="proj-header">'
        f'<div class="proj-title">2026 PROJECTIONS</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


    # ── Dropdown Filters ──────────────────────────────────────────────
    if "proj_player_type" not in st.session_state:
        st.session_state.proj_player_type = "Batter"
    if "proj_league" not in st.session_state:
        st.session_state.proj_league = "All"

    filter_cols = st.columns([2, 2, 2, 3])
    
    with filter_cols[0]:
        player_type = st.selectbox(
            "Player Type",
            options=["Batter", "Pitcher"],
            key="proj_player_type",
            label_visibility="collapsed"
        )
    with filter_cols[1]:
        league_filter = st.selectbox(
            "League",
            options=["All", "American League", "National League"],
            key="proj_league",
            label_visibility="collapsed"
        )
    with filter_cols[2]:
        n_show = st.selectbox(
            "Show Top", 
            [5, 10, 15], 
            index=1, 
            key="proj_n",
            label_visibility="collapsed"
        )
    with filter_cols[3]:
        # Add a tiny bit of margin so the checkbox vertically aligns with the text
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        show_watch = st.checkbox("Show Players to Watch", value=False, key="proj_watch")

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

    # Apply league filter
    if st.session_state.proj_league != "ALL" and league_lookup:
        target_league = "AL" if "American" in st.session_state.proj_league else "NL"
        league_ids = {pid for pid, lg in league_lookup.items() if lg == target_league}
        df = df[df[id_col].isin(league_ids)]

    # ── Leaderboard cards ─────────────────────────────────────────
    leaderboards = BATTER_LEADERBOARDS if player_type == "Batter" else PITCHER_LEADERBOARDS
    lt = "hitter" if player_type == "Batter" else "pitcher"

    for i in range(0, len(leaderboards), 3):
        batch = leaderboards[i:i+3]
        cols = st.columns(len(batch))
        for col, (title, prefix, fmt, hib, role_fn) in zip(cols, batch):
            with col:
                _render_leaderboard(
                    df, title, prefix, fmt, hib,
                    teams_lookup, id_col, name_col,
                    show_watch=show_watch, n_show=n_show,
                    role_filter=role_fn, link_type=lt,
                )

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Projections from PA-by-PA game simulator with Bayesian hierarchical rate models. "
        "Ranges show 80% credible interval (p10-p90). "
        "Players to Watch have limited MLB track record — projections carry higher uncertainty. "
        f"{'wRC+ uses FanGraphs linear weights (100 = league average).' if player_type == 'Batter' else 'FIP-ERA strips out BABIP/sequencing noise — more predictive than traditional ERA.'}"
    )
