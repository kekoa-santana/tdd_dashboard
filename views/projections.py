"""Projections page — Leaderboard cards per projected metric with headshots."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.data_loader import load_counting_sim, load_roster
from components.leaderboard import render_card
from utils.alerts import tdd_warn


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

_CSS = ""


# ── Helpers ────────────────────────────────────────────────────────────

def _fmt(val: float, fmt: str) -> str:
    if pd.isna(val):
        return ""
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

    def _row_dict(row: pd.Series) -> dict:
        pid = int(row[id_col])
        d = {
            "player_id": pid,
            "name": row[name_col],
            "value": _fmt(row[mean_col], fmt),
            "team": teams_lookup.get(pid, ""),
            "link_type": link_type,
        }
        if has_range and pd.notna(row.get(p10_col)) and pd.notna(row.get(p90_col)):
            lo = _fmt(row[p10_col], fmt)
            hi = _fmt(row[p90_col], fmt)
            d["range"] = f"({lo}-{hi})"
        return d

    rows = [_row_dict(row) for _, row in top.iterrows()]

    watch_rows: list[dict] = []
    if show_watch and not work_watch_df.empty:
        watch_df = work_watch_df.sort_values(mean_col, ascending=ascending).head(5)
        watch_rows = [_row_dict(row) for _, row in watch_df.iterrows()]

    st.markdown(
        render_card(
            title=f"Top {n_show} {title}",
            rows=rows,
            subtitle=subtitle,
            watch_rows=watch_rows or None,
        ),
        unsafe_allow_html=True,
    )


# ── Main page ─────────────────────────────────────────────────────────

def page_projections() -> None:
    """Leaderboard-style projections page."""
    # ── Title ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="tdd-page-header">'
        '<div class="tdd-page-title">2026 PROJECTIONS</div>'
        '<div class="tdd-page-subtitle">'
        'PA-by-PA game simulator with Bayesian hierarchical rate models'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Dropdown Filters ──────────────────────────────────────────────
    if "proj_player_type" not in st.session_state:
        st.session_state.proj_player_type = "Batter"
    if "proj_league" not in st.session_state:
        st.session_state.proj_league = "All"

    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        player_type = st.selectbox(
            "Player Type",
            options=["Batter", "Pitcher"],
            key="proj_player_type",
            label_visibility="collapsed",
        )
    with fc2:
        st.selectbox(
            "League",
            options=["All", "American League", "National League"],
            key="proj_league",
            label_visibility="collapsed",
        )
    with fc3:
        n_show = st.selectbox(
            "Show Top",
            [5, 10, 15],
            index=1,
            key="proj_n",
            label_visibility="collapsed",
        )
    show_watch = st.checkbox("Show Players to Watch", value=False, key="proj_watch")

    pt_key = "hitter" if player_type == "Batter" else "pitcher"

    # ── Load data ─────────────────────────────────────────────────
    df = load_counting_sim(pt_key)
    if df.empty:
        tdd_warn("No sim projection data found. Run precompute first.")
        return

    id_col = "batter_id" if player_type == "Batter" else "pitcher_id"
    name_col = "batter_name" if player_type == "Batter" else "pitcher_name"

    # Load teams + league
    teams_df = load_roster()
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
    if st.session_state.proj_league != "All" and league_lookup:
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
