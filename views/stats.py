"""Stats page — Traditional stat leaderboards for any season."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    AVAILABLE_SEASONS, UNRELIABLE_BB_SEASONS, PRIOR_SEASON,
    PITCHER_TRAD_STATS, HITTER_TRAD_STATS,
    PITCHER_TRAD_COUNTING, HITTER_TRAD_COUNTING,
)
from services.data_loader import (
    load_player_teams, load_traditional_stats, load_traditional_stats_all,
)
from utils.helpers import strip_accents
from utils.formatters import fmt_trad


def page_stats() -> None:
    """Traditional stat leaderboards for any season."""
    st.markdown('<div class="section-header">Stats</div>',
                unsafe_allow_html=True)

    ctrl_cols = st.columns([2, 2])
    with ctrl_cols[0]:
        player_type = st.radio(
            "Player type",
            ["Pitcher", "Hitter"],
            horizontal=True,
            key="stats_type",
        )
    with ctrl_cols[1]:
        season = st.selectbox(
            "Season",
            AVAILABLE_SEASONS,
            key="stats_season_sel",
        )

    _render_stats_table(player_type, season=season)


def _render_stats_table(
    player_type: str,
    key_prefix: str = "stats_inline",
    season: int = PRIOR_SEASON,
) -> None:
    """Render a compact traditional stats table for any season."""
    df = load_traditional_stats_all(player_type.lower())
    if not df.empty and "season" in df.columns:
        df = df[df["season"] == season].copy()
    else:
        df = load_traditional_stats(player_type.lower())
    if df.empty:
        st.info(f"No {season} stats data found. Run precompute first.")
        return

    if player_type == "Hitter":
        id_col, name_col = "batter_id", "batter_name"
        rate_configs = HITTER_TRAD_STATS
        counting_configs = HITTER_TRAD_COUNTING
        default_sort = "OPS"
    else:
        id_col, name_col = "pitcher_id", "pitcher_name"
        rate_configs = PITCHER_TRAD_STATS
        counting_configs = PITCHER_TRAD_COUNTING
        default_sort = "ERA"

    teams_df = load_player_teams()
    if not teams_df.empty:
        df = df.merge(
            teams_df[["player_id", "team_abbr"]].rename(columns={"player_id": id_col}),
            on=id_col, how="left",
        )
        df["team_abbr"] = df["team_abbr"].fillna("")
    else:
        df["team_abbr"] = ""

    filter_cols = st.columns([2, 1, 1, 1, 1])
    with filter_cols[0]:
        search = st.text_input("Search", "", placeholder="Type a name...", key=f"{key_prefix}_search")
    with filter_cols[1]:
        team_options = ["All"] + sorted(df["team_abbr"].replace("", pd.NA).dropna().unique().tolist())
        team_filter = st.selectbox("Team", team_options, key=f"{key_prefix}_team")
    with filter_cols[2]:
        if player_type == "Pitcher":
            role = st.selectbox("Role", ["All", "Starters", "Relievers"], key=f"{key_prefix}_role")
        else:
            role = "All"
    with filter_cols[3]:
        if player_type == "Hitter":
            min_pa = st.selectbox("Min PA", [0, 50, 100, 200, 400, 502], index=4, key=f"{key_prefix}_min")
        else:
            min_ip = st.selectbox("Min IP", [0, 10, 30, 50, 100, 162], index=3, key=f"{key_prefix}_min")
    with filter_cols[4]:
        sort_options = [s[0] for s in rate_configs] + [s[0] for s in counting_configs]
        sort_by = st.selectbox("Sort by", sort_options, index=sort_options.index(default_sort), key=f"{key_prefix}_sort")

    if search:
        _search_norm = strip_accents(search)
        df = df[df[name_col].apply(lambda x: _search_norm.lower() in strip_accents(str(x)).lower())]
    if team_filter != "All":
        df = df[df["team_abbr"] == team_filter]
    if player_type == "Pitcher":
        if role == "Starters" and "starts" in df.columns:
            df = df[df["starts"] >= 3]
        elif role == "Relievers" and "starts" in df.columns:
            df = df[df["starts"] < 3]
    if player_type == "Hitter":
        df = df[df["pa"] >= min_pa]
    else:
        df = df[df["ip"] >= min_ip]

    all_configs = rate_configs + counting_configs
    sort_col = next((s[1] for s in all_configs if s[0] == sort_by), "ops")
    is_rate = any(s[0] == sort_by for s in rate_configs)
    ascending = (not next(s[2] for s in rate_configs if s[0] == sort_by)) if is_rate else False

    df_sorted = df.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)

    display_rows = []
    for _, row in df_sorted.iterrows():
        name_display = row[name_col]
        team = row.get("team_abbr", "")
        if team:
            name_display = f"{name_display} ({team})"
        r: dict[str, object] = {"Rank": len(display_rows) + 1, "Name": name_display}
        for label, col_name in counting_configs:
            val = row.get(col_name)
            if pd.notna(val):
                r[label] = f"{val:.1f}" if col_name == "ip" else int(val)
            else:
                r[label] = "--"
        for label, col_name, _, fmt in rate_configs:
            r[label] = fmt_trad(row.get(col_name), fmt)
        display_rows.append(r)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)
    _season_note = f" | {season} regular season"
    if season == 2020:
        _season_note += " (60-game shortened season)"
    if season in UNRELIABLE_BB_SEASONS:
        _season_note += " | Note: batted ball metrics (xwOBA, barrel%) unreliable pre-2022"
    st.caption(f"Showing {len(display_df)} {player_type.lower()}s{_season_note}")
