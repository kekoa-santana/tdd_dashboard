"""Projections page — Sortable Bayesian projection tables for pitchers and hitters."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    PITCHER_STATS, HITTER_STATS,
    PITCHER_OBSERVED_STATS, HITTER_OBSERVED_STATS,
    PITCHER_COUNTING_DISPLAY, HITTER_COUNTING_DISPLAY,
)
from services.data_loader import load_projections, load_counting, load_player_teams
from utils.helpers import strip_accents, get_injury_lookup
from utils.formatters import fmt_stat


def page_projections() -> None:
    """Sortable Bayesian projection tables for pitchers and hitters."""
    st.markdown('<div class="section-header">Projections</div>',
                unsafe_allow_html=True)

    player_type = st.radio(
        "Player type",
        ["Pitcher", "Hitter"],
        horizontal=True,
        key="proj_type",
    )

    df = load_projections(player_type.lower())
    if df.empty:
        st.warning(
            "No projection data found. "
            "Run `python scripts/precompute_dashboard_data.py` first."
        )
        return

    if player_type == "Pitcher":
        id_col, name_col, hand_col = "pitcher_id", "pitcher_name", "pitch_hand"
        stat_configs = PITCHER_STATS
        counting_display = PITCHER_COUNTING_DISPLAY
    else:
        id_col, name_col, hand_col = "batter_id", "batter_name", "batter_stand"
        stat_configs = HITTER_STATS
        counting_display = HITTER_COUNTING_DISPLAY

    counting_df = load_counting(player_type.lower())
    if not counting_df.empty:
        counting_cols = [id_col] + [
            c for c in counting_df.columns
            if c.endswith("_mean") or c.endswith("_p10") or c.endswith("_p90")
            or c.startswith("actual_")
        ]
        available = [c for c in counting_cols if c in counting_df.columns]
        df = df.merge(counting_df[available], on=id_col, how="left")

    teams_df = load_player_teams()
    if not teams_df.empty:
        df = df.merge(
            teams_df[["player_id", "team_abbr"]].rename(columns={"player_id": id_col}),
            on=id_col, how="left",
        )
        df["team_abbr"] = df["team_abbr"].fillna("")
    else:
        df["team_abbr"] = ""

    obs_configs = PITCHER_OBSERVED_STATS if player_type == "Pitcher" else HITTER_OBSERVED_STATS

    filter_cols = st.columns([2, 1, 1, 1, 1])
    with filter_cols[0]:
        search = st.text_input(
            "Search player", "", placeholder="Type a name...", key="proj_search",
        )
    with filter_cols[1]:
        team_options = ["All"] + sorted(df["team_abbr"].replace("", pd.NA).dropna().unique().tolist())
        team_filter = st.selectbox("Team", team_options, key="proj_team")
    with filter_cols[2]:
        if player_type == "Pitcher":
            role = st.selectbox("Role", ["All", "Starters", "Relievers"], key="proj_role")
        else:
            role = "All"
    with filter_cols[3]:
        hand_options = ["All"] + sorted(df[hand_col].dropna().unique().tolist())
        hand_filter = st.selectbox("Hand", hand_options, key="proj_hand")
    with filter_cols[4]:
        sort_options = ["Composite Score"] + [s[0] for s in stat_configs] + [s[0] for s in obs_configs]
        sort_by = st.selectbox("Sort by", sort_options, key="proj_sort")

    if search:
        _search_norm = strip_accents(search)
        df = df[df[name_col].apply(lambda x: _search_norm.lower() in strip_accents(str(x)).lower())]
    if team_filter != "All":
        df = df[df["team_abbr"] == team_filter]
    if player_type == "Pitcher":
        if role == "Starters":
            df = df[df["is_starter"] == 1]
        elif role == "Relievers":
            df = df[df["is_starter"] == 0]
    if hand_filter != "All":
        df = df[df[hand_col] == hand_filter]

    all_sort_configs = stat_configs + obs_configs
    if sort_by == "Composite Score":
        sort_col = "composite_score"
        ascending = False
    else:
        stat_key = next(s[1] for s in all_sort_configs if s[0] == sort_by)
        higher_is_better = next(s[2] for s in all_sort_configs if s[0] == sort_by)
        if any(s[0] == sort_by for s in stat_configs):
            sort_col = f"delta_{stat_key}"
        else:
            sort_col = stat_key
        ascending = not higher_is_better

    df_sorted = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    injury_lookup = get_injury_lookup()
    display_rows = []
    for _, row in df_sorted.iterrows():
        name_display = row[name_col]
        team = row.get("team_abbr", "")
        if team:
            name_display = f"{name_display} ({team})"
        pid = int(row[id_col])
        inj_info = injury_lookup.get(pid)
        if inj_info and inj_info["missed_games"] > 0:
            sev = inj_info["severity"]
            if sev == "major":
                name_display = f"[IL-60] {name_display}"
            elif sev == "significant":
                name_display = f"[IL] {name_display}"
            else:
                name_display = f"[DTD] {name_display}"
        r: dict[str, object] = {
            "Rank": len(display_rows) + 1,
            "Name": name_display,
            "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
            "Hand": row.get(hand_col, ""),
            "Score": round(row["composite_score"], 2),
        }
        for label, key, higher_better, _ in stat_configs:
            proj_col = f"projected_{key}"
            delta_col = f"delta_{key}"
            if proj_col in row.index and pd.notna(row.get(proj_col)):
                proj_val = fmt_stat(row[proj_col], key)
                delta_pp = row[delta_col] * 100
                if abs(delta_pp) < 0.05:
                    r[label] = proj_val
                else:
                    r[label] = f"{proj_val} ({delta_pp:+.1f})"
            else:
                r[label] = "--"
        for c_label, c_prefix, c_actual, c_hb in counting_display:
            mean_col = f"{c_prefix}_mean"
            has_proj = mean_col in row.index and pd.notna(row.get(mean_col))
            has_actual = c_actual in row.index and pd.notna(row.get(c_actual))
            if has_proj:
                proj_val = int(round(row[mean_col]))
                if has_actual:
                    delta = proj_val - int(row[c_actual])
                    r[c_label] = f"{proj_val} ({delta:+d})"
                else:
                    r[c_label] = str(proj_val)
            else:
                r[c_label] = "--"
        display_rows.append(r)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.caption(
        f"Showing {len(display_df)} {player_type.lower()}s. "
        "Composite score weights stat deltas (normalized, direction-aware). "
        "Positive = projected improvement. Deltas shown in parentheses (pp). "
        "Counting stats (K, BB, HR, Outs) are Bayesian rate x playing time projections."
    )
