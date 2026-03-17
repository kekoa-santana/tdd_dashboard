"""Player Rankings page — TDD composite rankings for pitchers, hitters, and prospects."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER
from components.metric_cards import metric_card
from services.data_loader import load_rankings


# ── Tier / score color helpers ──────────────────────────────────────────────

_PROSPECT_TIER_COLORS = {
    "Elite": GOLD,
    "Impact": EMBER,
    "Solid": SAGE,
    "Developing": SLATE,
    "Org Filler": CREAM,
}

_LEVEL_ORDER = ["AAA", "AA", "A+", "A", "ROK"]


def _score_color(val: float) -> str:
    """Color-code a 0-1 TDD value score."""
    if val >= 0.80:
        return f"color: {GOLD}; font-weight: bold"
    if val >= 0.60:
        return f"color: {SAGE}; font-weight: bold"
    if val >= 0.40:
        return f"color: {SLATE}"
    return f"color: {CREAM}"


def _style_tier(val: str) -> str:
    """Color-code prospect tier."""
    color = _PROSPECT_TIER_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


# ── Pitcher Rankings ────────────────────────────────────────────────────────

def _render_pitcher_rankings(df: pd.DataFrame) -> None:
    """Render pitcher rankings table with filters."""
    # Filters
    col_role, col_hand, col_search = st.columns([1, 1, 2])
    with col_role:
        roles = ["All"] + sorted(df["role"].dropna().unique().tolist()) if "role" in df.columns else ["All"]
        role_filter = st.selectbox("Role", roles, key="rank_p_role")
    with col_hand:
        hands = ["All", "L", "R"]
        hand_filter = st.selectbox("Throws", hands, key="rank_p_hand")
    with col_search:
        search = st.text_input("Search pitcher", key="rank_p_search")

    filtered = df.copy()
    if role_filter != "All" and "role" in filtered.columns:
        filtered = filtered[filtered["role"] == role_filter]
    if hand_filter != "All" and "pitch_hand" in filtered.columns:
        filtered = filtered[filtered["pitch_hand"] == hand_filter]
    if search:
        filtered = filtered[
            filtered["pitcher_name"].str.contains(search, case=False, na=False)
        ]

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Pitchers Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_sp = (filtered["role"] == "SP").sum() if "role" in filtered.columns else 0
        st.markdown(metric_card("Starters", f"{n_sp}"), unsafe_allow_html=True)
    with cols[2]:
        n_rp = (filtered["role"] == "RP").sum() if "role" in filtered.columns else 0
        st.markdown(metric_card("Relievers", f"{n_rp}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["tdd_value_score"].mean() if len(filtered) > 0 else 0
        st.markdown(metric_card("Avg TDD Score", f"{avg_score:.3f}"), unsafe_allow_html=True)

    # Display table
    display_map = {
        "overall_rank": "#",
        "role_rank": "Role #",
        "pitcher_name": "Pitcher",
        "role": "Role",
        "age": "Age",
        "pitch_hand": "Throws",
        "tdd_value_score": "TDD Score",
        "stuff_score": "Stuff",
        "command_score": "Command",
        "workload_score": "Workload",
        "trajectory_score": "Trajectory",
        "k_pct": "K%",
        "bb_pct": "BB%",
        "swstr_pct": "SwStr%",
        "csw_pct": "CSW%",
        "batters_faced": "BF",
        "projected_k_rate": "Proj K%",
        "projected_bb_rate": "Proj BB%",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    sort_col = "overall_rank" if "overall_rank" in available else "tdd_value_score"
    ascending = sort_col == "overall_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("TDD Score", "{:.3f}"), ("Stuff", "{:.3f}"), ("Command", "{:.3f}"),
        ("Workload", "{:.3f}"), ("Trajectory", "{:.3f}"),
        ("K%", "{:.1%}"), ("BB%", "{:.1%}"), ("SwStr%", "{:.1%}"), ("CSW%", "{:.1%}"),
        ("Proj K%", "{:.1%}"), ("Proj BB%", "{:.1%}"),
        ("BF", "{:,.0f}"), ("#", "{:.0f}"), ("Role #", "{:.0f}"), ("Age", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "TDD Score" in display_df.columns:
        styler = styler.map(_score_color, subset=["TDD Score"])

    st.dataframe(styler, use_container_width=True, hide_index=True, height=600)

    st.caption(
        "**TDD Score** = weighted composite of Stuff (50%), Command (20%), "
        "Workload (15%), Trajectory (15%). Sub-scores are percentile-ranked (0-1). "
        "**K%/BB%** are 2025 observed; **Proj** columns are 2026 Bayesian projections."
    )


# ── Hitter Rankings ─────────────────────────────────────────────────────────

def _render_hitter_rankings(df: pd.DataFrame) -> None:
    """Render hitter rankings table with filters."""
    col_pos, col_bat, col_search = st.columns([1, 1, 2])
    with col_pos:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist()) if "position" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", positions, key="rank_h_pos")
    with col_bat:
        bats = ["All", "L", "R", "B"]
        bat_filter = st.selectbox("Bats", bats, key="rank_h_bat")
    with col_search:
        search = st.text_input("Search hitter", key="rank_h_search")

    filtered = df.copy()
    if pos_filter != "All" and "position" in filtered.columns:
        filtered = filtered[filtered["position"] == pos_filter]
    if bat_filter != "All" and "batter_stand" in filtered.columns:
        filtered = filtered[filtered["batter_stand"] == bat_filter]
    if search:
        filtered = filtered[
            filtered["batter_name"].str.contains(search, case=False, na=False)
        ]

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Hitters Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_pos = filtered["position"].nunique() if "position" in filtered.columns else 0
        st.markdown(metric_card("Positions", f"{n_pos}"), unsafe_allow_html=True)
    with cols[2]:
        avg_score = filtered["tdd_value_score"].mean() if len(filtered) > 0 else 0
        st.markdown(metric_card("Avg TDD Score", f"{avg_score:.3f}"), unsafe_allow_html=True)
    with cols[3]:
        top_woba = filtered["woba"].median() if "woba" in filtered.columns and len(filtered) > 0 else 0
        st.markdown(metric_card("Median wOBA", f"{top_woba:.3f}"), unsafe_allow_html=True)

    display_map = {
        "overall_rank": "#",
        "pos_rank": "Pos #",
        "batter_name": "Hitter",
        "position": "Pos",
        "age": "Age",
        "batter_stand": "Bats",
        "tdd_value_score": "TDD Score",
        "offense_score": "Offense",
        "fielding_combined": "Fielding",
        "pt_score": "Play Time",
        "trajectory_score": "Trajectory",
        "woba": "wOBA",
        "wrc_plus": "wRC+",
        "xwoba": "xwOBA",
        "barrel_pct": "Barrel%",
        "hard_hit_pct": "HardHit%",
        "pa": "PA",
        "projected_k_rate": "Proj K%",
        "projected_bb_rate": "Proj BB%",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    sort_col = "overall_rank" if "overall_rank" in available else "tdd_value_score"
    ascending = sort_col == "overall_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("TDD Score", "{:.3f}"), ("Offense", "{:.3f}"), ("Fielding", "{:.3f}"),
        ("Play Time", "{:.3f}"), ("Trajectory", "{:.3f}"),
        ("wOBA", "{:.3f}"), ("xwOBA", "{:.3f}"),
        ("wRC+", "{:.0f}"), ("PA", "{:,.0f}"),
        ("Barrel%", "{:.1%}"), ("HardHit%", "{:.1%}"),
        ("Proj K%", "{:.1%}"), ("Proj BB%", "{:.1%}"),
        ("#", "{:.0f}"), ("Pos #", "{:.0f}"), ("Age", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "TDD Score" in display_df.columns:
        styler = styler.map(_score_color, subset=["TDD Score"])

    st.dataframe(styler, use_container_width=True, hide_index=True, height=600)

    st.caption(
        "**TDD Score** = weighted composite of Offense (55%), Fielding (20%), "
        "Playing Time (15%), Trajectory (10%). Sub-scores are percentile-ranked (0-1). "
        "**wOBA/xwOBA/Barrel%** are 2025 observed; **Proj** columns are 2026 Bayesian projections."
    )


# ── Prospect Rankings ────────────────────────────────────────────────────────

def _render_prospect_rankings(df: pd.DataFrame) -> None:
    """Render prospect rankings table with filters."""
    col_tier, col_pos, col_level, col_search = st.columns([1, 1, 1, 2])

    with col_tier:
        tiers = ["All", "Elite", "Impact", "Solid", "Developing", "Org Filler"]
        tier_filter = st.selectbox("Tier", tiers, key="rank_pr_tier")
    with col_pos:
        pos_groups = ["All"] + sorted(df["pos_group"].dropna().unique().tolist()) if "pos_group" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", pos_groups, key="rank_pr_pos")
    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_pr_level")
    with col_search:
        search = st.text_input("Search prospect", key="rank_pr_search")

    filtered = df.copy()
    if tier_filter != "All" and "tdd_tier" in filtered.columns:
        filtered = filtered[filtered["tdd_tier"] == tier_filter]
    if pos_filter != "All" and "pos_group" in filtered.columns:
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if level_filter != "All" and "max_level" in filtered.columns:
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
        ]

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Prospects Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["tdd_tier"] == "Elite").sum() if "tdd_tier" in filtered.columns else 0
        n_impact = (filtered["tdd_tier"] == "Impact").sum() if "tdd_tier" in filtered.columns else 0
        st.markdown(metric_card("Elite + Impact", f"{n_elite + n_impact}"), unsafe_allow_html=True)
    with cols[2]:
        n_fg = filtered["fg_overall_rank"].notna().sum() if "fg_overall_rank" in filtered.columns else 0
        st.markdown(metric_card("FG Ranked", f"{int(n_fg)}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["tdd_prospect_score"].mean() if "tdd_prospect_score" in filtered.columns and len(filtered) > 0 else 0
        st.markdown(metric_card("Avg TDD Score", f"{avg_score:.3f}"), unsafe_allow_html=True)

    display_map = {
        "tdd_rank": "#",
        "name": "Player",
        "primary_position": "Pos",
        "max_level": "Level",
        "min_age": "Age",
        "tdd_prospect_score": "TDD Score",
        "tdd_tier": "Tier",
        "comp_readiness": "Readiness",
        "comp_rate_quality": "Rate Qual",
        "comp_age": "Age Score",
        "comp_trajectory": "Trajectory",
        "comp_positional": "Pos Scarcity",
        "wtd_k_pct": "K%",
        "wtd_bb_pct": "BB%",
        "wtd_iso": "ISO",
        "youngest_age_rel": "Age vs Lvl",
        "career_milb_pa": "MiLB PA",
        "fg_future_value": "FG FV",
        "fg_overall_rank": "FG Rank",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    sort_col = "tdd_rank" if "tdd_rank" in available else "tdd_prospect_score"
    ascending = sort_col == "tdd_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("TDD Score", "{:.3f}"), ("Readiness", "{:.3f}"), ("Rate Qual", "{:.3f}"),
        ("Age Score", "{:.3f}"), ("Trajectory", "{:.3f}"), ("Pos Scarcity", "{:.3f}"),
        ("K%", "{:.1%}"), ("BB%", "{:.1%}"), ("ISO", "{:.3f}"),
        ("Age vs Lvl", "{:+.1f}"), ("MiLB PA", "{:,.0f}"),
        ("#", "{:.0f}"), ("Age", "{:.0f}"),
        ("FG FV", "{:.0f}"), ("FG Rank", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "TDD Score" in display_df.columns:
        styler = styler.map(_score_color, subset=["TDD Score"])
    if "Tier" in display_df.columns:
        styler = styler.map(_style_tier, subset=["Tier"])

    st.dataframe(styler, use_container_width=True, hide_index=True, height=600)

    st.caption(
        "**TDD Score** = weighted composite of Rate Quality (30%), Readiness (25%), "
        "Age-Relative (15%), Trajectory (15%), Positional Scarcity (15%). "
        "**K%/BB%/ISO** are MLB-translated MiLB stats. "
        "**FG FV/Rank** are FanGraphs reference values (not used in TDD scoring)."
    )


# ── Main page ────────────────────────────────────────────────────────────────

def page_player_rankings() -> None:
    """Render the Player Rankings page with category selector."""
    st.markdown(
        '<div class="brand-header">'
        '<div><div class="brand-title">Player Rankings</div>'
        '<div class="brand-subtitle">'
        'TDD composite rankings blending Bayesian projections, '
        'observed performance, and scouting factors</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    category = st.selectbox(
        "Category",
        ["Pitchers", "Hitters", "Prospects"],
        key="rankings_category",
    )

    if category == "Pitchers":
        df = load_rankings("pitchers")
        if df.empty:
            st.warning(
                "No pitcher rankings data found. "
                "Run `precompute_dashboard_data.py` to generate pitchers_rankings.parquet."
            )
            return
        _render_pitcher_rankings(df)

    elif category == "Hitters":
        df = load_rankings("hitters")
        if df.empty:
            st.warning(
                "No hitter rankings data found. "
                "Run `precompute_dashboard_data.py` to generate hitters_rankings.parquet."
            )
            return
        _render_hitter_rankings(df)

    else:  # Prospects
        df = load_rankings("prospect")
        if df.empty:
            st.warning(
                "No prospect rankings data found. "
                "Run `precompute_dashboard_data.py` to generate prospect_rankings.parquet."
            )
            return
        _render_prospect_rankings(df)
