"""Player Rankings page — TDD composite rankings for pitchers, batters, and prospects."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER
from components.metric_cards import metric_card
from components.diamond_rating import diamond_rating_text
from lib.diamond_rating import score_to_diamonds, diamond_tier
from services.data_loader import (
    load_rankings,
    load_prospect_readiness,
    load_milb_translated,
    load_milb_factors,
)


# ── Tier / score / health color helpers ─────────────────────────────────────

_PROSPECT_TIER_COLORS = {
    "Elite": GOLD,
    "Impact": EMBER,
    "Solid": SAGE,
    "Developing": SLATE,
    "Org Filler": CREAM,
}

_HEALTH_COLORS = {
    "Excellent": SAGE,
    "Good": GOLD,
    "Fair": SLATE,
    "Caution": EMBER,
    "Unknown": CREAM,
}

_READINESS_TIER_COLORS = {
    "Elite": GOLD,
    "Strong": EMBER,
    "Developing": SAGE,
    "Fringe": SLATE,
    "Long Shot": CREAM,
}

_LEVEL_ORDER = ["AAA", "AA", "A+", "A", "ROK"]


def _score_color(val: float) -> str:
    """Color-code a diamond rating value (0-5 scale)."""
    if val >= 4.0:
        return f"color: {GOLD}; font-weight: bold"
    if val >= 3.0:
        return f"color: {SAGE}; font-weight: bold"
    if val >= 2.0:
        return f"color: {SLATE}"
    return f"color: {CREAM}"


def _style_tier(val: str) -> str:
    """Color-code prospect tier."""
    color = _PROSPECT_TIER_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


def _style_health(val: str) -> str:
    """Color-code health label."""
    color = _HEALTH_COLORS.get(val, CREAM)
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
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f} / 5"), unsafe_allow_html=True)

    # Compute diamond rating column
    filtered = filtered.copy()
    if "tdd_value_score" in filtered.columns:
        filtered["_diamond_rating"] = filtered["tdd_value_score"].apply(score_to_diamonds)

    # Display table
    display_map = {
        "overall_rank": "#",
        "role_rank": "Role #",
        "pitcher_name": "Pitcher",
        "role": "Role",
        "age": "Age",
        "pitch_hand": "Throws",
        "_diamond_rating": "Rating",
        "health_label": "Health",
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
    sort_col = "overall_rank" if "overall_rank" in available else "_diamond_rating"
    ascending = sort_col == "overall_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Rating", "{:.1f}"), ("Stuff", "{:.3f}"), ("Command", "{:.3f}"),
        ("Workload", "{:.3f}"), ("Trajectory", "{:.3f}"),
        ("K%", "{:.1%}"), ("BB%", "{:.1%}"), ("SwStr%", "{:.1%}"), ("CSW%", "{:.1%}"),
        ("Proj K%", "{:.1%}"), ("Proj BB%", "{:.1%}"),
        ("BF", "{:,.0f}"), ("#", "{:.0f}"), ("Role #", "{:.0f}"), ("Age", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "Rating" in display_df.columns:
        styler = styler.map(_score_color, subset=["Rating"])
    if "Health" in display_df.columns:
        styler = styler.map(_style_health, subset=["Health"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Rating** = Diamond Rating (0-5) from weighted composite of "
        "Stuff (50%), Command (20%), Workload (15%), Trajectory (15%). "
        "**Health** = injury risk tier blended into Workload score. "
        "**K%/BB%** are 2025 observed; **Proj** columns are 2026 Bayesian projections."
    )


# ── Batter Rankings ─────────────────────────────────────────────────────────

def _render_batter_rankings(df: pd.DataFrame) -> None:
    """Render batter rankings table with filters."""
    col_pos, col_bat, col_search = st.columns([1, 1, 2])
    with col_pos:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist()) if "position" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", positions, key="rank_h_pos")
    with col_bat:
        bats = ["All", "L", "R", "B"]
        bat_filter = st.selectbox("Bats", bats, key="rank_h_bat")
    with col_search:
        search = st.text_input("Search batter", key="rank_h_search")

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
        st.markdown(metric_card("Batters Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_pos = filtered["position"].nunique() if "position" in filtered.columns else 0
        st.markdown(metric_card("Positions", f"{n_pos}"), unsafe_allow_html=True)
    with cols[2]:
        avg_score = filtered["tdd_value_score"].mean() if len(filtered) > 0 else 0
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f} / 5"), unsafe_allow_html=True)
    with cols[3]:
        top_woba = filtered["woba"].median() if "woba" in filtered.columns and len(filtered) > 0 else 0
        st.markdown(metric_card("Median wOBA", f"{top_woba:.3f}"), unsafe_allow_html=True)

    # Compute diamond rating column
    filtered = filtered.copy()
    if "tdd_value_score" in filtered.columns:
        filtered["_diamond_rating"] = filtered["tdd_value_score"].apply(score_to_diamonds)

    display_map = {
        "overall_rank": "#",
        "pos_rank": "Pos #",
        "batter_name": "Batter",
        "position": "Pos",
        "age": "Age",
        "batter_stand": "Bats",
        "_diamond_rating": "Rating",
        "health_label": "Health",
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
    sort_col = "overall_rank" if "overall_rank" in available else "_diamond_rating"
    ascending = sort_col == "overall_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Rating", "{:.1f}"), ("Offense", "{:.3f}"), ("Fielding", "{:.3f}"),
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
    if "Rating" in display_df.columns:
        styler = styler.map(_score_color, subset=["Rating"])
    if "Health" in display_df.columns:
        styler = styler.map(_style_health, subset=["Health"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Rating** = Diamond Rating (0-5) from weighted composite of "
        "Offense (55%), Fielding (20%), Playing Time (15%), Trajectory (10%). "
        "**Health** = injury risk tier blended into Playing Time score. "
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
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f} / 5"), unsafe_allow_html=True)

    # Compute diamond rating column
    filtered = filtered.copy()
    if "tdd_prospect_score" in filtered.columns:
        filtered["_diamond_rating"] = filtered["tdd_prospect_score"].apply(score_to_diamonds)

    display_map = {
        "tdd_rank": "#",
        "name": "Player",
        "primary_position": "Pos",
        "max_level": "Level",
        "min_age": "Age",
        "_diamond_rating": "Rating",
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
    sort_col = "tdd_rank" if "tdd_rank" in available else "_diamond_rating"
    ascending = sort_col == "tdd_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Rating", "{:.1f}"), ("Readiness", "{:.3f}"), ("Rate Qual", "{:.3f}"),
        ("Age Score", "{:.3f}"), ("Trajectory", "{:.3f}"), ("Pos Scarcity", "{:.3f}"),
        ("K%", "{:.1%}"), ("BB%", "{:.1%}"), ("ISO", "{:.3f}"),
        ("Age vs Lvl", "{:+.1f}"), ("MiLB PA", "{:,.0f}"),
        ("#", "{:.0f}"), ("Age", "{:.0f}"),
        ("FG FV", "{:.0f}"), ("FG Rank", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "Rating" in display_df.columns:
        styler = styler.map(_score_color, subset=["Rating"])
    if "Tier" in display_df.columns:
        styler = styler.map(_style_tier, subset=["Tier"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Rating** = Diamond Rating (0-5) from weighted composite of "
        "Rate Quality (30%), Readiness (25%), Age-Relative (15%), "
        "Trajectory (15%), Positional Scarcity (15%). "
        "**K%/BB%/ISO** are MLB-translated MiLB stats. "
        "**FG FV/Rank** are FanGraphs reference values (not used in TDD scoring)."
    )


# ── Pitching Prospect Rankings ───────────────────────────────────────────────

def _render_pitching_prospect_rankings(df: pd.DataFrame) -> None:
    """Render pitching prospect rankings table with filters."""
    col_tier, col_role, col_level, col_search = st.columns([1, 1, 1, 2])

    with col_tier:
        tiers = ["All", "Elite", "Impact", "Solid", "Developing", "Org Filler"]
        tier_filter = st.selectbox("Tier", tiers, key="rank_pp_tier")
    with col_role:
        roles = ["All", "SP", "RP"]
        role_filter = st.selectbox("Role", roles, key="rank_pp_role")
    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_pp_level")
    with col_search:
        search = st.text_input("Search prospect", key="rank_pp_search")

    filtered = df.copy()
    if tier_filter != "All" and "tdd_tier" in filtered.columns:
        filtered = filtered[filtered["tdd_tier"] == tier_filter]
    if role_filter != "All" and "pitcher_role" in filtered.columns:
        filtered = filtered[filtered["pitcher_role"] == role_filter]
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
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f} / 5"), unsafe_allow_html=True)

    # Compute diamond rating column
    filtered = filtered.copy()
    if "tdd_prospect_score" in filtered.columns:
        filtered["_diamond_rating"] = filtered["tdd_prospect_score"].apply(score_to_diamonds)

    display_map = {
        "tdd_rank": "#",
        "name": "Player",
        "pitcher_role": "Role",
        "max_level": "Level",
        "min_age": "Age",
        "_diamond_rating": "Rating",
        "tdd_tier": "Tier",
        "comp_readiness": "Readiness",
        "comp_rate_quality": "Rate Qual",
        "comp_age": "Age Score",
        "comp_trajectory": "Trajectory",
        "comp_positional": "Pos Scarcity",
        "wtd_k_pct": "K%",
        "wtd_bb_pct": "BB%",
        "wtd_hr_bf": "HR/BF",
        "youngest_age_rel": "Age vs Lvl",
        "career_milb_bf": "MiLB BF",
        "sp_pct": "SP%",
        "fg_future_value": "FG FV",
        "fg_overall_rank": "FG Rank",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    sort_col = "tdd_rank" if "tdd_rank" in available else "_diamond_rating"
    ascending = sort_col == "tdd_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Rating", "{:.1f}"), ("Readiness", "{:.3f}"), ("Rate Qual", "{:.3f}"),
        ("Age Score", "{:.3f}"), ("Trajectory", "{:.3f}"), ("Pos Scarcity", "{:.3f}"),
        ("K%", "{:.1%}"), ("BB%", "{:.1%}"), ("HR/BF", "{:.4f}"),
        ("Age vs Lvl", "{:+.1f}"), ("MiLB BF", "{:,.0f}"), ("SP%", "{:.0%}"),
        ("#", "{:.0f}"), ("Age", "{:.0f}"),
        ("FG FV", "{:.0f}"), ("FG Rank", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "Rating" in display_df.columns:
        styler = styler.map(_score_color, subset=["Rating"])
    if "Tier" in display_df.columns:
        styler = styler.map(_style_tier, subset=["Tier"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Rating** = Diamond Rating (0-5) from weighted composite of "
        "Rate Quality (30%), Readiness (25%), Age-Relative (15%), "
        "Trajectory (15%), Positional Scarcity (15%). "
        "**K%/BB%/HR/BF** are MLB-translated MiLB stats. "
        "**SP%** = share of appearances as a starter. "
        "**FG FV/Rank** are FanGraphs reference values (not used in TDD scoring)."
    )


# ── Prospect Readiness ─────────────────────────────────────────────────────

def _style_readiness_tier(val: str) -> str:
    """Color-code readiness tier."""
    color = _READINESS_TIER_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


def _render_prospect_readiness(df: pd.DataFrame) -> None:
    """Render prospect readiness scores with filters (merged from prospects page)."""
    # Filters
    col_tier, col_pos, col_level, col_search = st.columns([1, 1, 1, 2])

    with col_tier:
        tier_options = ["All", "Elite", "Strong", "Developing", "Fringe"]
        tier_filter = st.selectbox("Readiness Tier", tier_options, key="rank_rd_tier")

    with col_pos:
        pos_groups = ["All"] + sorted(df["pos_group"].dropna().unique().tolist()) if "pos_group" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", pos_groups, key="rank_rd_pos")

    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_rd_level")

    with col_search:
        search = st.text_input("Search player", key="rank_rd_search")

    # Apply filters
    filtered = df.copy()
    if tier_filter != "All":
        filtered = filtered[filtered["readiness_tier"] == tier_filter]
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
        st.markdown(metric_card("Prospects", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["readiness_tier"] == "Elite").sum() if "readiness_tier" in filtered.columns else 0
        n_strong = (filtered["readiness_tier"] == "Strong").sum() if "readiness_tier" in filtered.columns else 0
        st.markdown(
            metric_card("Elite + Strong", f"{n_elite + n_strong}"),
            unsafe_allow_html=True,
        )
    with cols[2]:
        n_ranked = filtered["is_ranked"].sum() if "is_ranked" in filtered.columns else 0
        st.markdown(
            metric_card("FG Ranked", f"{int(n_ranked)}"),
            unsafe_allow_html=True,
        )
    with cols[3]:
        avg_score = filtered["readiness_score"].mean() if len(filtered) > 0 else 0
        st.markdown(
            metric_card("Avg Readiness", f"{avg_score:.1%}"),
            unsafe_allow_html=True,
        )

    # Main table
    display_map = {
        "readiness_score": "Score",
        "readiness_tier": "Tier",
        "name": "Player",
        "pos_group": "Pos",
        "max_level": "Level",
        "wtd_k_pct": "K%",
        "wtd_bb_pct": "BB%",
        "wtd_iso": "ISO",
        "sb_rate": "SB Rate",
        "youngest_age_rel": "Age vs Lvl",
        "min_age": "Age",
        "n_above": "Blocked By",
        "career_milb_pa": "MiLB PA",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    display_df = display_df.sort_values("readiness_score", ascending=False)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Score", "{:.3f}"), ("K%", "{:.1%}"), ("BB%", "{:.1%}"),
        ("ISO", "{:.3f}"), ("SB Rate", "{:.3f}"), ("Age vs Lvl", "{:+.1f}"),
        ("Blocked By", "{:.0f}"), ("MiLB PA", "{:,.0f}"), ("Age", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "Tier" in display_df.columns:
        styler = styler.map(_style_readiness_tier, subset=["Tier"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Readiness Score** = probability of sticking in MLB (200+ PA season), "
        "combining translated MiLB stats and organizational depth analysis. "
        "**K%/BB%/ISO** are MLB-translated stats from MiLB performance. "
        "**Blocked By** = prospects at same position ahead in the org pipeline."
    )

    # Translation factors reference
    with st.expander("Translation Factor Reference"):
        ptype = st.radio(
            "Type", ["Batters", "Pitchers"],
            horizontal=True, key="rank_factor_type",
        )
        factor_type = "batter" if ptype == "Batters" else "pitcher"
        factors = load_milb_factors(factor_type)

        if not factors.empty:
            st.caption(
                "Translation factors convert MiLB stats to MLB equivalents. "
                "Factor > 1.0 means the MiLB stat inflates relative to MLB."
            )
            pooled = factors[factors["pooled"] == True] if "pooled" in factors.columns else factors  # noqa: E712
            factor_cols = {
                "level": "Level", "stat": "Stat", "factor": "Factor",
                "n": "Sample Size", "p25": "P25", "p75": "P75",
            }
            f_avail = [c for c in factor_cols if c in pooled.columns]
            f_display = pooled[f_avail].copy()
            f_display.columns = [factor_cols[c] for c in f_avail]
            if "Level" in f_display.columns:
                level_cat = pd.CategoricalDtype(categories=_LEVEL_ORDER, ordered=True)
                f_display["Level"] = f_display["Level"].astype(level_cat)
                f_display = f_display.sort_values(["Level", "Stat"])
            st.dataframe(
                f_display.style.format({
                    "Factor": "{:.3f}", "P25": "{:.3f}", "P75": "{:.3f}",
                    "Sample Size": "{:,.0f}",
                }, na_rep="—"),
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("Translation factors not available.")


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
        ["Pitchers", "Batters", "Hitting Prospects", "Pitching Prospects", "Prospect Readiness"],
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

    elif category == "Batters":
        df = load_rankings("hitters")
        if df.empty:
            st.warning(
                "No batter rankings data found. "
                "Run `precompute_dashboard_data.py` to generate hitters_rankings.parquet."
            )
            return
        _render_batter_rankings(df)

    elif category == "Hitting Prospects":
        df = load_rankings("prospect")
        if df.empty:
            st.warning(
                "No prospect rankings data found. "
                "Run `precompute_dashboard_data.py` to generate prospect_rankings.parquet."
            )
            return
        _render_prospect_rankings(df)

    elif category == "Pitching Prospects":
        df = load_rankings("pitching_prospect")
        if df.empty:
            st.warning(
                "No pitching prospect rankings data found. "
                "Run `precompute_dashboard_data.py` to generate pitching_prospect_rankings.parquet."
            )
            return
        _render_pitching_prospect_rankings(df)

    else:  # Prospect Readiness
        df = load_prospect_readiness()
        if df.empty:
            st.warning(
                "No prospect readiness data found. "
                "Run `precompute_dashboard_data.py` to generate prospect_readiness.parquet."
            )
            return
        _render_prospect_readiness(df)
