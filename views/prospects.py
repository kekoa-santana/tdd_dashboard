"""Prospects page — MLB readiness scores, FanGraphs rankings, and translated MiLB stats."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
    DARK_CARD, DARK_BORDER,
)
from components.metric_cards import metric_card
from services.data_loader import (
    load_prospect_readiness,
    load_milb_translated,
    load_milb_factors,
)


# Tier styling
_TIER_COLORS = {
    "Elite": GOLD,
    "Strong": EMBER,
    "Developing": SAGE,
    "Fringe": SLATE,
    "Long Shot": CREAM,
}

_LEVEL_ORDER = ["AAA", "AA", "A+", "A", "ROK"]

_POS_GROUP_ORDER = ["All", "C", "MI", "CI", "OF"]


def _style_tier(val: str) -> str:
    """Color-code readiness tier."""
    color = _TIER_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


def page_prospects() -> None:
    """Render the Prospects page with readiness scores and rankings."""
    st.markdown(
        '<div class="brand-header">'
        '<div><div class="brand-title">Prospect Readiness</div>'
        '<div class="brand-subtitle">'
        'MLB readiness scores powered by translated stats '
        'and organizational depth analysis</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    df = load_prospect_readiness()
    if df.empty:
        st.warning(
            "No prospect readiness data found. "
            "Run `precompute_dashboard_data.py` to generate prospect_readiness.parquet."
        )
        return

    # ── Filters ──
    col_tier, col_pos, col_level, col_search = st.columns([1, 1, 1, 2])

    with col_tier:
        tier_options = ["All", "Elite", "Strong", "Developing", "Fringe"]
        tier_filter = st.selectbox("Readiness Tier", tier_options, key="prospect_tier")

    with col_pos:
        pos_filter = st.selectbox("Position", _POS_GROUP_ORDER, key="prospect_pos")

    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()]
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="prospect_level")

    with col_search:
        search = st.text_input("Search player", key="prospect_search")

    # Apply filters
    filtered = df.copy()
    if tier_filter != "All":
        filtered = filtered[filtered["readiness_tier"] == tier_filter]
    if pos_filter != "All":
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if level_filter != "All":
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
        ]

    # ── Summary metrics ──
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

    # ── Main table ──
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

    # Format
    fmt = {}
    if "Score" in display_df.columns:
        fmt["Score"] = "{:.3f}"
    if "K%" in display_df.columns:
        fmt["K%"] = "{:.1%}"
    if "BB%" in display_df.columns:
        fmt["BB%"] = "{:.1%}"
    if "ISO" in display_df.columns:
        fmt["ISO"] = "{:.3f}"
    if "SB Rate" in display_df.columns:
        fmt["SB Rate"] = "{:.3f}"
    if "Age vs Lvl" in display_df.columns:
        fmt["Age vs Lvl"] = "{:+.1f}"
    if "Blocked By" in display_df.columns:
        fmt["Blocked By"] = "{:.0f}"
    if "MiLB PA" in display_df.columns:
        fmt["MiLB PA"] = "{:,.0f}"
    if "Age" in display_df.columns:
        fmt["Age"] = "{:.0f}"

    # Apply tier styling
    styler = display_df.style.format(fmt, na_rep="—")
    if "Tier" in display_df.columns:
        styler = styler.applymap(_style_tier, subset=["Tier"])

    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.caption(
        "**Readiness Score** = probability of sticking in MLB (200+ PA season), "
        "combining translated MiLB stats and organizational depth analysis. "
        "**K%/BB%/ISO** are MLB-translated stats from MiLB performance. "
        "**Blocked By** = prospects at same position ahead in the org pipeline."
    )

    # ── Expandable: translation factors reference ──
    with st.expander("Translation Factor Reference"):
        ptype = st.radio(
            "Type", ["Batters", "Pitchers"],
            horizontal=True, key="factor_type",
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
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Translation factors not available.")
