"""Lineup Creator — build hypothetical lineups, swap players, score strength."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import GOLD, SAGE, SLATE, EMBER
from services.data_loader import (
    load_probable_starters,
    load_rankings,
    load_roster,
)
from components.diamond_rating import diamond_rating_html
from lib.diamond_rating import score_to_diamonds
from utils.alerts import tdd_warn

_POSITIONS = ["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "DH"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _load_hitter_pool() -> pd.DataFrame:
    """All ranked hitters with position eligibility."""
    rankings = load_rankings("hitters")
    roster = load_roster()

    if rankings.empty:
        return pd.DataFrame()

    cols = ["batter_id", "batter_name", "position",
            "tdd_value_score", "grade_hit", "grade_power",
            "grade_speed", "grade_discipline"]
    df = rankings[[c for c in cols if c in rankings.columns]].copy()
    # Derive diamond_rating from tdd_value_score
    if "tdd_value_score" in df.columns:
        df["diamond_rating"] = df["tdd_value_score"].apply(score_to_diamonds)
    else:
        df["diamond_rating"] = 5.0

    # Merge team from roster (same source as team overview depth chart)
    if not roster.empty and "team_abbr" in roster.columns:
        df = df.merge(
            roster[["player_id", "team_abbr"]].rename(
                columns={"player_id": "batter_id"}
            ).drop_duplicates(subset="batter_id"),
            on="batter_id", how="left",
        )
    else:
        df["team_abbr"] = ""

    # Build eligibility from roster
    elig: dict[int, set[str]] = {}
    if not roster.empty:
        for _, r in roster.iterrows():
            pid = int(r["player_id"])
            positions = set()
            pp = r.get("primary_position", "")
            if pp in _POSITIONS:
                positions.add(pp)
            sec = r.get("secondary_positions")
            if sec is not None:
                for sp in list(sec):
                    if sp in _POSITIONS:
                        positions.add(sp)
            positions.add("DH")  # everyone can DH
            elig[pid] = positions

    # Build a default set from ranking position for players not in roster
    ranking_pos = dict(zip(df["batter_id"], df["position"]))
    df["eligible_positions"] = df["batter_id"].apply(
        lambda pid: elig.get(pid, {ranking_pos.get(pid, "DH"), "DH"})
    )

    return df


def _position_median_ratings(pool: pd.DataFrame) -> dict[str, float]:
    """Median diamond rating at each position (league baseline)."""
    medians: dict[str, float] = {}
    for pos in _POSITIONS:
        sub = pool[pool["position"] == pos]
        medians[pos] = float(sub["diamond_rating"].median()) if not sub.empty else 5.0
    return medians


def _eligible_for_position(pool: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Filter hitter pool to players eligible at a position."""
    mask = pool["eligible_positions"].apply(lambda s: pos in s)
    return pool[mask].sort_values("diamond_rating", ascending=False)


# ---------------------------------------------------------------------------
# Default lineup builder
# ---------------------------------------------------------------------------

def _default_lineup(team_abbr: str, pool: pd.DataFrame) -> list[dict]:
    """Build the default 9-man lineup from probable_starters."""
    ps = load_probable_starters()
    lineup: list[dict] = []

    if not ps.empty:
        team_ps = ps[ps["team_abbr"] == team_abbr].copy()
        if not team_ps.empty:
            # Deduplicate positions (keep first per position)
            team_ps = team_ps.drop_duplicates(subset=["position"], keep="first")
            for pos in _POSITIONS:
                row = team_ps[team_ps["position"] == pos]
                if not row.empty:
                    pid = int(row.iloc[0]["player_id"])
                    prow = pool[pool["batter_id"] == pid]
                    dr = float(prow["diamond_rating"].iloc[0]) if not prow.empty else 5.0
                    name = row.iloc[0]["player_name"]
                    lineup.append({
                        "position": pos, "batter_id": pid,
                        "batter_name": name, "diamond_rating": dr,
                        "team_abbr": team_abbr,
                    })
                else:
                    lineup.append({
                        "position": pos, "batter_id": 0,
                        "batter_name": "Empty", "diamond_rating": 0.0,
                        "team_abbr": team_abbr,
                    })
            return lineup

    # Fallback: best eligible player per position from team roster
    team_pool = pool[pool["team_abbr"] == team_abbr]
    assigned: set[int] = set()
    for pos in _POSITIONS:
        eligible = _eligible_for_position(team_pool, pos)
        eligible = eligible[~eligible["batter_id"].isin(assigned)]
        if not eligible.empty:
            top = eligible.iloc[0]
            assigned.add(int(top["batter_id"]))
            lineup.append({
                "position": pos, "batter_id": int(top["batter_id"]),
                "batter_name": top["batter_name"],
                "diamond_rating": float(top["diamond_rating"]),
                "team_abbr": team_abbr,
            })
        else:
            lineup.append({
                "position": pos, "batter_id": 0,
                "batter_name": "Empty", "diamond_rating": 0.0,
                "team_abbr": team_abbr,
            })
    return lineup


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _lineup_score(lineup: list[dict]) -> float:
    """Average diamond rating of the 9 starters."""
    ratings = [s["diamond_rating"] for s in lineup if s["diamond_rating"] > 0]
    return float(np.mean(ratings)) if ratings else 0.0


def _grade_color(rating: float, median: float) -> str:
    """Color based on how rating compares to position median."""
    if rating >= median + 1.5:
        return GOLD
    if rating >= median:
        return SAGE
    if rating >= median - 1.5:
        return SLATE
    return EMBER


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_lineup_creator() -> None:
    """Lineup Creator — build, score, and compare hypothetical lineups."""
    st.markdown(
        '<div class="brand-header"><div>'
        '<div class="brand-title">Lineup Creator</div>'
        '<div class="brand-subtitle">'
        'Build hypothetical lineups, find weak spots, and swap in players '
        'from any team to see the impact</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    pool = _load_hitter_pool()
    if pool.empty:
        tdd_warn("Hitter rankings not available. Run precompute first.")
        return

    teams = sorted(pool["team_abbr"].dropna().unique())
    pos_medians = _position_median_ratings(pool)

    # --- Team selector ---
    selected_team = st.selectbox(
        "Team", teams, key="lc_team", label_visibility="collapsed",
    )

    # --- Initialize lineup in session state ---
    state_key = f"lc_lineup_{selected_team}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _default_lineup(selected_team, pool)

    lineup = st.session_state[state_key]
    score = _lineup_score(lineup)

    # --- Two column layout ---
    col_lineup, col_notes = st.columns([3, 2])

    with col_lineup:
        st.markdown(
            f'<div class="tdd-section-hdr">'
            f'<span class="tdd-team-abbr" data-team="{selected_team}">'
            f'{selected_team}</span> Starting Lineup</div>',
            unsafe_allow_html=True,
        )

        for i, slot in enumerate(lineup):
            pos = slot["position"]
            current_pid = slot["batter_id"]

            # Build options for this position slot
            eligible = _eligible_for_position(pool, pos)
            # Current player always first, then team players, then all others
            team_elig = eligible[eligible["team_abbr"] == selected_team]
            other_elig = eligible[eligible["team_abbr"] != selected_team]

            options: list[str] = []
            pid_map: dict[str, int] = {}

            # Add current player first
            if current_pid and current_pid > 0:
                current_label = f"{slot['batter_name']} ({slot.get('team_abbr', '')})"
                options.append(current_label)
                pid_map[current_label] = current_pid

            # Team eligible players
            for _, r in team_elig.iterrows():
                pid = int(r["batter_id"])
                if pid == current_pid:
                    continue
                label = f"{r['batter_name']} ({r['team_abbr']}) | {r['diamond_rating']:.1f}"
                options.append(label)
                pid_map[label] = pid

            # Divider + other teams
            for _, r in other_elig.head(50).iterrows():
                pid = int(r["batter_id"])
                if pid == current_pid:
                    continue
                label = f"{r['batter_name']} ({r['team_abbr']}) | {r['diamond_rating']:.1f}"
                if label not in pid_map:
                    options.append(label)
                    pid_map[label] = pid

            dr = slot["diamond_rating"]
            median = pos_medians.get(pos, 5.0)
            grade_c = _grade_color(dr, median)

            # Position label + selectbox inline
            pos_col, sel_col, dr_col = st.columns([0.6, 4, 1.2])
            with pos_col:
                st.markdown(
                    f'<div style="color:var(--tdd-gold); font-weight:700; '
                    f'font-size:0.9rem; padding-top:0.5rem;">{pos}</div>',
                    unsafe_allow_html=True,
                )
            with sel_col:
                selected = st.selectbox(
                    pos, options, key=f"lc_pos_{selected_team}_{pos}",
                    label_visibility="collapsed",
                )
                if selected and pid_map.get(selected, current_pid) != current_pid:
                    new_pid = pid_map[selected]
                    new_row = pool[pool["batter_id"] == new_pid]
                    if not new_row.empty:
                        nr = new_row.iloc[0]
                        lineup[i] = {
                            "position": pos,
                            "batter_id": new_pid,
                            "batter_name": nr["batter_name"],
                            "diamond_rating": float(nr["diamond_rating"]),
                            "team_abbr": nr.get("team_abbr", ""),
                        }
                        st.session_state[state_key] = lineup
                        st.rerun()
            with dr_col:
                st.markdown(
                    f'<div style="color:{grade_c}; font-weight:700; '
                    f'font-size:1rem; padding-top:0.4rem; text-align:right;">'
                    f'{dr:.1f}</div>',
                    unsafe_allow_html=True,
                )

        # Reset button
        if st.button("Reset to Default", key=f"lc_reset_{selected_team}"):
            st.session_state[state_key] = _default_lineup(selected_team, pool)
            st.rerun()

    with col_notes:
        # Overall score
        st.markdown(
            f'<div style="text-align:center; margin-bottom:1.5rem;">'
            f'<div style="color:var(--tdd-cream); font-size:2rem; font-weight:800;">'
            f'{score:.1f}</div>'
            f'<div style="color:var(--tdd-slate); font-size:0.8rem; '
            f'text-transform:uppercase; letter-spacing:1px;">Lineup Score</div>'
            f'{diamond_rating_html(0, size="md", precomputed=score)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Per-position breakdown
        strengths: list[str] = []
        weak_spots: list[str] = []
        for slot in lineup:
            pos = slot["position"]
            dr = slot["diamond_rating"]
            median = pos_medians.get(pos, 5.0)
            diff = dr - median
            name = slot["batter_name"]
            if diff >= 1.5:
                strengths.append(
                    f'<span style="color:{GOLD}; font-weight:600;">{pos}</span> '
                    f'{name} ({dr:.1f}, +{diff:.1f} vs avg)'
                )
            elif diff <= -1.0:
                weak_spots.append(
                    f'<span style="color:{EMBER}; font-weight:600;">{pos}</span> '
                    f'{name} ({dr:.1f}, {diff:.1f} vs avg)'
                )

        if strengths:
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-title">Strengths</div>'
                + "".join(
                    f'<div class="insight-bullet">'
                    f'<span class="dot" style="background:{GOLD};"></span>{s}</div>'
                    for s in strengths
                )
                + '</div>',
                unsafe_allow_html=True,
            )

        if weak_spots:
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-title">Weak Spots</div>'
                + "".join(
                    f'<div class="insight-bullet">'
                    f'<span class="dot" style="background:{EMBER};"></span>{s}</div>'
                    for s in weak_spots
                )
                + '</div>',
                unsafe_allow_html=True,
            )

        # Position-by-position bar chart
        st.markdown(
            '<div class="tdd-section-hdr" style="margin-top:1rem;">Position Grades</div>',
            unsafe_allow_html=True,
        )
        for slot in lineup:
            pos = slot["position"]
            dr = slot["diamond_rating"]
            median = pos_medians.get(pos, 5.0)
            pct = min(100, dr / 10.0 * 100)
            med_pct = min(100, median / 10.0 * 100)
            color = _grade_color(dr, median)
            st.markdown(
                f'<div style="margin-bottom:6px;">'
                f'<div style="display:flex; justify-content:space-between; '
                f'align-items:baseline; margin-bottom:1px;">'
                f'<span style="color:var(--tdd-cream); font-size:0.8rem; '
                f'font-weight:600; min-width:2rem;">{pos}</span>'
                f'<span style="color:var(--tdd-slate); font-size:0.72rem;">'
                f'{slot["batter_name"]}</span>'
                f'<span style="color:{color}; font-weight:700; font-size:0.82rem;">'
                f'{dr:.1f}</span>'
                f'</div>'
                f'<div style="position:relative; height:8px; '
                f'background:var(--tdd-dark-card); border-radius:4px;">'
                f'<div style="width:{pct:.0f}%; height:100%; background:{color}; '
                f'border-radius:4px;"></div>'
                f'<div style="position:absolute; left:{med_pct:.0f}%; top:-2px; '
                f'height:12px; width:1px; background:{SLATE}88;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="color:var(--tdd-slate); font-size:0.7rem; '
            f'margin-top:0.5rem;">Vertical line = position median</div>',
            unsafe_allow_html=True,
        )
