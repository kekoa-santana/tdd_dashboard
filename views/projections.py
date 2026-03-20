"""Projections page — Leaderboard & Discovery with editorial highlights."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, SAGE, SLATE, EMBER, CREAM, DARK_CARD, DARK_BORDER,
    PITCHER_STATS, HITTER_STATS,
    PITCHER_OBSERVED_STATS, HITTER_OBSERVED_STATS,
    PITCHER_COUNTING_DISPLAY, HITTER_COUNTING_DISPLAY,
    POSITIVE, NEGATIVE,
)
from services.data_loader import (
    load_projections, load_counting, load_player_teams,
    load_hitter_breakout_candidates, load_rankings,
)
from utils.helpers import strip_accents, get_injury_lookup
from utils.formatters import fmt_stat
from components.diamond_rating import (
    diamond_rating_text_composite, diamond_rating_html_composite,
)
from lib.diamond_rating import normalize_composite, score_to_diamonds


# ── Editorial card CSS ────────────────────────────────────────────────────

_EDITORIAL_CSS = f"""
<style>
.editorial-section {{
    background: {DARK_CARD};
    border: 1px solid {DARK_BORDER};
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    min-height: 280px;
}}
.editorial-header {{
    color: {GOLD};
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {DARK_BORDER};
}}
.editorial-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid {DARK_BORDER}22;
}}
.editorial-row:last-child {{
    border-bottom: none;
}}
.editorial-rank {{
    color: {SLATE};
    font-size: 0.8rem;
    min-width: 1.4rem;
}}
.editorial-name {{
    color: {CREAM};
    font-size: 0.85rem;
    font-weight: 600;
    flex: 1;
    margin-left: 0.4rem;
}}
.editorial-team {{
    color: {SLATE};
    font-size: 0.75rem;
    margin-left: 0.3rem;
}}
.editorial-detail {{
    font-size: 0.78rem;
    margin-left: 0.5rem;
    white-space: nowrap;
}}
.editorial-positive {{
    color: {POSITIVE};
    font-weight: 600;
}}
.editorial-negative {{
    color: {NEGATIVE};
    font-weight: 600;
}}
.editorial-muted {{
    color: {SLATE};
    font-size: 0.75rem;
    margin-top: 0.1rem;
}}
.breakout-type {{
    font-size: 0.7rem;
    color: {EMBER};
    font-weight: 600;
}}
.breakout-narrative {{
    font-size: 0.72rem;
    color: {SLATE};
    margin-top: 0.1rem;
    line-height: 1.3;
}}
</style>
"""


# ── Editorial section renderers ───────────────────────────────────────────

def _render_top_risers(
    df: pd.DataFrame,
    id_col: str,
    name_col: str,
    teams_lookup: dict[int, str],
    stat_configs: list,
) -> None:
    """Render Top Risers — players with the largest positive composite_score."""
    if df.empty or "composite_score" not in df.columns:
        st.markdown(
            '<div class="editorial-section">'
            '<div class="editorial-header">Top Risers</div>'
            f'<div class="editorial-muted">No projection data available.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    top = df.nlargest(5, "composite_score")
    rows_html: list[str] = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        team = teams_lookup.get(pid, "")
        team_span = f'<span class="editorial-team">({team})</span>' if team else ""

        best_label, best_delta = "", 0.0
        for label, key, higher_better, _ in stat_configs:
            delta_col = f"delta_{key}"
            if delta_col in row.index and pd.notna(row.get(delta_col)):
                d = row[delta_col]
                improving = (d > 0 and higher_better) or (d < 0 and not higher_better)
                if improving and abs(d) > abs(best_delta):
                    best_delta = d
                    best_label = label

        if best_label:
            delta_pp = best_delta * 100
            css_class = "editorial-positive" if delta_pp > 0 else "editorial-negative"
            detail = (
                f'<span class="{css_class}">'
                f'{best_label} {delta_pp:+.1f}pp</span>'
            )
        else:
            detail = f'<span class="editorial-muted">--</span>'

        rating_html = diamond_rating_html_composite(row["composite_score"], size="sm")

        rows_html.append(
            f'<div class="editorial-row">'
            f'<span class="editorial-rank">{i}.</span>'
            f'<span class="editorial-name">{name}</span>'
            f'{team_span}'
            f'<span class="editorial-detail">{detail}</span>'
            f'</div>'
            f'<div style="padding-left:1.8rem; margin-top:-0.2rem; margin-bottom:0.2rem;">'
            f'{rating_html}</div>'
        )

    st.markdown(
        '<div class="editorial-section">'
        '<div class="editorial-header">Top Risers</div>'
        + "".join(rows_html)
        + '</div>',
        unsafe_allow_html=True,
    )


def _render_breakout_watch(
    player_type: str,
    teams_lookup: dict[int, str],
) -> None:
    """Render Breakout Watch — hitter breakout candidates or pitcher risers."""
    if player_type == "Hitter":
        breakout_df = load_hitter_breakout_candidates()
        if breakout_df.empty:
            st.markdown(
                '<div class="editorial-section">'
                '<div class="editorial-header">Breakout Watch</div>'
                f'<div class="editorial-muted">No breakout data available.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        real_candidates = breakout_df[
            breakout_df["breakout_tier"].isin(["Breakout Candidate", "On the Radar"])
        ].nlargest(5, "breakout_score")

        if real_candidates.empty:
            real_candidates = breakout_df.nlargest(5, "breakout_score")

        rows_html: list[str] = []
        for i, (_, row) in enumerate(real_candidates.iterrows(), 1):
            name = row["batter_name"]
            pid = int(row["batter_id"])
            team = teams_lookup.get(pid, "")
            team_span = f'<span class="editorial-team">({team})</span>' if team else ""

            b_type = row.get("breakout_type", "")
            type_span = (
                f'<span class="breakout-type">{b_type}</span>'
                if b_type else ""
            )

            narrative = row.get("breakout_narrative", "")
            narrative_span = ""
            if narrative:
                short = narrative[:80] + "..." if len(narrative) > 80 else narrative
                narrative_span = (
                    f'<div class="breakout-narrative">{short}</div>'
                )

            score = row.get("breakout_score", 0)
            score_span = (
                f'<span class="editorial-detail editorial-positive">'
                f'{score:.0%}</span>'
            )

            rows_html.append(
                f'<div class="editorial-row">'
                f'<span class="editorial-rank">{i}.</span>'
                f'<span class="editorial-name">{name}</span>'
                f'{team_span}'
                f'{score_span}'
                f'</div>'
                f'<div style="padding-left:1.8rem; margin-top:-0.2rem; '
                f'margin-bottom:0.2rem;">'
                f'{type_span}'
                f'{narrative_span}'
                f'</div>'
            )

        st.markdown(
            '<div class="editorial-section">'
            '<div class="editorial-header">Breakout Watch</div>'
            + "".join(rows_html)
            + '</div>',
            unsafe_allow_html=True,
        )
    else:
        rankings_df = load_rankings("pitchers")
        if rankings_df.empty or "breakout_score" not in rankings_df.columns:
            st.markdown(
                '<div class="editorial-section">'
                '<div class="editorial-header">Breakout Watch</div>'
                f'<div class="editorial-muted">No pitcher breakout data available.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        real_candidates = rankings_df[
            rankings_df["breakout_tier"].isin(["Breakout Candidate", "On the Radar"])
        ].nlargest(5, "breakout_score") if "breakout_tier" in rankings_df.columns else pd.DataFrame()

        if real_candidates.empty:
            real_candidates = rankings_df.nlargest(5, "breakout_score")

        rows_html = []
        for i, (_, row) in enumerate(real_candidates.iterrows(), 1):
            name = row["pitcher_name"]
            pid = int(row["pitcher_id"])
            team = teams_lookup.get(pid, "")
            team_span = f'<span class="editorial-team">({team})</span>' if team else ""

            b_type = row.get("breakout_type", "")
            type_span = (
                f'<span class="breakout-type">{b_type}</span>'
                if b_type else ""
            )

            b_hole = row.get("breakout_hole", "")
            hole_span = ""
            if b_hole:
                hole_span = (
                    f'<div class="breakout-narrative">Key: {b_hole}</div>'
                )

            score = row.get("breakout_score", 0)
            score_span = (
                f'<span class="editorial-detail editorial-positive">'
                f'{score:.0%}</span>'
            )

            rows_html.append(
                f'<div class="editorial-row">'
                f'<span class="editorial-rank">{i}.</span>'
                f'<span class="editorial-name">{name}</span>'
                f'{team_span}'
                f'{score_span}'
                f'</div>'
                f'<div style="padding-left:1.8rem; margin-top:-0.2rem; '
                f'margin-bottom:0.2rem;">'
                f'{type_span}'
                f'{hole_span}'
                f'</div>'
            )

        st.markdown(
            '<div class="editorial-section">'
            '<div class="editorial-header">Breakout Watch</div>'
            + "".join(rows_html)
            + '</div>',
            unsafe_allow_html=True,
        )


def _render_diamond_leaders(
    df: pd.DataFrame,
    id_col: str,
    name_col: str,
    teams_lookup: dict[int, str],
) -> None:
    """Render Diamond Leaders — top 5 by Diamond Rating."""
    if df.empty or "composite_score" not in df.columns:
        st.markdown(
            '<div class="editorial-section">'
            '<div class="editorial-header">Diamond Leaders</div>'
            f'<div class="editorial-muted">No projection data available.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    top = df.nlargest(5, "composite_score")
    rows_html: list[str] = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        team = teams_lookup.get(pid, "")
        team_span = f'<span class="editorial-team">({team})</span>' if team else ""
        rating_html = diamond_rating_html_composite(row["composite_score"], size="sm")

        rows_html.append(
            f'<div class="editorial-row">'
            f'<span class="editorial-rank">{i}.</span>'
            f'<span class="editorial-name">{name}</span>'
            f'{team_span}'
            f'<span class="editorial-detail">{rating_html}</span>'
            f'</div>'
        )

    st.markdown(
        '<div class="editorial-section">'
        '<div class="editorial-header">Diamond Leaders</div>'
        + "".join(rows_html)
        + '</div>',
        unsafe_allow_html=True,
    )


# ── Main page ─────────────────────────────────────────────────────────────

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
        teams_lookup: dict[int, str] = dict(
            zip(teams_df["player_id"].astype(int), teams_df["team_abbr"])
        )
    else:
        df["team_abbr"] = ""
        teams_lookup = {}

    # ── Editorial highlights ──────────────────────────────────────────
    st.markdown(_EDITORIAL_CSS, unsafe_allow_html=True)

    ed_cols = st.columns(3)
    with ed_cols[0]:
        _render_top_risers(df, id_col, name_col, teams_lookup, stat_configs)
    with ed_cols[1]:
        _render_breakout_watch(player_type, teams_lookup)
    with ed_cols[2]:
        _render_diamond_leaders(df, id_col, name_col, teams_lookup)

    st.markdown("---")

    # ── Full table (preserved from original) ──────────────────────────
    obs_configs = PITCHER_OBSERVED_STATS if player_type == "Pitcher" else HITTER_OBSERVED_STATS

    search = st.text_input(
        "Search player", "", placeholder="Type a name...", key="proj_search",
    )
    filter_cols = st.columns(4)
    with filter_cols[0]:
        team_options = ["All"] + sorted(df["team_abbr"].replace("", pd.NA).dropna().unique().tolist())
        team_filter = st.selectbox("Team", team_options, key="proj_team")
    with filter_cols[1]:
        if player_type == "Pitcher":
            role = st.selectbox("Role", ["All", "Starters", "Relievers"], key="proj_role")
        else:
            role = "All"
    with filter_cols[2]:
        hand_options = ["All"] + sorted(df[hand_col].dropna().unique().tolist())
        hand_filter = st.selectbox("Hand", hand_options, key="proj_hand")
    with filter_cols[3]:
        sort_options = ["Diamond Rating"] + [s[0] for s in stat_configs] + [s[0] for s in obs_configs]
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
    if sort_by == "Diamond Rating":
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
            "Rating": diamond_rating_text_composite(row["composite_score"]),
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
        "Rating = Diamond Rating (0-5) derived from weighted stat deltas. "
        "Positive = projected improvement. "
        "Deltas shown in parentheses (pp = percentage points). "
        "Counting stats (K, BB, HR, Outs) are Bayesian rate x playing time projections."
    )
