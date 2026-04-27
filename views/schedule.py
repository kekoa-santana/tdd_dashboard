"""Schedule page | date-based game browser with projections and live data."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, POSITIVE, NEGATIVE, DASHBOARD_DIR,
)
from utils.alerts import tdd_info
from services.data_loader import (
    load_todays_games, load_todays_lineups, load_todays_batter_sims,
    load_update_metadata, load_pitcher_arsenal, load_hitter_vulnerability,
    load_hitter_strength,
    load_projections, load_counting, load_hitter_archetypes, load_pitcher_archetypes,
    load_bf_priors,
    load_roster, load_game_props,
    fetch_live_schedule, fetch_live_lineups, backfill_missing_lineups,
    load_pitcher_game_sim_samples, load_batter_game_sim_samples,
)
from utils.helpers import format_game_time
from utils.html import esc, esc_attr
from components.diamond_rating import diamond_rating_html
from components.expandable_card import EXPANDABLE_CARD_CSS, expandable_card_html
from components.headshot import headshot_html
from lib.fantasy_report import load_report_data, get_pitcher_scouting, ReportData


from components.scouting import render_scouting_html as _render_scouting_html


@st.cache_data(ttl=300)
def _get_scouting_report_data() -> ReportData:
    """Load scouting report data (cached 5 min)."""
    return load_report_data(DASHBOARD_DIR)


def _detect_lineup_changes(
    current_lineups: pd.DataFrame,
    precomputed_sims: pd.DataFrame,
    game_pk: int,
) -> dict:
    """Compare live lineups vs precomputed batter sims for a game.

    Parameters
    ----------
    current_lineups : pd.DataFrame
        Live lineup data (must have ``game_pk``, ``batter_id`` or
        ``player_id``, and ``team_id`` columns).
    precomputed_sims : pd.DataFrame
        Precomputed batter sims (must have ``game_pk``, ``batter_id``,
        and ``opp_starter_id`` columns).
    game_pk : int
        The game to check.

    Returns
    -------
    dict
        ``changed``: bool, ``new_batters``: list[int],
        ``missing_batters``: list[int], ``pitcher_changed``: bool.
    """
    result = {
        "changed": False,
        "new_batters": [],
        "missing_batters": [],
        "pitcher_changed": False,
    }

    if precomputed_sims.empty:
        return result

    # Batter IDs in current lineup for this game
    if not current_lineups.empty and "game_pk" in current_lineups.columns:
        game_lu = current_lineups[current_lineups["game_pk"] == game_pk]
    else:
        game_lu = pd.DataFrame()
    current_bids: set[int] = set()
    if not game_lu.empty:
        id_col = "batter_id" if "batter_id" in game_lu.columns else "player_id"
        if id_col in game_lu.columns:
            current_bids = {int(b) for b in game_lu[id_col].dropna()}

    # Batter IDs in precomputed sims for this game
    sim_game = precomputed_sims[precomputed_sims["game_pk"] == game_pk]
    sim_bids: set[int] = set()
    if not sim_game.empty:
        sim_bids = {int(b) for b in sim_game["batter_id"].dropna()}

    new_batters = sorted(current_bids - sim_bids)
    missing_batters = sorted(sim_bids - current_bids)

    result["new_batters"] = new_batters
    result["missing_batters"] = missing_batters
    result["changed"] = len(new_batters) > 0 or len(missing_batters) > 0

    return result


@st.cache_data(ttl=timedelta(minutes=5))
def _build_projection_lookup() -> dict:
    """Build pitcher_id → projection dict from pitcher_projections.parquet."""
    proj = load_projections("pitcher")
    if proj.empty or "pitcher_id" not in proj.columns:
        return {}
    lookup = {}
    for _, row in proj.iterrows():
        lookup[int(row["pitcher_id"])] = {
            "projected_k_rate": row.get("projected_k_rate"),
            "pitcher_name": row.get("pitcher_name", ""),
            "projected_bb_rate": row.get("projected_bb_rate"),
            "projected_hr_per_bf": row.get("projected_hr_per_bf"),
            "pitch_hand": row.get("pitch_hand", ""),
        }
    return lookup


@st.cache_data(ttl=timedelta(minutes=5))
def _build_schedule_lookups() -> dict:
    """Build all lookup dicts for schedule cards (cached).

    Extracts iterrows() loops that previously ran on every Streamlit rerender
    into a single cached function keyed on the underlying parquet TTLs.
    """
    from lib.diamond_rating import score_to_diamonds

    _h_arch = load_hitter_archetypes()
    _p_arch = load_pitcher_archetypes()
    _h_proj = load_projections("hitter")
    _h_count = load_counting("hitter")

    h_arch_lookup: dict[int, str] = {}
    if not _h_arch.empty:
        for _, r in _h_arch.iterrows():
            h_arch_lookup[int(r["batter_id"])] = r["archetype_name"]

    p_arch_lookup: dict[int, str] = {}
    if not _p_arch.empty:
        for _, r in _p_arch.iterrows():
            p_arch_lookup[int(r["pitcher_id"])] = r["archetype_name"]

    from services.data_loader import load_standings, load_rankings
    standings = load_standings()

    _h_rankings = load_rankings("hitters")
    _p_rankings = load_rankings("pitchers")
    diamond_lookup: dict[int, float] = {}
    if not _h_rankings.empty and "tdd_value_score" in _h_rankings.columns:
        for _, r in _h_rankings.iterrows():
            if pd.notna(r.get("tdd_value_score")):
                diamond_lookup[int(r["batter_id"])] = score_to_diamonds(r["tdd_value_score"])
    if not _p_rankings.empty and "tdd_value_score" in _p_rankings.columns:
        for _, r in _p_rankings.iterrows():
            if pd.notna(r.get("tdd_value_score")):
                diamond_lookup[int(r["pitcher_id"])] = score_to_diamonds(r["tdd_value_score"])

    # Hitter stat lookup: batter_id -> {k_rate, bb_rate, grades, ...}
    h_stat_lookup: dict[int, dict] = {}
    if not _h_proj.empty:
        for _, r in _h_proj.iterrows():
            bid = int(r["batter_id"])
            h_stat_lookup[bid] = {
                "k_rate": r.get("projected_k_rate"),
                "bb_rate": r.get("projected_bb_rate"),
                "tdd_value_score": diamond_lookup.get(bid),
                "bat_hand": r.get("batter_stand"),
            }
    if not _h_count.empty and "batter_id" in _h_count.columns:
        for _, r in _h_count.iterrows():
            bid = int(r["batter_id"])
            counting = {
                "hr": r.get("total_hr_mean"),
                "total_k": r.get("total_k_mean"),
                "total_bb": r.get("total_bb_mean"),
                "total_hr": r.get("total_hr_mean"),
            }
            if bid in h_stat_lookup:
                h_stat_lookup[bid].update(counting)
            else:
                h_stat_lookup[bid] = {
                    "k_rate": None, "bb_rate": None,
                    "tdd_value_score": diamond_lookup.get(bid),
                    **counting,
                }

    # Inject scouting grades
    _h_grade_cols = ["grade_hit", "grade_power", "grade_speed", "grade_discipline", "grade_fielding"]
    if not _h_rankings.empty:
        for _, r in _h_rankings.iterrows():
            bid = int(r["batter_id"])
            grades = {c: int(r[c]) for c in _h_grade_cols if c in r.index and pd.notna(r.get(c))}
            if bid in h_stat_lookup:
                h_stat_lookup[bid].update(grades)
            else:
                h_stat_lookup[bid] = {
                    "k_rate": None, "bb_rate": None,
                    "tdd_value_score": diamond_lookup.get(bid),
                    **grades,
                }

    p_grade_lookup: dict[int, dict] = {}
    _p_grade_cols = ["grade_stuff", "grade_command", "grade_durability"]
    if not _p_rankings.empty:
        for _, r in _p_rankings.iterrows():
            pid = int(r["pitcher_id"])
            grades = {c: int(r[c]) for c in _p_grade_cols if c in r.index and pd.notna(r.get(c))}
            grades["tdd_value_score"] = diamond_lookup.get(pid)
            p_grade_lookup[pid] = grades

    # Grade confidence intervals
    from services.data_loader import load_hitter_grade_ci, load_pitcher_grade_ci
    _h_ci = load_hitter_grade_ci()
    _p_ci = load_pitcher_grade_ci()
    _h_ci_cols = [
        "grade_hit_lo", "grade_hit_hi", "grade_power_lo", "grade_power_hi",
        "grade_speed_lo", "grade_speed_hi", "grade_discipline_lo", "grade_discipline_hi",
        "grade_fielding_lo", "grade_fielding_hi",
    ]
    if not _h_ci.empty:
        for _, r in _h_ci.iterrows():
            bid = int(r["player_id"])
            ci_vals = {c: int(r[c]) for c in _h_ci_cols if c in r.index and pd.notna(r.get(c))}
            if bid in h_stat_lookup:
                h_stat_lookup[bid].update(ci_vals)
            else:
                h_stat_lookup[bid] = ci_vals
    _p_ci_cols = [
        "grade_stuff_lo", "grade_stuff_hi", "grade_command_lo", "grade_command_hi",
        "grade_durability_lo", "grade_durability_hi",
    ]
    if not _p_ci.empty:
        for _, r in _p_ci.iterrows():
            pid = int(r["player_id"])
            ci_vals = {c: int(r[c]) for c in _p_ci_cols if c in r.index and pd.notna(r.get(c))}
            if pid in p_grade_lookup:
                p_grade_lookup[pid].update(ci_vals)
            else:
                p_grade_lookup[pid] = ci_vals

    # Position lookup
    _roster = load_roster()
    pos_lookup: dict[int, str] = {}
    if not _roster.empty and "primary_position" in _roster.columns:
        for _, r in _roster.iterrows():
            pos_lookup[int(r["player_id"])] = r["primary_position"]
    _prospects = load_rankings("prospect")
    if not _prospects.empty and "primary_position" in _prospects.columns:
        for _, r in _prospects.iterrows():
            pid = int(r.get("player_id", r.get("batter_id", 0)))
            if pid and pid not in pos_lookup:
                pos_lookup[pid] = r["primary_position"]

    # Projection lookup with injected grades/diamonds
    proj_lookup = _build_projection_lookup()
    for pid, pinfo in proj_lookup.items():
        pinfo["tdd_value_score"] = diamond_lookup.get(pid)
        if pid in p_grade_lookup:
            pinfo.update(p_grade_lookup[pid])

    # Name lookup for game props resolution
    name_lookup: dict[int, str] = {}
    if not _h_proj.empty and "batter_name" in _h_proj.columns:
        for _, r in _h_proj.iterrows():
            name_lookup[int(r["batter_id"])] = r["batter_name"]
    _pp = load_projections("pitcher")
    if not _pp.empty and "pitcher_name" in _pp.columns:
        for _, r in _pp.iterrows():
            name_lookup[int(r["pitcher_id"])] = r["pitcher_name"]

    return {
        "h_arch_lookup": h_arch_lookup,
        "p_arch_lookup": p_arch_lookup,
        "h_stat_lookup": h_stat_lookup,
        "pos_lookup": pos_lookup,
        "proj_lookup": proj_lookup,
        "standings": standings,
        "name_lookup": name_lookup,
    }


# ---------------------------------------------------------------------------
# Props Lab
# ---------------------------------------------------------------------------

_STAT_LABELS = {"TB": "Total Bases", "K": "Strikeouts", "H": "Hits", "HRR": "H+R+RBI", "Outs": "Outs Recorded"}
_PITCHER_LABELS = {"K": "Pitcher Strikeouts", "H": "Hits Allowed", "HR": "HR Allowed", "BB": "Walks Issued", "Outs": "Outs Recorded"}
_HITTER_LABELS = {"K": "Batter Strikeouts", "H": "Batter Hits", "HR": "Batter Home Runs", "BB": "Batter Walks", "TB": "Total Bases", "HRR": "H+R+RBI"}
_LINE_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}


def _edge_color(p_over: float) -> str:
    """Return CSS color based on how far P(over) is from 0.5."""
    if p_over >= 0.60:
        return "var(--tdd-sage)"
    if p_over <= 0.40:
        return "var(--tdd-ember)"
    return "var(--tdd-slate)"


def _render_props_section(
    gpk: int,
    props_df: pd.DataFrame,
    lineups_df: pd.DataFrame | None = None,
    live_stats_df: pd.DataFrame | None = None,
) -> None:
    """Render projected performer edges for a single game.

    When confirmed lineups are available, only shows players who are in
    the starting lineup (pitchers always included).  When lineups have
    not been released yet, shows all projected players with a warning
    banner.
    """
    if props_df.empty:
        st.markdown(
            '<div class="tdd-meta">No projection data available for this game.</div>',
            unsafe_allow_html=True,
        )
        return

    game_df = props_df[props_df["game_pk"] == gpk].copy()
    if game_df.empty:
        st.markdown(
            '<div class="tdd-meta">No projection data available for this game.</div>',
            unsafe_allow_html=True,
        )
        return

    # Check if confirmed lineups exist for this game
    game_lu = (
        lineups_df[lineups_df["game_pk"] == gpk]
        if lineups_df is not None and not lineups_df.empty
        else pd.DataFrame()
    )
    lineup_confirmed = not game_lu.empty

    if lineup_confirmed:
        # Filter batters to only confirmed starters; keep all pitchers
        confirmed_pids = set(game_lu["batter_id"].astype(int))
        is_pitcher = game_df["player_type"] == "pitcher"
        is_in_lineup = game_df["player_id"].astype(int).isin(confirmed_pids)
        game_df = game_df[is_pitcher | is_in_lineup].copy()
    else:
        st.markdown(
            '<div class="tdd-callout">'
            'Estimated lineup -- probable lineup not yet released'
            '</div>',
            unsafe_allow_html=True,
        )

    if game_df.empty:
        st.markdown(
            '<div class="tdd-meta">No projection data for confirmed starters.</div>',
            unsafe_allow_html=True,
        )
        return

    # Results: prefer backfilled actuals already in game_props,
    # then overlay with live boxscore data for in-progress games.
    for col in ("actual", "over_hit", "game_status"):
        if col not in game_df.columns:
            game_df[col] = None

    game_live = (
        live_stats_df[live_stats_df["game_pk"] == gpk]
        if live_stats_df is not None and not live_stats_df.empty
        else pd.DataFrame()
    )
    if not game_live.empty:
        _STAT_TO_ACTUAL = {
            "K": "actual_K", "H": "actual_H", "HR": "actual_HR",
            "BB": "actual_BB", "TB": "actual_TB", "Outs": "actual_Outs",
        }
        live_lookup: dict[int, pd.Series] = {}
        for _, lr in game_live.iterrows():
            live_lookup[int(lr["player_id"])] = lr
        for idx, row in game_df.iterrows():
            pid = int(row["player_id"])
            if pid in live_lookup:
                lr = live_lookup[pid]
                stat = row["stat"]
                if stat == "HRR":
                    # Combined: Hits + Runs + RBIs
                    h = float(lr.get("actual_H", 0)) if pd.notna(lr.get("actual_H")) else 0
                    r = float(lr.get("actual_R", 0)) if pd.notna(lr.get("actual_R")) else 0
                    rbi = float(lr.get("actual_RBI", 0)) if pd.notna(lr.get("actual_RBI")) else 0
                    if any(c in lr.index for c in ("actual_H", "actual_R", "actual_RBI")):
                        actual = h + r + rbi
                        game_df.at[idx, "actual"] = actual
                        game_df.at[idx, "over_hit"] = actual > row["line"]
                else:
                    actual_col = _STAT_TO_ACTUAL.get(stat)
                    if actual_col and actual_col in lr.index and pd.notna(lr[actual_col]):
                        game_df.at[idx, "actual"] = float(lr[actual_col])
                        game_df.at[idx, "over_hit"] = float(lr[actual_col]) > row["line"]
                game_df.at[idx, "game_status"] = lr.get("game_status", "")

    # Compat: coalesce old (line_mid/p_over_mid) and new (line/p_over) columns
    if "line_mid" in game_df.columns:
        if "line" not in game_df.columns:
            game_df.rename(columns={"line_mid": "line", "p_over_mid": "p_over"},
                           inplace=True)
        else:
            game_df["line"] = game_df["line"].fillna(game_df["line_mid"])
            game_df["p_over"] = game_df["p_over"].fillna(game_df["p_over_mid"])

    # Over projections only -- P(over) >= 63%
    pop_df = game_df[game_df["p_over"] >= 0.63].sort_values("p_over", ascending=False)

    st.markdown(EXPANDABLE_CARD_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="tdd-props-header">Over Projections</div>',
        unsafe_allow_html=True,
    )
    if pop_df.empty:
        st.markdown(
            '<div class="tdd-meta" style="margin-bottom:0.5rem;">No strong over projections for this game.</div>',
            unsafe_allow_html=True,
        )
    else:
        cards_html = ""
        for _, row in pop_df.head(5).iterrows():
            cards_html += _prop_card_html(row)
        st.markdown(cards_html, unsafe_allow_html=True)


def _prop_card_html(row: pd.Series) -> str:
    """Build an expandable card for a single prop edge."""
    name = row["player_name"]
    stat = row["stat"]
    ptype = row.get("player_type", "")
    if ptype == "pitcher":
        stat_label = _PITCHER_LABELS.get(stat, _STAT_LABELS.get(stat, stat))
    else:
        stat_label = _HITTER_LABELS.get(stat, _STAT_LABELS.get(stat, stat))
    stat_short = stat  # K, H, HR, TB, BB
    team = row["team"]
    opp = row["opponent"]
    expected = row["expected"]
    p_over = row["p_over"]
    line = row["line"]
    type_badge = "P" if ptype == "pitcher" else "H"

    color = _edge_color(p_over)
    pct = p_over * 100

    # Format line as integer if whole number, else 1 decimal
    line_str = f"{line:.0f}" if line == int(line) else f"{line:.1f}"

    # Live result
    actual = row.get("actual")
    game_status = str(row.get("game_status", ""))
    is_final = "final" in game_status.lower() or "game over" in game_status.lower()
    result_html = ""
    if pd.notna(actual) and actual is not None:
        actual_val = float(actual)
        over_hit = actual_val > line
        if over_hit:
            result_html = (
                f'<span style="color:var(--tdd-sage); font-size:0.8rem; '
                f'font-weight:700; flex-shrink:0;">'
                f'\u2705 {actual_val:.0f} {stat_short}</span>'
            )
        elif is_final:
            result_html = (
                f'<span style="color:var(--tdd-ember); font-size:0.8rem; '
                f'flex-shrink:0; opacity:0.7;">'
                f'\u274c {actual_val:.0f} {stat_short}</span>'
            )
        else:
            result_html = (
                f'<span style="color:var(--tdd-slate); font-size:0.75rem; '
                f'flex-shrink:0;">'
                f'{actual_val:.0f} {stat_short}</span>'
            )

    # Summary: Name | Stat Expected | P(Over > line) = X% | result
    summary = (
        f'<span style="display:flex; align-items:center; gap:0.5rem; width:100%;">'
        # Type badge
        f'<span style="font-size:0.65rem; color:var(--tdd-slate); '
        f'border:1px solid var(--tdd-dark-border); border-radius:3px; '
        f'padding:0 0.25rem; flex-shrink:0;">{type_badge}</span>'
        # Name + team
        f'<span class="tdd-player-name" style="min-width:0; flex:1;">{esc(name)}'
        f'<span class="tdd-stat-label" style="margin-left:0.3rem;">{esc(team)} vs {esc(opp)}</span></span>'
        # Stat + expected
        f'<span style="color:var(--tdd-cream); font-size:0.8rem; flex-shrink:0;">'
        f'{stat_label} {expected:.2f}</span>'
        # P(Over > line) = X%
        f'<span style="color:{color}; font-size:0.75rem; font-weight:600; '
        f'flex-shrink:0; white-space:nowrap;">'
        f'P(Over &gt; {line_str}) = {pct:.0f}%</span>'
        # Live result
        f'{result_html}'
        f'</span>'
    )

    detail = (
        f'<div class="tdd-meta" style="margin-top:0.3rem;">'
        f'Expected: {expected:.2f} | Std Dev: {row["std"]:.2f}</div>'
    )

    return expandable_card_html(summary, detail)


# ---------------------------------------------------------------------------
# Scouting Report section
# ---------------------------------------------------------------------------

def _render_scouting_section(sides: list[dict], gpk: int) -> None:
    """Side-by-side pitcher scouting reports for a game."""
    scouting_data = _get_scouting_report_data()

    col_away, col_home = st.columns(2)
    for side_info, col in zip(sides, [col_away, col_home]):
        pid = side_info["pitcher_id"]
        pitcher_name = side_info["pitcher_name"]
        side_abbr = side_info["abbr"]
        opp_abbr = side_info["opp_abbr"]

        with col:
            st.markdown(
                f'<div class="tdd-section-hdr">'
                f'<span class="tdd-team-abbr" data-team="{side_abbr}">'
                f'{side_abbr}</span>'
                f' <span style="color:var(--tdd-slate); font-size:0.8rem;">'
                f'{pitcher_name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if not pid:
                st.markdown(
                    '<div class="tdd-meta">Starter TBD</div>',
                    unsafe_allow_html=True,
                )
                continue

            report = get_pitcher_scouting(
                pid, pitcher_name, side_abbr, opp_abbr,
                scouting_data,
            )
            _render_scouting_html(report)


# ---------------------------------------------------------------------------
# Game Drill-Down (Phase 2: Game Center)
# ---------------------------------------------------------------------------

@st.fragment
def _render_game_drilldown(
    game: pd.Series,
    lineups: pd.DataFrame,
    h_arch_lookup: dict[int, str],
    p_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    bf_priors: pd.DataFrame,
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    proj_lookup: dict[int, dict],
    gpk: int,
    pos_lookup: dict[int, str] | None = None,
    str_df: pd.DataFrame | None = None,
    game_props: pd.DataFrame | None = None,
    live_stats: pd.DataFrame | None = None,
    batter_sims_df: pd.DataFrame | None = None,
    pitcher_sim_samples: object | None = None,
    batter_sim_samples: object | None = None,
) -> None:
    """Rich game drill-down: lineup matchups, matchup analysis, and game simulator."""
    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty else pd.DataFrame()

    # Build per-side info
    sides = []
    for side, opp_side in [("away", "home"), ("home", "away")]:
        pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
        pitcher_id_raw = game.get(f"{side}_pitcher_id")
        pid = int(pitcher_id_raw) if pd.notna(pitcher_id_raw) else None
        side_team_id = game.get(f"{side}_team_id")
        opp_team_id = game.get(f"{opp_side}_team_id")
        opp_abbr = game.get(f"{opp_side}_abbr", "?")
        side_abbr = game.get(f"{side}_abbr", "?")
        opp_lu = (
            game_lu[game_lu["team_id"] == opp_team_id].sort_values("batting_order")
            if not game_lu.empty and pd.notna(opp_team_id) else pd.DataFrame()
        )
        own_lu = (
            game_lu[game_lu["team_id"] == side_team_id].sort_values("batting_order")
            if not game_lu.empty and pd.notna(side_team_id) else pd.DataFrame()
        )

        sides.append({
            "side": side,
            "abbr": side_abbr,
            "opp_abbr": opp_abbr,
            "pitcher_name": pitcher_name,
            "pitcher_id": pid,
            "pitcher_arch": p_arch_lookup.get(pid) if pid else None,
            "pitcher_proj": proj_lookup.get(pid, {}) if pid else {},
            "opp_lineup": opp_lu,
            "own_lineup": own_lu,
        })

    # --- Inline toolbar: Section ---
    _SECTIONS = ["Lineups", "Scouting Report", "Props Lab"]

    section = st.selectbox(
        "Section", _SECTIONS,
        key=f"dd_section_{gpk}",
        label_visibility="collapsed",
    )

    _batter_sims = batter_sims_df if batter_sims_df is not None else pd.DataFrame()

    # --- Render selected section ---
    if section == "Lineups":
        changes = _detect_lineup_changes(game_lu, _batter_sims, gpk)
        if changes["changed"]:
            n_new = len(changes["new_batters"])
            n_miss = len(changes["missing_batters"])
            parts = []
            if n_new:
                parts.append(f"{n_new} new batter(s)")
            if n_miss:
                parts.append(f"{n_miss} removed")
            tdd_info(f"Lineup changed since last sim. {', '.join(parts)}. Sims will auto-update shortly.")

        _render_matchup_tab_sidebyside(
            sides, h_arch_lookup, h_stat_lookup,
            arsenal_df, vuln_df, gpk,
            pos_lookup=pos_lookup or {},
            str_df=str_df if str_df is not None else pd.DataFrame(),
            bf_priors=bf_priors if bf_priors is not None else pd.DataFrame(),
            batter_sims_df=_batter_sims,
            pitcher_sim_samples=pitcher_sim_samples,
            batter_sim_samples=batter_sim_samples,
        )

    elif section == "Scouting Report":
        _render_scouting_section(sides, gpk)

    elif section == "Props Lab":
        _render_props_section(
            gpk,
            game_props if game_props is not None else pd.DataFrame(),
            lineups_df=game_lu,
            live_stats_df=live_stats,
        )


from components.sim_chart import (
    PITCHER_STAT_META as _PITCHER_STAT_META,
    BATTER_STAT_META as _BATTER_STAT_META,
    render_player_sim_from_props as _render_player_sim_from_props,
)


def _render_matchup_tab_sidebyside(
    sides: list[dict],
    h_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    gpk: int,
    pos_lookup: dict[int, str] | None = None,
    str_df: pd.DataFrame | None = None,
    bf_priors: pd.DataFrame | None = None,
    batter_sims_df: pd.DataFrame | None = None,
    pitcher_sim_samples: object | None = None,
    batter_sim_samples: object | None = None,
) -> None:
    """Lineup matchups for both sides rendered in side-by-side columns."""
    col_away, col_home = st.columns(2)
    for i, (side_info, col) in enumerate(zip(sides, [col_away, col_home])):
        opp_side = sides[1 - i]
        with col:
            _render_matchup_tab(
                [side_info], h_arch_lookup, h_stat_lookup,
                arsenal_df, vuln_df, gpk,
                pos_lookup=pos_lookup or {},
                opp_side=opp_side,
                str_df=str_df if str_df is not None else pd.DataFrame(),
                bf_priors=bf_priors if bf_priors is not None else pd.DataFrame(),
                batter_sims_df=batter_sims_df if batter_sims_df is not None else pd.DataFrame(),
                pitcher_sim_samples=pitcher_sim_samples,
                batter_sim_samples=batter_sim_samples,
            )


from components.grades import (
    grade_color as _grade_color,
    hitter_grades_html as _hitter_grades_html,
    pitcher_grades_html as _pitcher_grades_html,
    matchup_lift_badge_html as _matchup_lift_badge_html,
)


def _render_matchup_tab(
    sides: list[dict],
    h_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    gpk: int,
    pos_lookup: dict[int, str] | None = None,
    opp_side: dict | None = None,
    str_df: pd.DataFrame | None = None,
    bf_priors: pd.DataFrame | None = None,
    batter_sims_df: pd.DataFrame | None = None,
    pitcher_sim_samples: object | None = None,
    batter_sim_samples: object | None = None,
) -> None:
    """Team roster with inline matchup scouting on expand.

    When *opp_side* is provided (side-by-side view), each column shows:
      - Team header
      - Own SP at top
      - Own lineup with batter matchup badges vs opposing SP
      - Expanding a batter shows matchup scouting bullets + grades + game projections
      - Avg K matchup summary for own SP vs opposing lineup

    All sim results and matchup lifts come from precomputed data -- no live
    Monte Carlo or matchup scoring runs at render time.
    """
    from components.scouting import build_matchup_scouting_bullets, compute_matchup_xwoba_edge
    from services.data_loader import load_game_props

    if pos_lookup is None:
        pos_lookup = {}
    if bf_priors is None:
        bf_priors = pd.DataFrame()
    if batter_sims_df is None:
        batter_sims_df = pd.DataFrame()

    _gp_all = load_game_props()
    game_df = (
        _gp_all[_gp_all["game_pk"] == gpk] if not _gp_all.empty else pd.DataFrame()
    )

    for side_info in sides:
        pitcher_name = side_info["pitcher_name"]
        pid = side_info["pitcher_id"]
        p_arch = side_info["pitcher_arch"]
        p_proj = side_info["pitcher_proj"]
        opp_lu = side_info["opp_lineup"]
        opp_abbr = side_info["opp_abbr"]
        side_abbr = side_info["abbr"]

        # Team-centric view: own roster, batter matchups vs opposing SP
        if opp_side is not None:
            display_lu = side_info["own_lineup"]
            matchup_pid = opp_side["pitcher_id"]
        else:
            display_lu = opp_lu
            matchup_pid = pid

        # Section header
        st.markdown(
            f'<div class="tdd-section-hdr">'
            f'<span class="tdd-team-abbr" data-team="{side_abbr}">{side_abbr}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if display_lu.empty:
            st.markdown(
                f'<div style="color:var(--tdd-slate); font-size:0.85rem; padding:0.5rem 0;">'
                f'No lineup data available</div>',
                unsafe_allow_html=True,
            )
            continue

        # Detect roster-sourced lineup (no confirmed lineup set)
        _is_roster_lineup = (
            "lineup_source" in display_lu.columns
            and (display_lu["lineup_source"] == "roster").any()
        )
        if _is_roster_lineup:
            st.markdown(
                f'<div style="color:var(--tdd-slate); font-size:0.82rem; '
                f'padding:0.3rem 0; margin-bottom:0.3rem; font-style:italic;">'
                f'No probable lineup set -- showing active roster</div>',
                unsafe_allow_html=True,
            )

        name_col = "batter_name" if "batter_name" in display_lu.columns else "player_name"
        id_col = "batter_id" if "batter_id" in display_lu.columns else "player_id"

        total_k_lift = total_bb_lift = 0.0
        n_scored = 0

        # Pitcher card at top
        if pid:
            _p_hand_letter = ""
            _p_ars_hand = arsenal_df[arsenal_df["pitcher_id"] == pid] if not arsenal_df.empty else pd.DataFrame()
            if not _p_ars_hand.empty and "pitch_hand" in _p_ars_hand.columns:
                _h = str(_p_ars_hand["pitch_hand"].iloc[0]).strip().upper()
                if _h and _h[0] in ("L", "R"):
                    _p_hand_letter = _h[0]
            p_arch_tag = f" | {p_arch}" if p_arch else ""
            _p_label = f"SP  {pitcher_name} | {_p_hand_letter}{p_arch_tag}"

            with st.expander(_p_label):
                # Rich detail HTML
                p_composite = p_proj.get("tdd_value_score")
                p_diamond = diamond_rating_html(0, size="sm", precomputed=p_composite) if pd.notna(p_composite) else ""
                p_grades_html = _pitcher_grades_html(p_proj)
                _p_detail_parts = []
                if p_diamond:
                    _p_detail_parts.append(p_diamond)
                if p_grades_html:
                    _p_detail_parts.append(p_grades_html)
                if _p_detail_parts:
                    st.markdown(" ".join(_p_detail_parts), unsafe_allow_html=True)

                # Pitcher game sim — read from game_props (same source as Props Lab)
                if not game_df.empty and "player_id" in game_df.columns:
                    _p_rows = game_df[
                        (game_df["player_id"] == pid)
                        & (game_df.get("player_type", "pitcher") == "pitcher")
                    ]
                    if not _p_rows.empty:
                        _render_player_sim_from_props(
                            _p_rows,
                            _PITCHER_STAT_META,
                            ["K", "BB", "H", "HR", "Outs"],
                            f"p{pid}_{gpk}",
                        )

        _display_limit = len(display_lu) if _is_roster_lineup else 9
        for _, brow in display_lu.head(_display_limit).iterrows():
            bid = int(brow[id_col]) if pd.notna(brow.get(id_col)) else None
            bname = brow.get(name_col, "Unknown")
            order = int(brow["batting_order"])

            # Prefer game-day position (DH, etc.) over roster primary
            _game_pos = brow.get("game_position", "")
            if _game_pos and str(_game_pos).strip():
                pos = str(_game_pos).strip()
                # Pitcher in batting order = two-way player batting as DH
                if pos in ("P", "SP", "RP") and order <= 9:
                    pos = "DH"
            else:
                pos = pos_lookup.get(bid, "--") if bid else "--"
            arch = h_arch_lookup.get(bid, "Prospect") if bid else ""
            stats = h_stat_lookup.get(bid, {}) if bid else {}
            hand_letter = ""
            _bh = stats.get("bat_hand")
            if _bh and str(_bh).strip().upper()[0] in ("L", "R", "S"):
                hand_letter = str(_bh).strip().upper()[0]

            # Matchup advantage — from precomputed batter sims
            k_lift = bb_lift = hr_lift = 0.0
            _bsim_row = None
            if bid and not batter_sims_df.empty:
                _bs = batter_sims_df[
                    (batter_sims_df["game_pk"] == gpk)
                    & (batter_sims_df["batter_id"] == bid)
                ]
                if not _bs.empty:
                    _bsim_row = _bs.iloc[0]
                    k_lift = float(_bsim_row.get("matchup_k_lift", 0.0))
                    bb_lift = float(_bsim_row.get("matchup_bb_lift", 0.0))
                    hr_lift = float(_bsim_row.get("matchup_hr_lift", 0.0))
                    total_k_lift += k_lift
                    total_bb_lift += bb_lift
                    n_scored += 1

            # Expander label
            arch_tag = f" | {arch}" if arch else ""
            _b_label = f"{order}. {bname} | {pos} | {hand_letter}{arch_tag}"

            with st.expander(_b_label):
                # Rich HTML detail
                detail_parts: list[str] = []

                # Matchup edge via odds-ratio xwOBA
                _advantage_badge = ""
                _p_arsenal = pd.DataFrame()
                _h_vuln = pd.DataFrame()
                _h_str = pd.DataFrame()
                _pitcher_hand = None
                if matchup_pid and bid and not arsenal_df.empty and not vuln_df.empty:
                    _p_arsenal = arsenal_df[arsenal_df["pitcher_id"] == matchup_pid]
                    _h_vuln = vuln_df[vuln_df["batter_id"] == bid]
                    _h_str = (
                        str_df[str_df["batter_id"] == bid]
                        if str_df is not None and not str_df.empty
                        else pd.DataFrame()
                    )
                    if not _p_arsenal.empty and "pitch_hand" in _p_arsenal.columns:
                        _pitcher_hand = str(_p_arsenal["pitch_hand"].iloc[0])
                    _batter_hand = hand_letter or None

                    if not _p_arsenal.empty and not _h_vuln.empty:
                        _edge_result = compute_matchup_xwoba_edge(
                            _p_arsenal, _h_vuln, _h_str,
                            pitcher_hand=_pitcher_hand,
                            batter_hand=_batter_hand,
                        )
                        _adv_label = _edge_result["advantage"]
                        _edge_val = _edge_result["edge"]
                        _xw = _edge_result["matchup_xwoba"]
                        if _adv_label == "pitcher":
                            _advantage_badge = (
                                f'<span style="color:var(--tdd-ember); font-size:0.68rem; '
                                f'font-weight:600; margin-left:0.3rem;">Pitcher Edge'
                                f'<span style="font-weight:400; font-size:0.6rem; '
                                f'margin-left:0.2rem;">.{int(_xw*1000):03d} xwOBA</span></span>'
                            )
                        elif _adv_label == "hitter":
                            _advantage_badge = (
                                f'<span style="color:var(--tdd-sage); font-size:0.68rem; '
                                f'font-weight:600; margin-left:0.3rem;">Hitter Edge'
                                f'<span style="font-weight:400; font-size:0.6rem; '
                                f'margin-left:0.2rem;">.{int(_xw*1000):03d} xwOBA</span></span>'
                            )
                        else:
                            _advantage_badge = (
                                f'<span style="color:var(--tdd-slate); font-size:0.68rem; '
                                f'margin-left:0.3rem;">Even'
                                f'<span style="font-weight:400; font-size:0.6rem; '
                                f'margin-left:0.2rem;">.{int(_xw*1000):03d} xwOBA</span></span>'
                            )

                # Headshot + diamond + matchup badge
                composite = stats.get("tdd_value_score")
                _top_parts = []
                if bid:
                    _top_parts.append(headshot_html(bid, size=32))
                if pd.notna(composite):
                    _top_parts.append(diamond_rating_html(0, size="sm", precomputed=composite))
                if _advantage_badge:
                    _top_parts.append(_advantage_badge)
                if _top_parts:
                    detail_parts.append(
                        f'<div style="display:flex; align-items:center; gap:0.5rem;">'
                        + "".join(_top_parts) + '</div>'
                    )

                # Scouting bullets (descriptive + platoon info)
                if not _p_arsenal.empty and not _h_vuln.empty:
                    _opp_pname = opp_side["pitcher_name"] if opp_side else pitcher_name
                    _scouting_bullets = build_matchup_scouting_bullets(
                        _p_arsenal, _h_vuln, _h_str,
                        pitcher_name=_opp_pname, hitter_name=bname,
                        pitcher_hand=_pitcher_hand,
                        batter_hand=hand_letter or None,
                    )
                    if _scouting_bullets:
                        bullet_html = "".join(
                            f'<div style="color:{c}; font-size:0.78rem; margin:0.12rem 0; '
                            f'padding-left:0.65rem; border-left:2px solid {c};">{t}</div>'
                            for c, t in _scouting_bullets[:4]
                        )
                        detail_parts.append(bullet_html)

                # Grades
                grades_html = _hitter_grades_html(stats)
                if grades_html:
                    detail_parts.append(grades_html)

                if detail_parts:
                    st.markdown("".join(detail_parts), unsafe_allow_html=True)

                # Batter game sim — read from game_props (same source as Props Lab)
                if bid and not game_df.empty and "player_id" in game_df.columns:
                    _b_rows = game_df[
                        (game_df["player_id"] == bid)
                        & (game_df.get("player_type", "batter") == "batter")
                    ]
                    if not _b_rows.empty:
                        _render_player_sim_from_props(
                            _b_rows,
                            _BATTER_STAT_META,
                            ["H", "HR", "K", "BB"],
                            f"b{bid}_{gpk}",
                        )

        # Avg K matchup — from precomputed pitcher sims
        if n_scored > 0:
            avg_k = total_k_lift / n_scored
            avg_bb = total_bb_lift / n_scored
            if avg_k > 0.05:
                k_color, k_word = POSITIVE, "favorable"
            elif avg_k < -0.05:
                k_color, k_word = NEGATIVE, "unfavorable"
            else:
                k_color, k_word = SLATE, "neutral"
            st.markdown(
                f'<div style="font-size:0.82rem; margin-bottom:0.5rem;">'
                f'<span style="color:{k_color};">{pitcher_name} avg K matchup: {avg_k:+.3f} '
                f'({k_word})</span>'
                f' · <span style="color:var(--tdd-slate);">BB: {avg_bb:+.3f}</span></div>',
                unsafe_allow_html=True,
            )

        # Scouting report (new engine with creative narratives)
        if pid:
            _scouting_data = _get_scouting_report_data()
            report = get_pitcher_scouting(
                pid, pitcher_name, side_abbr, opp_abbr,
                _scouting_data,
            )
            _render_scouting_html(report)


def _parse_temp_bucket(temp_str: object) -> str:
    """Convert temperature string to weather bucket."""
    if not temp_str:
        return "warm"
    try:
        temp = int(temp_str)
    except (ValueError, TypeError):
        return "warm"
    if temp < 55:
        return "cold"
    if temp < 70:
        return "cool"
    if temp < 85:
        return "warm"
    return "hot"


def _parse_wind_category(wind_str: object) -> str:
    """Convert wind string to category."""
    if not wind_str:
        return "none"
    w = str(wind_str).lower()
    if "calm" in w or w.strip() == "":
        return "none"
    if "out" in w:
        return "out"
    if "in from" in w or "in," in w:
        return "in"
    if "l to r" in w or "r to l" in w:
        return "cross"
    return "none"


def _schedule_masthead_html(selected_date, n_games: int, n_hitters: int) -> str:
    """Shared masthead for layout A and B."""
    d = selected_date
    date_str = d.strftime("%a %b %d")
    year_str = d.strftime("%Y")
    return (
        f'<div class="sched-a-masthead">'
        f'<div class="sched-a-date">{date_str}'
        f'<span class="sched-a-date-year">&#183; {year_str}</span></div>'
        f'<div class="sched-a-stats">'
        f'<div><div class="sched-a-stat-v">{n_games}</div>'
        f'<div class="sched-a-stat-l">Games</div></div>'
        f'<div><div class="sched-a-stat-v">{n_hitters}</div>'
        f'<div class="sched-a-stat-l">Hitters</div></div>'
        f'</div></div>'
    )


def _pitcher_proj_line_html(
    gpk: int,
    pid: int | None,
    lookup: dict[tuple[int, int], str],
) -> str:
    """Return HTML snippet for a pitcher's projected stat line."""
    if pid is None:
        return ""
    line = lookup.get((gpk, pid))
    if not line:
        return ""
    return (
        f'<div style="color:var(--tdd-slate); font-size:0.65rem; '
        f'margin-top:0.1rem;">{esc(line)}</div>'
    )


def _render_layout_a(
    schedule: pd.DataFrame,
    sims: pd.DataFrame,
    lineups: pd.DataFrame,
    meta: dict,
    selected_date,
    batter_sims: pd.DataFrame,
) -> None:
    """Direction A: Home-First Mirror -- compact, scannable cards."""
    n_games = len(schedule)
    n_hitters = len(batter_sims) if not batter_sims.empty else 0

    st.markdown(_schedule_masthead_html(selected_date, n_games, n_hitters), unsafe_allow_html=True)

    if schedule.empty:
        tdd_info("No games scheduled for this date.")
        return

    # Sort chronologically
    if "game_datetime_utc" in schedule.columns:
        schedule = schedule.sort_values("game_datetime_utc", na_position="last")

    # Build hitter count per game_pk for badge
    hitters_per_game: dict[int, int] = {}
    if not batter_sims.empty:
        for gpk, grp in batter_sims.groupby("game_pk"):
            hitters_per_game[int(gpk)] = len(grp)

    # Lookups + drilldown data (lazy, shared across all games)
    _lookups = _build_schedule_lookups()

    # Build pitcher projected line lookup from game_props: (game_pk, pitcher_id) -> line str
    _pitcher_line_lookup: dict[tuple[int, int], str] = {}
    if not sims.empty and "player_type" in sims.columns:
        _p_props = sims[sims["player_type"] == "pitcher"]
        for (gpk_key, pid_key), grp in _p_props.groupby(["game_pk", "player_id"]):
            parts = {}
            ip_val = None
            for _, pr in grp.iterrows():
                stat = pr["stat"]
                exp = pr["expected"]
                if pd.notna(exp):
                    parts[stat] = exp
                if ip_val is None and pd.notna(pr.get("expected_ip")):
                    ip_val = pr["expected_ip"]
            line_parts = []
            if ip_val is not None:
                line_parts.append(f"{ip_val:.1f} IP")
            for s in ["K", "BB", "H", "HR"]:
                if s in parts:
                    line_parts.append(f"{parts[s]:.1f} {s}")
            if line_parts:
                _pitcher_line_lookup[(int(gpk_key), int(pid_key))] = ", ".join(line_parts)

    _drilldown_data: dict | None = None

    def _get_dd() -> dict:
        nonlocal _drilldown_data
        if _drilldown_data is not None:
            return _drilldown_data
        _game_props = load_game_props()
        if not _game_props.empty:
            _name_lookup = dict(_lookups["name_lookup"])
            if not lineups.empty and "batter_name" in lineups.columns:
                for _, _r in lineups.iterrows():
                    pid = int(_r.get("batter_id", 0))
                    if pid and pid not in _name_lookup:
                        _name_lookup[pid] = _r["batter_name"]
            _game_props["player_name"] = _game_props["player_id"].map(
                lambda pid: _name_lookup.get(int(pid), str(pid))
            )
        _drilldown_data = {
            "bf_priors": load_bf_priors(),
            "arsenal_df": load_pitcher_arsenal(),
            "vuln_df": load_hitter_vulnerability(career=True),
            "str_df": load_hitter_strength(career=True),
            "batter_sims_df": load_todays_batter_sims(),
            "pitcher_sim_samples": load_pitcher_game_sim_samples(),
            "batter_sim_samples": load_batter_game_sim_samples(),
            "game_props": _game_props,
        }
        return _drilldown_data

    # Layout A: styled game cards with button-driven drilldown
    st.markdown(
        '<style>'
        '[data-testid="stButton"] button {'
        '  background: transparent !important;'
        '  border: 1px solid var(--tdd-dark-border) !important;'
        '  border-top: none !important;'
        '  border-radius: 0 0 8px 8px !important;'
        '  margin-top: -0.5rem !important;'
        '  padding: 0.3rem 1rem !important;'
        '  font-size: 0.7rem !important;'
        '  color: var(--tdd-slate) !important;'
        '  font-weight: 500 !important;'
        '  letter-spacing: 0.3px !important;'
        '  transition: border-color 0.15s, color 0.15s !important;'
        '}'
        '[data-testid="stButton"] button:hover {'
        '  border-color: var(--tdd-gold) !important;'
        '  color: var(--tdd-gold) !important;'
        '  background: rgba(255,255,255,0.03) !important;'
        '}'
        '</style>',
        unsafe_allow_html=True,
    )

    # Track which games are expanded
    if "sched_a_open" not in st.session_state:
        st.session_state["sched_a_open"] = set()

    for _, game in schedule.iterrows():
        gpk = int(game["game_pk"])
        away = game.get("away_abbr", "?")
        home = game.get("home_abbr", "?")
        away_name = game.get("away_team_name", away)
        home_name = game.get("home_team_name", home)
        time_str = format_game_time(
            game.get("game_datetime_utc"), fallback=game.get("game_time", ""),
        )
        status = game.get("status", "")
        away_sp = game.get("away_pitcher_name", "") or "TBD"
        home_sp = game.get("home_pitcher_name", "") or "TBD"
        away_pid_raw = game.get("away_pitcher_id")
        home_pid_raw = game.get("home_pitcher_id")
        away_pid = int(away_pid_raw) if pd.notna(away_pid_raw) else None
        home_pid = int(home_pid_raw) if pd.notna(home_pid_raw) else None
        nh = hitters_per_game.get(gpk, 0)
        is_open = gpk in st.session_state["sched_a_open"]

        # Status badge
        if "Progress" in status or "Live" in status:
            status_html = '<span style="color:var(--tdd-sage); font-weight:700;">&#9679; LIVE</span>'
        elif "Final" in status:
            status_html = '<span style="color:var(--tdd-slate);">FINAL</span>'
        else:
            status_html = f'<span style="color:var(--tdd-slate);">{esc(time_str)}</span>'

        nh_html = f'<span style="color:var(--tdd-gold); font-size:0.65rem;">{nh} hitters</span>' if nh else ""

        border_color = "var(--tdd-gold)" if is_open else "var(--tdd-dark-border)"
        card_html = (
            f'<div style="border:1px solid {border_color}; border-bottom:none; '
            f'border-radius:8px 8px 0 0; padding:0.7rem 1rem 0.5rem; '
            f'background:rgba(255,255,255,0.015); display:block !important;">'
            # Row 1: teams + time
            f'<div style="display:flex; justify-content:space-between; align-items:baseline;">'
            f'<div style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span data-team="{esc_attr(away)}" style="font-family:var(--tdd-font-heading); '
            f'font-weight:800; font-size:1.05rem;">{esc(away)}</span>'
            f' <span style="color:var(--tdd-slate); font-size:0.72rem;">{esc(away_name)}</span>'
            f' <span style="color:var(--tdd-slate); font-size:0.72rem;">@</span> '
            f'<span data-team="{esc_attr(home)}" style="font-family:var(--tdd-font-heading); '
            f'font-weight:800; font-size:1.05rem;">{esc(home)}</span>'
            f' <span style="color:var(--tdd-slate); font-size:0.72rem;">{esc(home_name)}</span>'
            f'</div>'
            f'<div style="flex-shrink:0; text-align:right; margin-left:1rem;">'
            f'<div style="font-size:0.65rem;">{status_html}</div>'
            f'{nh_html}'
            f'</div>'
            f'</div>'
            # Row 2: pitchers with projected lines
            f'<div style="display:flex; justify-content:space-between; margin-top:0.3rem; gap:1rem;">'
            # Away pitcher
            f'<div style="flex:1; min-width:0;">'
            f'<div style="color:var(--tdd-cream); font-size:0.75rem; font-weight:600;">{esc(away_sp)}</div>'
            f'{_pitcher_proj_line_html(gpk, away_pid, _pitcher_line_lookup)}'
            f'</div>'
            # Home pitcher
            f'<div style="flex:1; min-width:0; text-align:right;">'
            f'<div style="color:var(--tdd-cream); font-size:0.75rem; font-weight:600;">{esc(home_sp)}</div>'
            f'{_pitcher_proj_line_html(gpk, home_pid, _pitcher_line_lookup)}'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Toggle button fused to bottom of card
        if st.button(
            f"{'Hide' if is_open else 'Show'} Matchups & Projections",
            key=f"a_btn_{gpk}",
            use_container_width=True,
        ):
            if is_open:
                st.session_state["sched_a_open"].discard(gpk)
            else:
                st.session_state["sched_a_open"].add(gpk)
            st.rerun()

        # Drilldown content when open
        if is_open:
            st.markdown(
                f'<a href="?page=game_analysis&game_pk={gpk}" target="_self" '
                f'style="color:var(--tdd-gold); font-size:0.75rem; '
                f'text-decoration:none; font-weight:600; display:inline-block; '
                f'margin-bottom:0.5rem;">Full Analysis &#8594;</a>',
                unsafe_allow_html=True,
            )
            _dd = _get_dd()
            _render_game_drilldown(
                game, lineups,
                _lookups["h_arch_lookup"], _lookups["p_arch_lookup"],
                _lookups["h_stat_lookup"],
                _dd["bf_priors"], _dd["arsenal_df"], _dd["vuln_df"],
                _lookups["proj_lookup"], gpk,
                pos_lookup=_lookups["pos_lookup"],
                str_df=_dd["str_df"],
                game_props=_dd["game_props"],
                batter_sims_df=_dd["batter_sims_df"],
                pitcher_sim_samples=_dd["pitcher_sim_samples"],
                batter_sim_samples=_dd["batter_sim_samples"],
            )


def page_schedule() -> None:
    """Schedule page | browse games by date with projections."""
    # Use ET date as "today" (MLB games are scheduled in ET)
    meta = load_update_metadata()

    # Freshness badge
    _last = meta.get("last_updated", "") if meta else ""
    if _last:
        try:
            from datetime import datetime as _dt
            _ts = _dt.fromisoformat(_last.replace("Z", "+00:00"))
            _ago = (datetime.now(timezone.utc) - _ts).total_seconds() / 3600
            _fresh_color = "var(--tdd-sage)" if _ago < 6 else "var(--tdd-gold)" if _ago < 24 else "var(--tdd-ember)"
            st.markdown(
                f'<div class="tdd-meta" style="margin-bottom:0.5rem;">'
                f'<span style="color:{_fresh_color};">Data updated {_ago:.0f}h ago</span></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)  # EDT during baseball season
    today = et_now.date()

    # Date range: 7 days back through 7 days forward
    dates = [today + timedelta(days=d) for d in range(-7, 8)]
    date_labels: list[str] = []
    default_idx = 0
    for i, d in enumerate(dates):
        label = d.strftime("%a, %b %d")
        if d == today:
            label += "  (Today)"
            default_idx = i
        date_labels.append(label)

    selected_label = st.selectbox(
        "Game Date", date_labels, index=default_idx,
        key="schedule_date", label_visibility="collapsed",
    )
    selected_date = dates[date_labels.index(selected_label)]
    is_today = selected_date == today

    tomorrow = today + timedelta(days=1)
    is_tomorrow = selected_date == tomorrow
    cached_all = load_todays_games()
    cached_dates = (
        set(cached_all["game_date"].astype(str).unique())
        if not cached_all.empty and "game_date" in cached_all.columns
        else set()
    )
    cache_has_today = today.isoformat() in cached_dates
    cache_has_tomorrow = tomorrow.isoformat() in cached_dates

    if is_today and cache_has_today:
        schedule = cached_all[cached_all["game_date"] == today.isoformat()].copy()
        sims = load_game_props()
        lineups = load_todays_lineups()
        if not lineups.empty and "game_pk" in lineups.columns and not schedule.empty:
            lineups = lineups[lineups["game_pk"].isin(schedule["game_pk"])]
        lineups = backfill_missing_lineups(schedule, lineups)
        # Overlay live game status from MLB API so cards reflect
        # current state (In Progress / Final) between refreshes
        try:
            live_sched = fetch_live_schedule(today.isoformat())
            if (
                not live_sched.empty
                and "game_pk" in live_sched.columns
                and "status" in live_sched.columns
            ):
                live_status = live_sched.set_index("game_pk")["status"]
                schedule["status"] = (
                    schedule["game_pk"]
                    .map(live_status)
                    .fillna(schedule["status"])
                )
        except Exception:
            pass  # keep cached status on API failure
    elif is_tomorrow and cache_has_tomorrow:
        schedule = cached_all[cached_all["game_date"] == tomorrow.isoformat()].copy()
        sims = load_game_props()
        lineups = load_todays_lineups()
        if not lineups.empty and "game_pk" in lineups.columns and not schedule.empty:
            lineups = lineups[lineups["game_pk"].isin(schedule["game_pk"])]
        lineups = backfill_missing_lineups(schedule, lineups)
    elif is_today:
        # Parquets are stale (e.g. past midnight), fetch live
        schedule = fetch_live_schedule(today.isoformat())
        lineups = (
            fetch_live_lineups(schedule)
            if not schedule.empty
            else pd.DataFrame()
        )
        sims = load_game_props()
    else:
        # Other dates: fetch schedule from MLB API, no sims
        schedule = fetch_live_schedule(selected_date.isoformat())
        lineups = fetch_live_lineups(schedule) if not schedule.empty else pd.DataFrame()
        sims = pd.DataFrame()

    batter_sims = load_todays_batter_sims()

    _render_layout_a(schedule, sims, lineups, meta, selected_date, batter_sims)
