"""Schedule page | date-based game browser with projections and live data."""
from __future__ import annotations

import html
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK, DARK_CARD, DARK_BORDER,
    POSITIVE, NEGATIVE, DASHBOARD_DIR, PRIOR_SEASON, TRAINING_RANGE,
    PITCH_DISPLAY,
    SCHEDULE_REFRESH_MINUTES, GAME_WINDOW_START_HOUR, GAME_WINDOW_END_HOUR,
)
from services.data_loader import (
    load_todays_games, load_todays_lineups, load_todays_batter_sims,
    load_update_metadata, load_pitcher_arsenal, load_hitter_vulnerability,
    load_hitter_strength,
    load_projections, load_counting, load_game_info, load_player_teams,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_k_samples, load_bb_samples, load_hr_samples, load_bf_priors,
    load_hitter_k_samples, load_hitter_bb_samples, load_hitter_hr_samples,
    load_exit_model, load_pitcher_pitch_count_features,
    load_batter_pitch_count_features, load_tto_profiles,
    load_pitcher_exit_tendencies,
    load_pitcher_location_grid, load_hitter_zone_grid,
    load_roster, load_game_props,
    fetch_live_schedule, fetch_live_lineups, fetch_live_boxscores,
    backfill_missing_lineups,
)
from utils.helpers import format_ip, get_team_lookup
from components.charts import _STAT_CHART_CONFIG
from components.diamond_rating import diamond_rating_html
from components.expandable_card import EXPANDABLE_CARD_CSS, expandable_card_html
from components.team_logo import team_logo_html
from components.headshot import headshot_html


def _is_game_window() -> bool:
    """Check if current ET time is within the game window."""
    # Approximate ET as UTC-4 (EDT during baseball season)
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    return GAME_WINDOW_START_HOUR <= et_now.hour < GAME_WINDOW_END_HOUR


def _staleness_indicator(ts_str: str | None) -> str:
    """Return colored staleness indicator HTML based on timestamp age."""
    if not ts_str:
        return f'<span style="color:var(--tdd-slate);">unknown</span>'
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        minutes = age.total_seconds() / 60
        if minutes < 15:
            color = SAGE
        elif minutes < 60:
            color = GOLD
        else:
            color = EMBER
        return (
            f'<span style="color:{color};">●</span> '
            f'{ts.strftime("%I:%M %p")}'
        )
    except Exception:
        return f'<span style="color:var(--tdd-slate);">unknown</span>'


_SCHEDULE_CSS = ""


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


def _render_schedule_cards(
    schedule: pd.DataFrame,
    sims: pd.DataFrame,
    lineups: pd.DataFrame,
    meta: dict,
    *,
    live_stats: pd.DataFrame | None = None,
) -> None:
    """Render game cards from schedule + sim + lineup data."""
    # Pre-build archetype + projection lookups for enrichment
    _h_arch = load_hitter_archetypes()
    _p_arch = load_pitcher_archetypes()
    _h_proj = load_projections("hitter")
    _h_count = load_counting("hitter")

    # Hitter archetype lookup: batter_id → archetype_name
    _h_arch_lookup: dict[int, str] = {}
    if not _h_arch.empty:
        for _, _r in _h_arch.iterrows():
            _h_arch_lookup[int(_r["batter_id"])] = _r["archetype_name"]

    # Pitcher archetype lookup: pitcher_id → archetype_name
    _p_arch_lookup: dict[int, str] = {}
    if not _p_arch.empty:
        for _, _r in _p_arch.iterrows():
            _p_arch_lookup[int(_r["pitcher_id"])] = _r["archetype_name"]

    # Standings lookup: abbreviation → (wins, losses)
    from services.data_loader import load_standings
    _standings = load_standings()

    # Diamond rating lookup — always derived from tdd_value_score via score_to_diamonds
    from services.data_loader import load_rankings
    from lib.diamond_rating import score_to_diamonds
    _h_rankings = load_rankings("hitters")
    _p_rankings = load_rankings("pitchers")
    _diamond_lookup: dict[int, float] = {}
    if not _h_rankings.empty and "tdd_value_score" in _h_rankings.columns:
        for _, _r in _h_rankings.iterrows():
            if pd.notna(_r.get("tdd_value_score")):
                _diamond_lookup[int(_r["batter_id"])] = score_to_diamonds(_r["tdd_value_score"])
    if not _p_rankings.empty and "tdd_value_score" in _p_rankings.columns:
        for _, _r in _p_rankings.iterrows():
            if pd.notna(_r.get("tdd_value_score")):
                _diamond_lookup[int(_r["pitcher_id"])] = score_to_diamonds(_r["tdd_value_score"])

    # Hitter projection lookup: batter_id → {k_rate, bb_rate, hr, ...}
    _h_stat_lookup: dict[int, dict] = {}
    if not _h_proj.empty:
        for _, _r in _h_proj.iterrows():
            bid = int(_r["batter_id"])
            _h_stat_lookup[bid] = {
                "k_rate": _r.get("projected_k_rate"),
                "bb_rate": _r.get("projected_bb_rate"),
                "tdd_value_score": _diamond_lookup.get(bid),
            }
    if not _h_count.empty and "batter_id" in _h_count.columns:
        for _, _r in _h_count.iterrows():
            bid = int(_r["batter_id"])
            counting = {
                "hr": _r.get("total_hr_mean"),
                "total_k": _r.get("total_k_mean"),
                "total_bb": _r.get("total_bb_mean"),
                "total_hr": _r.get("total_hr_mean"),
            }
            if bid in _h_stat_lookup:
                _h_stat_lookup[bid].update(counting)
            else:
                _h_stat_lookup[bid] = {
                    "k_rate": None, "bb_rate": None,
                    "tdd_value_score": _diamond_lookup.get(bid),
                    **counting,
                }

    # Inject scouting grades from rankings into stat lookups
    _h_grade_cols = ["grade_hit", "grade_power", "grade_speed", "grade_discipline", "grade_fielding"]
    if not _h_rankings.empty:
        for _, _r in _h_rankings.iterrows():
            bid = int(_r["batter_id"])
            grades = {c: int(_r[c]) for c in _h_grade_cols if c in _r.index and pd.notna(_r.get(c))}
            if bid in _h_stat_lookup:
                _h_stat_lookup[bid].update(grades)
            else:
                _h_stat_lookup[bid] = {
                    "k_rate": None, "bb_rate": None,
                    "tdd_value_score": _diamond_lookup.get(bid),
                    **grades,
                }

    _p_grade_lookup: dict[int, dict] = {}
    _p_grade_cols = ["grade_stuff", "grade_command", "grade_durability"]
    if not _p_rankings.empty:
        for _, _r in _p_rankings.iterrows():
            pid = int(_r["pitcher_id"])
            grades = {c: int(_r[c]) for c in _p_grade_cols if c in _r.index and pd.notna(_r.get(c))}
            grades["tdd_value_score"] = _diamond_lookup.get(pid)
            _p_grade_lookup[pid] = grades

    # Inject grade confidence intervals into lookups
    from services.data_loader import load_hitter_grade_ci, load_pitcher_grade_ci
    _h_ci = load_hitter_grade_ci()
    _p_ci = load_pitcher_grade_ci()
    _h_ci_cols = ["grade_hit_lo", "grade_hit_hi", "grade_power_lo", "grade_power_hi",
                  "grade_speed_lo", "grade_speed_hi", "grade_discipline_lo", "grade_discipline_hi",
                  "grade_fielding_lo", "grade_fielding_hi"]
    if not _h_ci.empty:
        for _, _r in _h_ci.iterrows():
            bid = int(_r["player_id"])
            ci_vals = {c: int(_r[c]) for c in _h_ci_cols if c in _r.index and pd.notna(_r.get(c))}
            if bid in _h_stat_lookup:
                _h_stat_lookup[bid].update(ci_vals)
            else:
                _h_stat_lookup[bid] = ci_vals
    _p_ci_cols = ["grade_stuff_lo", "grade_stuff_hi", "grade_command_lo", "grade_command_hi",
                  "grade_durability_lo", "grade_durability_hi"]
    if not _p_ci.empty:
        for _, _r in _p_ci.iterrows():
            pid = int(_r["player_id"])
            ci_vals = {c: int(_r[c]) for c in _p_ci_cols if c in _r.index and pd.notna(_r.get(c))}
            if pid in _p_grade_lookup:
                _p_grade_lookup[pid].update(ci_vals)
            else:
                _p_grade_lookup[pid] = ci_vals

    # Position lookup from roster + prospect rankings
    _roster = load_roster()
    _pos_lookup: dict[int, str] = {}
    if not _roster.empty and "primary_position" in _roster.columns:
        for _, _r in _roster.iterrows():
            _pos_lookup[int(_r["player_id"])] = _r["primary_position"]
    # Fill gaps from prospect data
    from services.data_loader import load_rankings
    _prospects = load_rankings("prospect")
    if not _prospects.empty and "primary_position" in _prospects.columns:
        for _, _r in _prospects.iterrows():
            pid = int(_r.get("player_id", _r.get("batter_id", 0)))
            if pid and pid not in _pos_lookup:
                _pos_lookup[pid] = _r["primary_position"]

    # Game simulator + matchup scoring data
    _k_samples_dict = load_k_samples()
    _bb_samples_dict = load_bb_samples()
    _hr_samples_dict = load_hr_samples()
    _bf_priors = load_bf_priors()
    _arsenal_df = load_pitcher_arsenal()
    _vuln_df = load_hitter_vulnerability(career=True)
    _str_df = load_hitter_strength(career=True)
    _ploc_df = load_pitcher_location_grid()
    _hzone_df = load_hitter_zone_grid(career=True)

    # Hitter posterior rate samples (for batter game sim)
    _hitter_k_samples = load_hitter_k_samples()
    _hitter_bb_samples = load_hitter_bb_samples()
    _hitter_hr_samples = load_hitter_hr_samples()

    # Game sim v2 component data
    _exit_model = load_exit_model()
    _pitcher_pc = load_pitcher_pitch_count_features()
    _batter_pc = load_batter_pitch_count_features()
    _tto_profiles = load_tto_profiles()
    _exit_tendencies = load_pitcher_exit_tendencies()

    # Precomputed batter game sims (10K sims from daily pipeline)
    _batter_sims_df = load_todays_batter_sims()

    # Game props (player projection edges)
    _game_props = load_game_props()
    # Resolve numeric player_name values using projection data
    if not _game_props.empty:
        _name_lookup: dict[int, str] = {}
        _hp = load_projections("hitter")
        if not _hp.empty and "batter_name" in _hp.columns:
            for _, _r in _hp.iterrows():
                _name_lookup[int(_r["batter_id"])] = _r["batter_name"]
        _pp = load_projections("pitcher")
        if not _pp.empty and "pitcher_name" in _pp.columns:
            for _, _r in _pp.iterrows():
                _name_lookup[int(_r["pitcher_id"])] = _r["pitcher_name"]
        # Also pull names from lineups (covers anyone missing from projections)
        if not lineups.empty and "batter_name" in lineups.columns:
            for _, _r in lineups.iterrows():
                pid = int(_r.get("batter_id", 0))
                if pid and pid not in _name_lookup:
                    _name_lookup[pid] = _r["batter_name"]
        _game_props["player_name"] = _game_props["player_id"].map(
            lambda pid: _name_lookup.get(int(pid), str(pid))
        )

    # Always build projection lookup (used by cards + drilldown)
    proj_lookup = _build_projection_lookup()
    # Inject accurate tdd_value_score + scouting grades into pitcher proj lookup
    for pid, pinfo in proj_lookup.items():
        pinfo["tdd_value_score"] = _diamond_lookup.get(pid)
        if pid in _p_grade_lookup:
            pinfo.update(_p_grade_lookup[pid])

    if schedule.empty:
        st.info("No games scheduled for this date.")
        return

    n_games = len(schedule)

    # Check if sims are stale (different date than schedule)
    sims_stale = False
    if not sims.empty and "game_pk" in sims.columns:
        sims_gpks = set(sims["game_pk"].tolist())
        sched_gpks = set(schedule["game_pk"].tolist())
        sims_stale = len(sims_gpks & sched_gpks) == 0

    if sims_stale:
        st.markdown(
            f'<div style="color:var(--tdd-ember); font-size:0.85rem; margin-bottom:0.5rem;">'
            f'Simulations are from a previous date | showing base projections only.'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="color:var(--tdd-slate); font-size:0.9rem; margin-bottom:1rem;">'
        f'{n_games} games</div>',
        unsafe_allow_html=True,
    )

    for _, game in schedule.iterrows():
        gpk = game["game_pk"]
        away_abbr = game.get("away_abbr", "?")
        home_abbr = game.get("home_abbr", "?")
        away_tid = game.get("away_team_id")
        home_tid = game.get("home_team_id")
        game_time = game.get("game_time", "")
        game_dt = game.get("game_date", "")
        status = game.get("status", "")

        # game_props has per-stat rows; pivot to get expected_k / expected_ip per side
        game_sims = sims[(sims["game_pk"] == gpk) & (sims["player_type"] == "pitcher")] if not sims.empty else pd.DataFrame()

        def _extract_pitcher_sim(side_label: str) -> dict | None:
            side_rows = game_sims[game_sims["side"] == side_label] if not game_sims.empty else pd.DataFrame()
            if side_rows.empty:
                return None
            k_row = side_rows[side_rows["stat"] == "K"]
            outs_row = side_rows[side_rows["stat"] == "Outs"]
            result = {}
            if not k_row.empty:
                result["expected_k"] = float(k_row.iloc[0]["expected"])
            if not outs_row.empty:
                result["expected_ip"] = float(outs_row.iloc[0]["expected"]) / 3.0
            # Pass through metadata from any row
            first = side_rows.iloc[0]
            for col in ("dk_mean", "espn_mean", "umpire_k_lift", "weather_k_lift",
                         "expected_ip", "expected_pitches", "expected_bf"):
                if col in first.index and pd.notna(first.get(col)):
                    result.setdefault(col, first[col])
            return result if result else None

        away_sim = _extract_pitcher_sim("away")
        home_sim = _extract_pitcher_sim("home")

        # Status badge
        status_badge = ""
        if status:
            if status in ("In Progress", "Live"):
                status_badge = f'<span style="color:var(--tdd-sage); font-size:0.75rem; font-weight:600;">● LIVE</span>'
            elif "Final" in status:
                status_badge = f'<span style="color:var(--tdd-slate); font-size:0.75rem;">{status}</span>'
            elif "Scheduled" not in status:
                status_badge = f'<span style="color:var(--tdd-slate); font-size:0.75rem;">{status}</span>'

        # Game context
        venue_name = game.get("venue_name", "")
        hp_ump = game.get("hp_umpire_name", "")
        wx_temp = game.get("weather_temp", "")
        wx_cond = game.get("weather_condition", "")
        ctx_parts = []
        if venue_name:
            ctx_parts.append(venue_name)
        if hp_ump:
            ctx_parts.append(f"HP: {hp_ump}")
        if wx_temp:
            wx_str = f"{wx_temp}°F"
            if wx_cond:
                wx_str += f", {wx_cond}"
            ctx_parts.append(wx_str)

        # Build condensed pitcher summary for each side
        def _pitcher_summary(sim, pitcher_name_field, pitcher_id_field, abbr):
            pp_name = game.get(pitcher_name_field, "TBD") or "TBD"
            _pp_id = game.get(pitcher_id_field)
            _pp_arch = _p_arch_lookup.get(int(_pp_id)) if pd.notna(_pp_id) else None
            arch_tag = f'<span class="tdd-stat-label"> · {_pp_arch}</span>' if _pp_arch else ""

            if sim is not None:
                exp_k = sim.get("expected_k")
                exp_ip = sim.get("expected_ip")
                stats = f'E[K] {exp_k:.1f}' if exp_k is not None else ""
                if exp_ip is not None:
                    ip_str = format_ip(exp_ip)
                    stats += f' · IP {ip_str}' if stats else f'IP {ip_str}'
                return (
                    f'<span class="tdd-player-name">{pp_name}</span>'
                    f'{arch_tag}'
                    f'<span class="tdd-stat-label" style="margin-left:0.4rem;">{stats}</span>'
                )
            else:
                pid = game.get(pitcher_id_field)
                proj_info = proj_lookup.get(int(pid)) if pd.notna(pid) else None
                k_str = ""
                if proj_info and pd.notna(proj_info.get("projected_k_rate")):
                    k_str = f'K% {proj_info["projected_k_rate"]*100:.1f}%'
                return (
                    f'<span class="tdd-player-name">{pp_name}</span>'
                    f'{arch_tag}'
                    f'<span class="tdd-stat-label" style="margin-left:0.4rem;">{k_str}</span>'
                )

        away_sp_html = _pitcher_summary(away_sim, "away_pitcher_name", "away_pitcher_id", away_abbr)
        home_sp_html = _pitcher_summary(home_sim, "home_pitcher_name", "home_pitcher_id", home_abbr)

        # Team logos
        away_logo = team_logo_html(int(away_tid), size=80) if pd.notna(away_tid) else ""
        home_logo = team_logo_html(int(home_tid), size=80) if pd.notna(home_tid) else ""

        # Build game card header HTML
        ctx_html = (
            f'<div class="tdd-meta" style="margin-top:0.3rem;">'
            f'{" · ".join(ctx_parts)}</div>'
        ) if ctx_parts else ""

        # Date/time/status line
        dt_parts = []
        if game_dt:
            dt_parts.append(game_dt)
        if game_time:
            dt_parts.append(game_time)
        dt_line = " · ".join(dt_parts)
        if status_badge:
            dt_line += f'  {status_badge}'

        # Game card with logos
        card_html = (
            f'<div class="tdd-game-card">'
            # Date/time/status row
            f'<div class="tdd-meta" style="margin-bottom:0.4rem;">{dt_line}</div>'
            # Team row: logos + abbreviations
            f'<div class="tdd-game-teams" style="display:flex; align-items:center; gap:0.6rem;">'
            f'{away_logo}'
            f'<span class="tdd-team-abbr" data-team="{away_abbr}">{away_abbr}</span>'
            f'{f"""<span class="tdd-meta" style="font-size:0.72rem; margin-left:-0.3rem;">{_standings[away_abbr][0]}-{_standings[away_abbr][1]}</span>""" if away_abbr in _standings else ""}'
            f'<span style="color:var(--tdd-slate); font-size:0.9rem; margin:0 0.3rem;">@</span>'
            f'<span class="tdd-team-abbr" data-team="{home_abbr}">{home_abbr}</span>'
            f'{f"""<span class="tdd-meta" style="font-size:0.72rem; margin-left:-0.3rem;">{_standings[home_abbr][0]}-{_standings[home_abbr][1]}</span>""" if home_abbr in _standings else ""}'
            f'{home_logo}'
            f'</div>'
            # Pitcher summaries
            f'<div style="display:flex; gap:2rem; margin-top:0.5rem;">'
            f'<div>{away_sp_html}</div>'
            f'<div>{home_sp_html}</div>'
            f'</div>'
            f'{ctx_html}'
            f'</div>'
        )

        st.markdown(card_html, unsafe_allow_html=True)

        with st.expander("View Matchups & Projections"):
            _render_game_drilldown(
                game, lineups, _h_arch_lookup, _p_arch_lookup, _h_stat_lookup,
                _k_samples_dict, _bf_priors, _arsenal_df, _vuln_df,
                proj_lookup, gpk,
                bb_samples_dict=_bb_samples_dict,
                hr_samples_dict=_hr_samples_dict,
                exit_model=_exit_model,
                pitcher_pc=_pitcher_pc,
                batter_pc=_batter_pc,
                tto_profiles=_tto_profiles,
                exit_tendencies=_exit_tendencies,
                pos_lookup=_pos_lookup,
                hitter_k_samples=_hitter_k_samples,
                hitter_bb_samples=_hitter_bb_samples,
                hitter_hr_samples=_hitter_hr_samples,
                str_df=_str_df,
                ploc_df=_ploc_df,
                hzone_df=_hzone_df,
                game_props=_game_props,
                live_stats=live_stats,
                batter_sims_df=_batter_sims_df,
            )

    if not sims.empty and not sims_stale:
        pitcher_sims = sims[(sims["player_type"] == "pitcher") & (sims["stat"] == "K")]
        n_pitchers = len(pitcher_sims)
        st.markdown("---")
        st.markdown(
            f'<div style="color:var(--tdd-slate); font-size:0.8rem;">'
            f'{n_pitchers} pitchers simulated | '
            f'10,000 Monte Carlo draws per pitcher</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Props Lab
# ---------------------------------------------------------------------------

_STAT_LABELS = {"TB": "Total Bases", "K": "Strikeouts", "H": "Hits", "HRR": "H+R+RBI", "Outs": "Outs Recorded"}
_LINE_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}


def _edge_color(p_over: float) -> str:
    """Return CSS color based on how far P(over) is from 0.5."""
    if p_over >= 0.60:
        return "var(--tdd-sage)"
    if p_over <= 0.40:
        return "var(--tdd-ember)"
    return "var(--tdd-slate)"


def _edge_label(p_over: float) -> str:
    """Human-readable edge label."""
    if p_over >= 0.60:
        return "Over"
    if p_over <= 0.40:
        return "Under"
    return "Neutral"


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
            '<div style="background:var(--tdd-dark-border); border-left:3px solid '
            'var(--tdd-gold); padding:0.4rem 0.75rem; margin-bottom:0.5rem; '
            'font-size:0.8rem; color:var(--tdd-gold);">'
            'Estimated lineup — probable lineup not yet released'
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
        '<div style="font-size:0.85rem; font-weight:600; color:var(--tdd-sage); '
        'margin:0.6rem 0 0.3rem; letter-spacing:0.3px;">Over Projections</div>',
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
    stat_label = _STAT_LABELS.get(stat, stat)
    stat_short = stat  # K, H, HR, TB, BB
    team = row["team"]
    opp = row["opponent"]
    expected = row["expected"]
    p_over = row["p_over"]
    line = row["line"]
    ptype = row["player_type"]
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
        f'<span class="tdd-player-name" style="min-width:0; flex:1;">{name}'
        f'<span class="tdd-stat-label" style="margin-left:0.3rem;">{team} vs {opp}</span></span>'
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
# Game Drill-Down (Phase 2: Game Center)
# ---------------------------------------------------------------------------

def _render_game_drilldown(
    game: pd.Series,
    lineups: pd.DataFrame,
    h_arch_lookup: dict[int, str],
    p_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    k_samples_dict: dict[str, np.ndarray],
    bf_priors: pd.DataFrame,
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    proj_lookup: dict[int, dict],
    gpk: int,
    bb_samples_dict: dict[str, np.ndarray] | None = None,
    hr_samples_dict: dict[str, np.ndarray] | None = None,
    exit_model: object | None = None,
    pitcher_pc: pd.DataFrame | None = None,
    batter_pc: pd.DataFrame | None = None,
    tto_profiles: pd.DataFrame | None = None,
    exit_tendencies: pd.DataFrame | None = None,
    pos_lookup: dict[int, str] | None = None,
    hitter_k_samples: dict[str, np.ndarray] | None = None,
    hitter_bb_samples: dict[str, np.ndarray] | None = None,
    hitter_hr_samples: dict[str, np.ndarray] | None = None,
    str_df: pd.DataFrame | None = None,
    ploc_df: pd.DataFrame | None = None,
    hzone_df: pd.DataFrame | None = None,
    game_props: pd.DataFrame | None = None,
    live_stats: pd.DataFrame | None = None,
    batter_sims_df: pd.DataFrame | None = None,
) -> None:
    """Rich game drill-down: lineup matchups, matchup analysis, and game simulator."""
    away_abbr = game.get("away_abbr", "?")
    home_abbr = game.get("home_abbr", "?")
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

    # Resolve umpire + weather context for this game
    game_context = _resolve_game_context(game)

    # --- Inline toolbar: Section | Team ---
    _SECTIONS = [
        "Lineups", "Props Lab",
        "Game Simulator",
    ]
    team_options = [away_abbr, home_abbr]

    tb_cols = st.columns([2, 1])
    with tb_cols[0]:
        section = st.selectbox(
            "Section", _SECTIONS,
            key=f"dd_section_{gpk}",
            label_visibility="collapsed",
        )
    with tb_cols[1]:
        selected_team = st.selectbox(
            "Team", team_options,
            key=f"dd_team_{gpk}",
            label_visibility="collapsed",
            disabled=section in ("Lineups", "Props Lab"),
        )
    side_idx = 0 if selected_team == away_abbr else 1
    active_side = sides[side_idx]

    _batter_sims = batter_sims_df if batter_sims_df is not None else pd.DataFrame()

    # --- Render selected section ---
    if section == "Lineups":
        # Detect lineup changes vs precomputed sims
        changes = _detect_lineup_changes(game_lu, _batter_sims, gpk)
        if changes["changed"]:
            n_new = len(changes["new_batters"])
            n_miss = len(changes["missing_batters"])
            parts = []
            if n_new:
                parts.append(f"{n_new} new batter(s)")
            if n_miss:
                parts.append(f"{n_miss} removed")
            st.warning(f"Lineup changed since last update. {', '.join(parts)}.")
            if st.button("Re-run sims", key=f"resim_{gpk}"):
                import subprocess
                subprocess.Popen(
                    [sys.executable, "scripts/update_in_season.py", "--batter-sims-only"],
                    cwd=str(Path(__file__).resolve().parents[1]),
                )
                st.success("Batter sims re-triggered. Refresh in ~30 seconds.")
                st.cache_data.clear()

        _render_matchup_tab_sidebyside(
            sides, h_arch_lookup, h_stat_lookup,
            arsenal_df, vuln_df, gpk,
            pos_lookup=pos_lookup or {},
            str_df=str_df if str_df is not None else pd.DataFrame(),
            hitter_k_samples=hitter_k_samples or {},
            hitter_bb_samples=hitter_bb_samples or {},
            hitter_hr_samples=hitter_hr_samples or {},
            bf_priors=bf_priors if bf_priors is not None else pd.DataFrame(),
            batter_sims_df=_batter_sims,
        )

    elif section == "Props Lab":
        _render_props_section(
            gpk,
            game_props if game_props is not None else pd.DataFrame(),
            lineups_df=game_lu,
            live_stats_df=live_stats,
        )

    elif section == "Game Simulator":
        _render_sim_tab(
            [active_side], h_stat_lookup, k_samples_dict,
            bf_priors, arsenal_df, vuln_df, gpk,
            game_context=game_context,
            bb_samples_dict=bb_samples_dict or {},
            hr_samples_dict=hr_samples_dict or {},
            exit_model=exit_model,
            pitcher_pc=pitcher_pc if pitcher_pc is not None else pd.DataFrame(),
            batter_pc=batter_pc if batter_pc is not None else pd.DataFrame(),
            tto_profiles=tto_profiles if tto_profiles is not None else pd.DataFrame(),
            exit_tendencies=exit_tendencies if exit_tendencies is not None else pd.DataFrame(),
            hitter_k_samples=hitter_k_samples or {},
            hitter_bb_samples=hitter_bb_samples or {},
            hitter_hr_samples=hitter_hr_samples or {},
            h_arch_lookup=h_arch_lookup,
            pos_lookup=pos_lookup or {},
            sides_all=sides,
        )


def _render_hitter_projections_tab(
    sides: list[dict],
    h_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    gpk: int,
    hitter_k_samples: dict[str, np.ndarray] | None = None,
    hitter_bb_samples: dict[str, np.ndarray] | None = None,
    hitter_hr_samples: dict[str, np.ndarray] | None = None,
    bf_priors: pd.DataFrame | None = None,
    arsenal_df: pd.DataFrame | None = None,
    vuln_df: pd.DataFrame | None = None,
    pos_lookup: dict[int, str] | None = None,
    sides_all: list[dict] | None = None,
) -> None:
    """Hitter projection cards | MLB-style lineup with game-level sim projections."""
    from lib.game_sim.batter_simulator import simulate_batter_game
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    if pos_lookup is None:
        pos_lookup = {}
    if hitter_k_samples is None:
        hitter_k_samples = {}
    if hitter_bb_samples is None:
        hitter_bb_samples = {}
    if hitter_hr_samples is None:
        hitter_hr_samples = {}
    if bf_priors is None:
        bf_priors = pd.DataFrame()
    if arsenal_df is None:
        arsenal_df = pd.DataFrame()
    if vuln_df is None:
        vuln_df = pd.DataFrame()

    # Use sides_all for opposing pitcher lookup when sides is a single-element list
    _all_sides = sides_all if sides_all is not None else sides

    # Build baselines for matchup scoring
    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    # Bullpen league-average fallback rates
    bullpen_k = 0.23
    bullpen_bb = 0.09
    bullpen_hr = 0.03

    for side_info in sides:
        side_abbr = side_info["abbr"]
        own_lu = side_info["own_lineup"]
        pitcher_name = side_info["pitcher_name"]
        pid = side_info["pitcher_id"]
        p_arch = side_info["pitcher_arch"]
        p_proj = side_info["pitcher_proj"]

        # Find opposing pitcher from the full sides list
        opp_side = next(
            (s for s in _all_sides if s["side"] != side_info["side"]),
            None,
        )
        opp_pid = opp_side["pitcher_id"] if opp_side else None
        opp_proj = opp_side["pitcher_proj"] if opp_side else {}
        opp_pitcher_name = opp_side["pitcher_name"] if opp_side else "TBD"

        # Opposing pitcher rates
        opp_k_rate = float(opp_proj.get("projected_k_rate", 0.22)) if opp_proj else 0.22
        opp_bb_rate = float(opp_proj.get("projected_bb_rate", 0.08)) if opp_proj else 0.08
        opp_hr_rate = float(opp_proj.get("projected_hr_per_bf", 0.03)) if opp_proj else 0.03

        # Opposing pitcher BF priors
        opp_bf_mu = 22.0
        opp_bf_sigma = 4.5
        if opp_pid and not bf_priors.empty:
            bp_row = bf_priors[bf_priors["pitcher_id"] == opp_pid]
            if not bp_row.empty:
                bp_last = bp_row.sort_values("season").iloc[-1]
                opp_bf_mu = float(bp_last["mu_bf"])
                opp_bf_sigma = float(bp_last["sigma_bf"])

        # Header | include opposing pitcher context
        opp_ctx = ""
        if opp_pid:
            opp_ctx = (
                f' <span style="color:var(--tdd-slate); font-size:0.8rem; font-weight:400;">'
                f'vs {opp_pitcher_name}</span>'
            )
        st.markdown(
            f'<div class="tdd-section-hdr">'
            f'<span class="tdd-team-abbr" data-team="{side_abbr}">{side_abbr}</span>'
            f' Lineup{opp_ctx}</div>',
            unsafe_allow_html=True,
        )

        if own_lu.empty:
            st.markdown(
                f'<div style="color:var(--tdd-slate); font-size:0.85rem; '
                f'padding:0.5rem 0;">No probable lineup yet</div>',
                unsafe_allow_html=True,
            )
            continue

        name_col = "batter_name" if "batter_name" in own_lu.columns else "player_name"
        id_col = "batter_id" if "batter_id" in own_lu.columns else "player_id"

        rows_html: list[str] = []
        for _, brow in own_lu.head(9).iterrows():
            bid = int(brow[id_col]) if pd.notna(brow.get(id_col)) else None
            bname = brow.get(name_col, "Unknown")
            order = int(brow["batting_order"])

            pos = pos_lookup.get(bid, "--") if bid else "--"
            stats = h_stat_lookup.get(bid, {}) if bid else {}
            arch = h_arch_lookup.get(bid, "Prospect") if bid else ""

            composite = stats.get("tdd_value_score")
            diamond_html = ""
            if pd.notna(composite):
                diamond_html = diamond_rating_html(0, size="sm", precomputed=composite)

            arch_html = (
                f'<span class="tdd-badge">{arch}</span>'
            ) if arch else ""

            # --- Game-level batter simulation ---
            stat_html = ""
            bid_key = str(bid) if bid else None
            has_batter_samples = (
                bid_key is not None
                and bid_key in hitter_k_samples
                and bid_key in hitter_bb_samples
                and bid_key in hitter_hr_samples
                and opp_pid is not None
            )

            if has_batter_samples:
                try:
                    # Matchup lifts
                    matchup_k_lift = 0.0
                    matchup_bb_lift = 0.0
                    matchup_hr_lift = 0.0
                    if (
                        opp_pid and bid
                        and not arsenal_df.empty and not vuln_df.empty
                    ):
                        k_res = score_matchup(
                            opp_pid, bid, arsenal_df, vuln_df, baselines_pt,
                        )
                        bb_res = score_matchup_bb(
                            opp_pid, bid, arsenal_df, vuln_df, baselines_pt,
                        )
                        hr_res = score_matchup_hr(
                            opp_pid, bid, arsenal_df, vuln_df, baselines_pt,
                        )
                        matchup_k_lift = k_res.get("matchup_k_logit_lift", 0.0)
                        matchup_bb_lift = bb_res.get("matchup_bb_logit_lift", 0.0)
                        matchup_hr_lift = hr_res.get("matchup_hr_logit_lift", 0.0)

                    sim_result = simulate_batter_game(
                        batter_k_rate_samples=hitter_k_samples[bid_key],
                        batter_bb_rate_samples=hitter_bb_samples[bid_key],
                        batter_hr_rate_samples=hitter_hr_samples[bid_key],
                        batting_order=order,
                        starter_k_rate=opp_k_rate,
                        starter_bb_rate=opp_bb_rate,
                        starter_hr_rate=opp_hr_rate,
                        starter_bf_mu=opp_bf_mu,
                        starter_bf_sigma=opp_bf_sigma,
                        matchup_k_lift=matchup_k_lift,
                        matchup_bb_lift=matchup_bb_lift,
                        matchup_hr_lift=matchup_hr_lift,
                        bullpen_k_rate=bullpen_k,
                        bullpen_bb_rate=bullpen_bb,
                        bullpen_hr_rate=bullpen_hr,
                        n_sims=5_000,
                        random_seed=gpk + (bid or 0),
                    )
                    summary = sim_result.summary()
                    e_k = summary["k"]["mean"]
                    e_bb = summary["bb"]["mean"]
                    e_h = summary["h"]["mean"]
                    e_hr = summary["hr"]["mean"]

                    stat_html = (
                        f'<span style="color:var(--tdd-slate); font-size:0.7rem;">'
                        f'K {e_k:.2f} · BB {e_bb:.2f} · '
                        f'H {e_h:.2f} · HR {e_hr:.2f}'
                        f'</span>'
                    )
                except Exception:
                    # Fall back to season projections on any error
                    has_batter_samples = False

            if not has_batter_samples:
                # Fallback: season counting projections
                proj_k = stats.get("total_k")
                proj_bb = stats.get("total_bb")
                proj_hr = stats.get("total_hr")
                stat_parts: list[str] = []
                if pd.notna(proj_k):
                    stat_parts.append(f'K {proj_k:.0f}')
                if pd.notna(proj_bb):
                    stat_parts.append(f'BB {proj_bb:.0f}')
                if pd.notna(proj_hr):
                    stat_parts.append(f'HR {proj_hr:.0f}')
                if stat_parts:
                    stat_html = (
                        f'<span style="color:var(--tdd-slate); font-size:0.7rem;">'
                        f'{" · ".join(stat_parts)}</span>'
                    )

            hs = ""
            if bid:
                hs = f'<span class="lineup-hs" style="margin:0 0.3rem;">{headshot_html(bid, size=32)}</span>'

            _order_color = "var(--tdd-gold)" if order <= 3 else "var(--tdd-slate)"
            rows_html.append(
                f'<div style="display:flex; align-items:center; gap:0.3rem; '
                f'padding:0.3rem 0.6rem; border-bottom:1px solid var(--tdd-dark-border-faint);">'
                f'<span style="color:{_order_color}; font-size:0.8rem; '
                f'min-width:1.2rem; text-align:right; font-weight:700;">{order}</span>'
                f'{hs}'
                f'<span style="color:var(--tdd-slate); font-size:0.72rem; min-width:1.8rem; margin-left:0.3rem;">{pos}</span>'
                f'<span style="color:var(--tdd-cream); font-size:0.88rem; font-weight:600; '
                f'flex:1; min-width:5rem;">{bname}</span>'
                f'{arch_html}'
                f'<span class="lineup-diamonds" style="margin:0 0.3rem;">{diamond_html}</span>'
                f'{stat_html}'
                f'</div>'
            )

        # Pitcher at bottom (MLB style)
        if pid:
            p_composite = p_proj.get("tdd_value_score")
            p_diamond = diamond_rating_html(0, size="sm", precomputed=p_composite) if pd.notna(p_composite) else ""
            p_arch_html = (
                f'<span class="tdd-badge">{p_arch}</span>'
            ) if p_arch else ""
            p_hs = f'<span class="lineup-hs" style="margin:0 0.3rem;">{headshot_html(pid, size=32)}</span>'

            rows_html.append(
                f'<div class="tdd-lineup-row" style="background:rgba(200,169,110,0.06);">'
                f'<span style="color:var(--tdd-gold); font-size:var(--tdd-fs-meta); '
                f'min-width:1.2rem; text-align:right; font-weight:700;">P</span>'
                f'{p_hs}'
                f'<span class="tdd-stat-label" style="min-width:1.8rem;">SP</span>'
                f'<span class="tdd-player-name" style="flex:1; min-width:5rem;">{pitcher_name}</span>'
                f'{p_arch_html}'
                f'<span class="lineup-diamonds" style="margin:0 0.3rem;">{p_diamond}</span>'
                f'</div>'
            )

        if rows_html:
            st.markdown(
                f'<div style="background:transparent; border:none; '
                f'border-bottom:1px solid var(--tdd-dark-border); margin-bottom:1rem; overflow:hidden;">'
                + "".join(rows_html)
                + '</div>',
                unsafe_allow_html=True,
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
    hitter_k_samples: dict[str, np.ndarray] | None = None,
    hitter_bb_samples: dict[str, np.ndarray] | None = None,
    hitter_hr_samples: dict[str, np.ndarray] | None = None,
    bf_priors: pd.DataFrame | None = None,
    batter_sims_df: pd.DataFrame | None = None,
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
                hitter_k_samples=hitter_k_samples or {},
                hitter_bb_samples=hitter_bb_samples or {},
                hitter_hr_samples=hitter_hr_samples or {},
                bf_priors=bf_priors if bf_priors is not None else pd.DataFrame(),
                batter_sims_df=batter_sims_df if batter_sims_df is not None else pd.DataFrame(),
            )


def _grade_color(grade: int) -> str:
    """Return color for a 20-80 scouting grade."""
    if grade >= 70:
        return "var(--tdd-gold)"
    if grade >= 55:
        return "var(--tdd-sage)"
    if grade >= 45:
        return "var(--tdd-cream)"
    return "var(--tdd-slate)"


def _hitter_grades_html(stats: dict) -> str:
    """Compact grade chips for a hitter: Con/Pow/Spd/Disc/Fld with CI ranges."""
    labels = [
        ("Con", "grade_hit"), ("Pow", "grade_power"), ("Spd", "grade_speed"),
        ("Disc", "grade_discipline"), ("Fld", "grade_fielding"),
    ]
    parts = []
    for abbr, key in labels:
        v = stats.get(key)
        if v is not None:
            c = _grade_color(v)
            lo = stats.get(f"{key}_lo")
            hi = stats.get(f"{key}_hi")
            ci_html = ""
            if lo is not None and hi is not None:
                ci_html = (
                    f'<span style="color:var(--tdd-slate); font-size:0.55rem; '
                    f'opacity:0.7; margin-left:1px;">({lo}-{hi})</span>'
                )
            parts.append(f'<span style="color:{c};">{abbr} {v}{ci_html}</span>')
    if not parts:
        return ""
    return (
        f'<span style="font-size:0.65rem; letter-spacing:0.3px; '
        f'display:inline-flex; gap:0.4rem;">'
        + "".join(parts) + '</span>'
    )


def _pitcher_grades_html(proj: dict) -> str:
    """Compact grade chips for a pitcher: Stuff/Cmd/Dur with CI ranges."""
    labels = [
        ("Stuff", "grade_stuff"), ("Cmd", "grade_command"), ("Dur", "grade_durability"),
    ]
    parts = []
    for abbr, key in labels:
        v = proj.get(key)
        if v is not None:
            c = _grade_color(v)
            lo = proj.get(f"{key}_lo")
            hi = proj.get(f"{key}_hi")
            ci_html = ""
            if lo is not None and hi is not None:
                ci_html = (
                    f'<span style="color:var(--tdd-slate); font-size:0.55rem; '
                    f'opacity:0.7; margin-left:1px;">({lo}-{hi})</span>'
                )
            parts.append(f'<span style="color:{c};">{abbr} {v}{ci_html}</span>')
    if not parts:
        return ""
    return (
        f'<span style="font-size:0.65rem; letter-spacing:0.3px; '
        f'display:inline-flex; gap:0.4rem;">'
        + "".join(parts) + '</span>'
    )


def _build_scouting_bullets(
    pitcher_id: int,
    pitcher_name: str,
    opp_lu: pd.DataFrame,
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
) -> list[str]:
    """Generate natural-language scouting bullets for a pitcher vs lineup.

    Returns a list of 2-4 plain-English insights like:
    "Webb's #1 pitch is a cutter, but Judge crushes it at .470 xwOBA"
    """
    if arsenal_df.empty or vuln_df.empty or opp_lu.empty:
        return []

    p_ars = arsenal_df[arsenal_df["pitcher_id"] == pitcher_id].copy()
    if p_ars.empty:
        return []
    p_ars = p_ars.sort_values("usage_pct", ascending=False)

    id_col = "batter_id" if "batter_id" in opp_lu.columns else "player_id"
    name_col = "batter_name" if "batter_name" in opp_lu.columns else "player_name"
    batter_ids = [int(b) for b in opp_lu.head(9)[id_col].dropna()]
    batter_names = dict(zip(
        opp_lu.head(9)[id_col].dropna().astype(int),
        opp_lu.head(9)[name_col],
    ))

    # Last name only for brevity
    def _last(name: str) -> str:
        parts = name.strip().split()
        return parts[-1] if parts else name

    p_last = _last(pitcher_name)

    # Primary pitch info
    top_pitch = p_ars.iloc[0]
    top_pt = top_pitch["pitch_type"]
    top_pt_name = PITCH_DISPLAY.get(top_pt, top_pt).lower()
    top_usage = top_pitch["usage_pct"]

    bullets: list[str] = []

    # 1. Best hitter matchup vs pitcher's primary pitch (highest xwOBA)
    best_xwoba, best_batter = 0.0, ""
    for bid in batter_ids:
        bv = vuln_df[(vuln_df["batter_id"] == bid) & (vuln_df["pitch_type"] == top_pt)]
        if not bv.empty:
            xw = bv.iloc[0].get("xwoba_contact")
            if pd.notna(xw) and xw > best_xwoba:
                best_xwoba = xw
                best_batter = _last(batter_names.get(bid, ""))

    if best_batter and best_xwoba > 0.370:
        bullets.append(
            f"{p_last}'s #1 pitch is the {top_pt_name} ({top_usage:.0%} usage), "
            f"but {best_batter} crushes it at .{int(best_xwoba * 1000):03d} xwOBA"
        )
    elif best_batter:
        bullets.append(
            f"{p_last} leans on the {top_pt_name} ({top_usage:.0%} usage); "
            f"{best_batter} has the best contact at .{int(best_xwoba * 1000):03d} xwOBA"
        )

    # 2. Pitcher's best whiff pitch vs lineup weakness
    whiff_pitches = p_ars[p_ars["whiff_rate"] > 0.28].head(2)
    for _, wp in whiff_pitches.iterrows():
        pt = wp["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt).lower()
        wr = wp["whiff_rate"]
        # Count batters vulnerable (high whiff) to this pitch
        vulnerable = 0
        for bid in batter_ids:
            bv = vuln_df[(vuln_df["batter_id"] == bid) & (vuln_df["pitch_type"] == pt)]
            if not bv.empty and bv.iloc[0].get("whiff_rate", 0) > 0.30:
                vulnerable += 1
        if vulnerable >= 3:
            bullets.append(
                f"{p_last}'s {pt_name} generates a {wr:.0%} whiff rate; "
                f"{vulnerable} of 9 batters whiff over 30% against that pitch"
            )
            break  # one whiff bullet is enough

    # 3. Hitter with biggest xwOBA edge across all pitch types
    best_edge_batter, best_edge_pt, best_edge_val = "", "", 0.0
    for bid in batter_ids:
        for _, pa in p_ars.iterrows():
            pt = pa["pitch_type"]
            p_xwoba = pa.get("xwoba_against", 0.320)
            bv = vuln_df[(vuln_df["batter_id"] == bid) & (vuln_df["pitch_type"] == pt)]
            if not bv.empty:
                h_xwoba = bv.iloc[0].get("xwoba_contact")
                if pd.notna(h_xwoba) and pd.notna(p_xwoba) and h_xwoba > 0.400:
                    edge = h_xwoba - p_xwoba
                    if edge > best_edge_val:
                        best_edge_val = edge
                        best_edge_batter = _last(batter_names.get(bid, ""))
                        best_edge_pt = PITCH_DISPLAY.get(pt, pt).lower()

    if best_edge_batter and best_edge_val > 0.050:
        bullets.append(
            f"{best_edge_batter} has a big edge on {p_last}'s {best_edge_pt}; "
            f"look for the pitcher to avoid that zone"
        )

    # 4. Pitcher dominance angle: elite whiff rate on primary pitch
    if top_pitch.get("whiff_rate", 0) > 0.32:
        weak_batters = []
        for bid in batter_ids:
            bv = vuln_df[(vuln_df["batter_id"] == bid) & (vuln_df["pitch_type"] == top_pt)]
            if not bv.empty and bv.iloc[0].get("whiff_rate", 0) > 0.35:
                weak_batters.append(_last(batter_names.get(bid, "")))
        if weak_batters:
            names = ", ".join(weak_batters[:3])
            bullets.append(
                f"Pitcher advantage: {names} all struggle against the {top_pt_name}"
            )

    return bullets[:4]  # cap at 4


# ---------------------------------------------------------------------------
# Matchup advantage badge with short explanation
# ---------------------------------------------------------------------------
def _matchup_advantage_html(
    k_result: dict,
    bb_result: dict,
    hr_result: dict,
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame | None,
    bid: int,
    matchup_pid: int,
) -> tuple[str, float, float, float]:
    """Return (HTML badge + short reason, k_lift, bb_lift, hr_lift).

    Analyzes per-pitch-type data to find the specific pitch driving the edge
    and references a concrete stat (whiff rate, xwOBA, chase rate).
    """
    k_lift = k_result.get("matchup_k_logit_lift", 0.0)
    bb_lift = bb_result.get("matchup_bb_logit_lift", 0.0)
    hr_lift = hr_result.get("matchup_hr_logit_lift", 0.0)
    k_lift = 0.0 if np.isnan(k_lift) else k_lift
    bb_lift = 0.0 if np.isnan(bb_lift) else bb_lift
    hr_lift = 0.0 if np.isnan(hr_lift) else hr_lift

    net = k_lift - bb_lift * 0.5 - hr_lift * 0.5

    # Determine which outcome channel dominates
    abs_k = abs(k_lift)
    abs_bb = abs(bb_lift) * 0.5
    abs_hr = abs(hr_lift) * 0.5

    # Gather per-pitch detail from arsenal/vuln for the dominant pitch
    p_ars = arsenal_df[arsenal_df["pitcher_id"] == matchup_pid].copy()
    p_ars = p_ars[p_ars["pitches"] >= 20]
    h_vuln = vuln_df[vuln_df["batter_id"] == bid].copy()
    s_df_b = (
        str_df[(str_df["batter_id"] == bid)]
        if str_df is not None and not str_df.empty and "batter_id" in str_df.columns
        else pd.DataFrame()
    )

    if abs(net) <= 0.03:
        reason = _find_even_reason(p_ars, h_vuln, s_df_b)
        html_out = (
            f'<span style="color:var(--tdd-slate); font-size:0.68rem; '
            f'margin-left:0.3rem;" title="{reason}">'
            f'Even</span>'
        )
        return html_out, k_lift, bb_lift, hr_lift

    if net > 0.03:
        color_var = "--tdd-ember"
        label = "Pitcher"
        reason = _find_pitcher_reason(
            k_lift, bb_lift, hr_lift, abs_k, abs_bb, abs_hr,
            p_ars, h_vuln, s_df_b,
        )
    else:
        color_var = "--tdd-sage"
        label = "Hitter"
        reason = _find_hitter_reason(
            k_lift, bb_lift, hr_lift, abs_k, abs_bb, abs_hr,
            p_ars, h_vuln, s_df_b,
        )

    html_out = (
        f'<span style="color:var({color_var}); font-size:0.68rem; '
        f'font-weight:600; margin-left:0.3rem;" title="{reason}">'
        f'{label} - '
        f'<span style="font-weight:400; font-size:0.62rem;">{reason}</span>'
        f'</span>'
    )
    return html_out, k_lift, bb_lift, hr_lift


def _find_pitcher_reason(
    k_lift: float,
    bb_lift: float,
    hr_lift: float,
    abs_k: float,
    abs_bb: float,
    abs_hr: float,
    p_ars: pd.DataFrame,
    h_vuln: pd.DataFrame,
    s_df: pd.DataFrame,
) -> str:
    """Build a short reason string for pitcher advantage."""
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    _MIN_USAGE = 0.10  # ignore pitches thrown <10% of the time

    # K-driven advantage: find the pitch with highest usage-weighted whiff edge
    if abs_k >= abs_bb and abs_k >= abs_hr and not p_ars.empty:
        best_pt, best_score = "", 0.0
        for _, row in p_ars.iterrows():
            pt = row["pitch_type"]
            usage = row.get("usage_pct", 0)
            if pd.isna(usage) or usage < _MIN_USAGE:
                continue
            p_whiff = row.get("whiff_rate", 0)
            if pd.isna(p_whiff):
                continue
            lg_whiff = LEAGUE_AVG_BY_PITCH_TYPE.get(
                pt, LEAGUE_AVG_OVERALL,
            ).get("whiff_rate", 0.25)
            hv = h_vuln[h_vuln["pitch_type"] == pt]
            h_whiff = (
                hv.iloc[0].get("whiff_rate", lg_whiff) if len(hv) > 0
                else lg_whiff
            )
            if pd.isna(h_whiff):
                h_whiff = lg_whiff
            combined = p_whiff * 0.6 + h_whiff * 0.4
            score = combined * usage  # weight by how often it's actually thrown
            if score > best_score:
                best_score = score
                best_pt = pt
                _best_whiff = combined

        if best_pt and _best_whiff > 0.25:
            pt_name = PITCH_DISPLAY.get(best_pt, best_pt).lower()
            pct = int(_best_whiff * 100)
            return f"whiffs on {pt_name} ({pct}%)"

        return "K matchup favors pitcher"

    # BB-driven (pitcher induces chases): find highest chase pitch by usage
    if abs_bb >= abs_k and abs_bb >= abs_hr and not p_ars.empty:
        best_pt, best_score = "", 0.0
        _best_chase = 0.0
        for _, row in p_ars.iterrows():
            pt = row["pitch_type"]
            usage = row.get("usage_pct", 0)
            if pd.isna(usage) or usage < _MIN_USAGE:
                continue
            hv = h_vuln[h_vuln["pitch_type"] == pt]
            if len(hv) > 0:
                cs = hv.iloc[0].get("chase_swings", 0)
                oz = hv.iloc[0].get("out_of_zone_pitches", 0)
                if pd.notna(oz) and oz > 0 and pd.notna(cs):
                    chase = cs / oz
                    score = chase * usage
                    if score > best_score:
                        best_score = score
                        _best_chase = chase
                        best_pt = pt
        if best_pt and _best_chase > 0.30:
            pt_name = PITCH_DISPLAY.get(best_pt, best_pt).lower()
            pct = int(_best_chase * 100)
            return f"chases {pt_name} ({pct}%)"

        return "high chase rate"

    # HR-driven (pitcher suppresses barrels)
    if not p_ars.empty:
        return "weak contact expected"

    return "K matchup favors pitcher"


def _find_hitter_reason(
    k_lift: float,
    bb_lift: float,
    hr_lift: float,
    abs_k: float,
    abs_bb: float,
    abs_hr: float,
    p_ars: pd.DataFrame,
    h_vuln: pd.DataFrame,
    s_df: pd.DataFrame,
) -> str:
    """Build a short reason string for hitter advantage."""
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    _MIN_USAGE = 0.10

    # HR-driven advantage: find pitch with highest xwOBA (weighted by usage)
    if abs_hr >= abs_k and abs_hr >= abs_bb:
        best_pt, best_score, _best_xwoba = "", 0.0, 0.0
        # Check hitter strength data first
        if not s_df.empty and "xwoba_contact" in s_df.columns:
            for _, row in p_ars.iterrows():
                pt = row["pitch_type"]
                usage = row.get("usage_pct", 0)
                if pd.isna(usage) or usage < _MIN_USAGE:
                    continue
                sr = (
                    s_df[s_df["pitch_type"] == pt]
                    if "pitch_type" in s_df.columns
                    else pd.DataFrame()
                )
                if len(sr) > 0:
                    xw = sr.iloc[0].get("xwoba_contact", 0)
                    if pd.notna(xw):
                        score = xw * usage
                        if score > best_score:
                            best_score = score
                            _best_xwoba = xw
                            best_pt = pt
        # Fallback to vuln data
        if not best_pt:
            for _, row in p_ars.iterrows():
                pt = row["pitch_type"]
                usage = row.get("usage_pct", 0)
                if pd.isna(usage) or usage < _MIN_USAGE:
                    continue
                hv = h_vuln[h_vuln["pitch_type"] == pt]
                if len(hv) > 0 and "xwoba_contact" in hv.columns:
                    xw = hv.iloc[0].get("xwoba_contact", 0)
                    if pd.notna(xw):
                        score = xw * usage
                        if score > best_score:
                            best_score = score
                            _best_xwoba = xw
                            best_pt = pt

        if best_pt and _best_xwoba > 0.350:
            pt_name = PITCH_DISPLAY.get(best_pt, best_pt).lower()
            return f"barrels {pt_name} (.{int(_best_xwoba * 1000):03d})"

        return "power matchup favors hitter"

    # BB-driven (hitter has plate discipline): find pitcher's most-used pitch
    # that the hitter lays off
    if abs_bb >= abs_k and abs_bb >= abs_hr:
        best_pt, best_score, _best_chase = "", 0.0, 1.0
        for _, row in p_ars.iterrows():
            pt = row["pitch_type"]
            usage = row.get("usage_pct", 0)
            if pd.isna(usage) or usage < _MIN_USAGE:
                continue
            hv = h_vuln[h_vuln["pitch_type"] == pt]
            if len(hv) > 0:
                cs = hv.iloc[0].get("chase_swings", 0)
                oz = hv.iloc[0].get("out_of_zone_pitches", 0)
                if pd.notna(oz) and oz > 10 and pd.notna(cs):
                    chase = cs / oz
                    # Score: low chase on high-usage pitches matters most
                    score = (1 - chase) * usage
                    if score > best_score:
                        best_score = score
                        _best_chase = chase
                        best_pt = pt
        if best_pt and _best_chase < 0.30:
            pt_name = PITCH_DISPLAY.get(best_pt, best_pt).lower()
            pct = int(_best_chase * 100)
            return f"lays off {pt_name} ({pct}% chase)"

        return "patient, low chase rate"

    # K-driven (hitter resists whiffs): find the high-usage pitch hitter handles
    if not p_ars.empty:
        best_pt, best_score, _best_whiff = "", 0.0, 1.0
        for _, row in p_ars.iterrows():
            pt = row["pitch_type"]
            usage = row.get("usage_pct", 0)
            p_whiff = row.get("whiff_rate", 0)
            if pd.isna(usage) or usage < _MIN_USAGE:
                continue
            if pd.isna(p_whiff) or p_whiff < 0.20:
                continue  # only consider pitches pitcher uses to get whiffs
            hv = h_vuln[h_vuln["pitch_type"] == pt]
            if len(hv) > 0:
                h_whiff = hv.iloc[0].get("whiff_rate", 1.0)
                if pd.notna(h_whiff):
                    # Score: low hitter whiff on high-usage whiff pitches
                    score = (1 - h_whiff) * usage
                    if score > best_score:
                        best_score = score
                        _best_whiff = h_whiff
                        best_pt = pt
        if best_pt and _best_whiff < 0.25:
            pt_name = PITCH_DISPLAY.get(best_pt, best_pt).lower()
            pct = int(_best_whiff * 100)
            return f"handles {pt_name} ({pct}% whiff)"

        return "low whiff rate vs arsenal"

    return "matchup favors hitter"


def _find_even_reason(
    p_ars: pd.DataFrame,
    h_vuln: pd.DataFrame,
    s_df: pd.DataFrame,
) -> str:
    """Build a short reason for an even matchup."""
    if p_ars.empty:
        return "limited data"
    return "no strong pitch edges"


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
    hitter_k_samples: dict[str, np.ndarray] | None = None,
    hitter_bb_samples: dict[str, np.ndarray] | None = None,
    hitter_hr_samples: dict[str, np.ndarray] | None = None,
    bf_priors: pd.DataFrame | None = None,
    batter_sims_df: pd.DataFrame | None = None,
) -> None:
    """Team roster with inline matchup scouting on expand.

    When *opp_side* is provided (side-by-side view), each column shows:
      - Team header
      - Own SP at top
      - Own lineup with batter matchup badges vs opposing SP
      - Expanding a batter shows matchup scouting bullets + grades + game projections
      - Avg K matchup summary for own SP vs opposing lineup
    """
    from scipy.special import expit, logit as sp_logit
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE
    from components.scouting import build_matchup_scouting_bullets

    if pos_lookup is None:
        pos_lookup = {}
    if hitter_k_samples is None:
        hitter_k_samples = {}
    if hitter_bb_samples is None:
        hitter_bb_samples = {}
    if hitter_hr_samples is None:
        hitter_hr_samples = {}
    if bf_priors is None:
        bf_priors = pd.DataFrame()
    if batter_sims_df is None:
        batter_sims_df = pd.DataFrame()

    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    def _safe_rate(r: object, default: float) -> float:
        if pd.notna(r) and 0 < float(r) < 1:
            return float(r)
        return default

    for side_info in sides:
        pitcher_name = side_info["pitcher_name"]
        pid = side_info["pitcher_id"]
        p_arch = side_info["pitcher_arch"]
        p_proj = side_info["pitcher_proj"]
        opp_lu = side_info["opp_lineup"]
        opp_abbr = side_info["opp_abbr"]
        side_abbr = side_info["abbr"]

        p_k = _safe_rate(p_proj.get("projected_k_rate"), 0.22)
        p_bb = _safe_rate(p_proj.get("projected_bb_rate"), 0.08)
        p_hr = _safe_rate(p_proj.get("projected_hr_per_bf"), 0.03)

        # Team-centric view: own roster, batter matchups vs opposing SP
        if opp_side is not None:
            display_lu = side_info["own_lineup"]
            matchup_pid = opp_side["pitcher_id"]
            _opp_proj = opp_side.get("pitcher_proj") or {}
        else:
            display_lu = opp_lu
            matchup_pid = pid
            _opp_proj = p_proj

        # Opposing pitcher rates + BF priors for batter sim
        _opp_k_rate = _safe_rate(_opp_proj.get("projected_k_rate"), 0.22)
        _opp_bb_rate = _safe_rate(_opp_proj.get("projected_bb_rate"), 0.08)
        _opp_hr_rate = _safe_rate(_opp_proj.get("projected_hr_per_bf"), 0.03)
        _opp_bf_mu = 22.0
        _opp_bf_sigma = 4.5
        if matchup_pid and not bf_priors.empty:
            _bp_row = bf_priors[bf_priors["pitcher_id"] == matchup_pid]
            if not _bp_row.empty:
                _bp_last = _bp_row.sort_values("season").iloc[-1]
                _opp_bf_mu = float(_bp_last["mu_bf"])
                _opp_bf_sigma = float(_bp_last["sigma_bf"])

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
                f'No probable lineup yet</div>',
                unsafe_allow_html=True,
            )
            continue

        name_col = "batter_name" if "batter_name" in display_lu.columns else "player_name"
        id_col = "batter_id" if "batter_id" in display_lu.columns else "player_id"

        rows_html: list[str] = []
        total_k_lift = total_bb_lift = 0.0
        n_scored = 0

        # Pitcher card at top
        if pid:
            p_composite = p_proj.get("tdd_value_score")
            p_diamond = diamond_rating_html(0, size="sm", precomputed=p_composite) if pd.notna(p_composite) else ""
            p_arch_html = (
                f'<span class="tdd-badge">{p_arch}</span>'
            ) if p_arch else ""
            p_hs = f'<span class="lineup-hs" style="margin:0 0.3rem;">{headshot_html(pid, size=32)}</span>'

            p_summary = (
                f'<span style="color:var(--tdd-gold); font-size:var(--tdd-fs-meta); '
                f'min-width:1.2rem; text-align:right; font-weight:700;">P</span>'
                f'{p_hs}'
                f'<span class="tdd-stat-label" style="min-width:1.8rem;">SP</span>'
                f'<span class="tdd-player-name" style="flex:1; min-width:5rem;">{pitcher_name}</span>'
                f'{p_arch_html}'
                f'<span class="lineup-diamonds" style="margin:0 0.3rem;">{p_diamond}</span>'
            )
            p_grades_html = _pitcher_grades_html(p_proj)
            p_detail = p_grades_html if p_grades_html else (
                '<span style="color:var(--tdd-slate); font-size:0.7rem;">No grade data</span>'
            )
            rows_html.append(expandable_card_html(p_summary, p_detail))

        for _, brow in display_lu.head(9).iterrows():
            bid = int(brow[id_col]) if pd.notna(brow.get(id_col)) else None
            bname = brow.get(name_col, "Unknown")
            order = int(brow["batting_order"])

            # Position
            pos = pos_lookup.get(bid, "--") if bid else "--"

            # Archetype
            arch = h_arch_lookup.get(bid, "Prospect") if bid else ""
            arch_html = (
                f'<span class="tdd-badge">{arch}</span>'
            ) if arch else ""

            # Diamond rating
            stats = h_stat_lookup.get(bid, {}) if bid else {}
            composite = stats.get("tdd_value_score")
            diamond_html = ""
            if pd.notna(composite):
                diamond_html = diamond_rating_html(0, size="sm", precomputed=composite)

            # Matchup advantage (batter vs opposing SP)
            advantage_html = ""
            k_lift = bb_lift = hr_lift = 0.0
            if matchup_pid and bid and not arsenal_df.empty and not vuln_df.empty:
                k_result = score_matchup(matchup_pid, bid, arsenal_df, vuln_df, baselines_pt)
                bb_result = score_matchup_bb(matchup_pid, bid, arsenal_df, vuln_df, baselines_pt)
                hr_result = score_matchup_hr(matchup_pid, bid, arsenal_df, vuln_df, baselines_pt)

                advantage_html, k_lift, bb_lift, hr_lift = _matchup_advantage_html(
                    k_result, bb_result, hr_result,
                    arsenal_df, vuln_df, str_df,
                    bid, matchup_pid,
                )

                total_k_lift += k_lift
                total_bb_lift += bb_lift
                n_scored += 1

            # Headshot
            hs = ""
            if bid:
                hs = f'<span class="lineup-hs" style="margin:0 0.3rem;">{headshot_html(bid, size=32)}</span>'

            # Summary row (always visible)
            summary = (
                f'<span style="color:{"var(--tdd-gold)" if order <= 3 else "var(--tdd-slate)"}; '
                f'font-size:var(--tdd-fs-meta); min-width:1.2rem; text-align:right; font-weight:700;">{order}</span>'
                f'{hs}'
                f'<span class="tdd-stat-label" style="min-width:1.8rem; margin-left:0.3rem;">{pos}</span>'
                f'<span class="tdd-player-name" style="flex:1; min-width:5rem;">{bname}</span>'
                f'{arch_html}'
                f'<span class="lineup-diamonds" style="margin:0 0.3rem;">{diamond_html}</span>'
                f'{advantage_html}'
            )

            # Detail section: scouting + grades + game projections + link
            detail_parts: list[str] = []
            matchup_link_html = ""

            # Matchup scouting bullets (batter vs opposing SP)
            if matchup_pid and bid and not arsenal_df.empty and not vuln_df.empty:
                _p_arsenal = arsenal_df[arsenal_df["pitcher_id"] == matchup_pid]
                _h_vuln = vuln_df[vuln_df["batter_id"] == bid]
                _h_str = (
                    str_df[str_df["batter_id"] == bid]
                    if str_df is not None and not str_df.empty
                    else pd.DataFrame()
                )
                _opp_pname = opp_side["pitcher_name"] if opp_side else pitcher_name
                bullets = build_matchup_scouting_bullets(
                    _p_arsenal, _h_vuln, _h_str,
                    pitcher_name=_opp_pname, hitter_name=bname,
                )
                if bullets:
                    bullet_html = "".join(
                        f'<div style="color:{c}; font-size:0.78rem; margin:0.12rem 0; '
                        f'padding-left:0.65rem; border-left:2px solid {c};">{t}</div>'
                        for c, t in bullets[:3]
                    )
                    detail_parts.append(bullet_html)

                # Prepare matchup link for bottom
                matchup_link_html = (
                    f'<a href="?page=matchup_explorer&pitcher_id={matchup_pid}&hitter_id={bid}" '
                    f'target="_self" style="color:var(--tdd-gold); font-size:0.75rem; '
                    f'text-decoration:none;">View full matchup</a>'
                )

            # Hitter grades
            grades_html = _hitter_grades_html(stats)
            if grades_html:
                detail_parts.append(grades_html)

            # Game-level batter projections (from precomputed sims)
            _precomp_row = pd.DataFrame()
            if bid and not batter_sims_df.empty:
                _precomp_row = batter_sims_df[
                    (batter_sims_df["game_pk"] == gpk)
                    & (batter_sims_df["batter_id"] == bid)
                ]
            if not _precomp_row.empty:
                _pr = _precomp_row.iloc[0]
                _e_k = float(_pr["expected_k"])
                _e_bb = float(_pr["expected_bb"])
                _e_h = float(_pr["expected_h"])
                _e_hr = float(_pr["expected_hr"])

                detail_parts.append(
                    f'<div style="display:flex; gap:0.6rem; font-size:0.72rem; '
                    f'margin:0.25rem 0; padding:0.2rem 0; '
                    f'border-top:1px solid var(--tdd-dark-border-faint);">'
                    f'<span style="color:var(--tdd-slate);">Game Proj</span>'
                    f'<span style="color:var(--tdd-cream);">{_e_k:.2f} K</span>'
                    f'<span style="color:var(--tdd-cream);">{_e_bb:.2f} BB</span>'
                    f'<span style="color:var(--tdd-cream);">{_e_h:.2f} H</span>'
                    f'<span style="color:var(--tdd-cream);">{_e_hr:.2f} HR</span>'
                    f'</div>'
                )
            elif bid and batter_sims_df.empty:
                pass  # No precomputed sims available at all
            elif bid:
                detail_parts.append(
                    f'<div style="font-size:0.72rem; color:var(--tdd-slate); '
                    f'margin:0.25rem 0; padding:0.2rem 0; '
                    f'border-top:1px solid var(--tdd-dark-border-faint);">'
                    f'Sim pending - lineup may have changed</div>'
                )

            # Matchup Explorer link at bottom
            if matchup_link_html:
                detail_parts.append(matchup_link_html)

            detail = (
                "".join(detail_parts) if detail_parts
                else '<span style="color:var(--tdd-slate); font-size:0.7rem;">No scouting data</span>'
            )

            rows_html.append(expandable_card_html(summary, detail))

        if rows_html:
            st.markdown(
                EXPANDABLE_CARD_CSS
                + f'<div style="margin-bottom:1rem;">'
                + "".join(rows_html)
                + '</div>',
                unsafe_allow_html=True,
            )

        # Avg K matchup for this side's pitcher vs opposing lineup
        if opp_side is not None and pid and not opp_lu.empty:
            opp_id_col = "batter_id" if "batter_id" in opp_lu.columns else "player_id"
            total_k_lift = total_bb_lift = 0.0
            n_scored = 0
            for _, ob in opp_lu.head(9).iterrows():
                ob_id = int(ob[opp_id_col]) if pd.notna(ob.get(opp_id_col)) else None
                if ob_id and not arsenal_df.empty and not vuln_df.empty:
                    kr = score_matchup(pid, ob_id, arsenal_df, vuln_df, baselines_pt)
                    br = score_matchup_bb(pid, ob_id, arsenal_df, vuln_df, baselines_pt)
                    kl = kr.get("matchup_k_logit_lift", 0.0)
                    bl = br.get("matchup_bb_logit_lift", 0.0)
                    total_k_lift += 0.0 if np.isnan(kl) else kl
                    total_bb_lift += 0.0 if np.isnan(bl) else bl
                    n_scored += 1

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

        # Scouting report
        if pid and not opp_lu.empty:
            bullets = _build_scouting_bullets(
                pid, pitcher_name, opp_lu, arsenal_df, vuln_df,
            )
            if bullets:
                bullet_html = "".join(
                    f'<li style="margin-bottom:0.3rem;">{b}</li>' for b in bullets
                )
                st.markdown(
                    f'<div style="margin:0.5rem 0 1rem; padding:0.6rem 0.8rem; '
                    f'border-left:2px solid var(--tdd-gold); font-size:0.82rem; '
                    f'color:var(--tdd-cream);">'
                    f'<div style="color:var(--tdd-gold); font-weight:600; '
                    f'font-size:0.75rem; letter-spacing:0.5px; margin-bottom:0.3rem;">'
                    f'SCOUTING REPORT</div>'
                    f'<ul style="margin:0; padding-left:1.2rem; '
                    f'color:var(--tdd-cream);">{bullet_html}</ul></div>',
                    unsafe_allow_html=True,
                )


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


def _resolve_game_context(game: pd.Series) -> dict:
    """Look up umpire and weather adjustments from parquet data."""
    from scipy.special import logit as _logit

    ump_name = game.get("hp_umpire_name", "")
    ump_k_lift = 0.0
    ump_detail = ""

    if ump_name:
        ump_path = DASHBOARD_DIR / "umpire_tendencies.parquet"
        if ump_path.exists():
            ump_df = pd.read_parquet(ump_path)
            row = ump_df[ump_df["hp_umpire_name"] == ump_name]
            if not row.empty:
                ump_k_lift = float(row.iloc[0]["k_logit_lift"])
                k_rate = float(row.iloc[0]["k_rate_shrunk"])
                league_k = float(row.iloc[0]["league_k_rate"])
                delta_pp = (k_rate - league_k) * 100
                ump_detail = f"{k_rate:.1%} K-rate ({delta_pp:+.1f}pp vs avg)"

    temp_str = game.get("weather_temp", "")
    wind_str = game.get("weather_wind", "")
    condition = game.get("weather_condition", "")
    wx_k_lift = 0.0
    wx_detail = ""

    temp_bucket = _parse_temp_bucket(temp_str)
    wind_cat = _parse_wind_category(wind_str)

    wx_path = DASHBOARD_DIR / "weather_effects.parquet"
    if wx_path.exists():
        wx_df = pd.read_parquet(wx_path)
        wx_row = wx_df[
            (wx_df["temp_bucket"] == temp_bucket) &
            (wx_df["wind_category"] == wind_cat)
        ]
        if not wx_row.empty:
            k_mult = float(wx_row.iloc[0]["k_multiplier"])
            overall_k = float(wx_row.iloc[0]["overall_k_rate"])
            adj_k = np.clip(overall_k * k_mult, 1e-6, 1 - 1e-6)
            wx_k_lift = float(
                _logit(adj_k) - _logit(np.clip(overall_k, 1e-6, 1 - 1e-6))
            )
            k_delta = (k_mult - 1.0) * 100
            if abs(k_delta) > 0.3:
                wx_detail = f"K-rate {k_delta:+.1f}%"

    weather_display = ""
    parts = []
    if temp_str:
        parts.append(f"{temp_str}°F")
    if condition:
        parts.append(condition)
    if wind_str:
        parts.append(wind_str)
    weather_display = " | ".join(parts) if parts else ""

    return {
        "ump_name": ump_name,
        "ump_k_lift": ump_k_lift,
        "ump_detail": ump_detail,
        "wx_k_lift": wx_k_lift,
        "wx_detail": wx_detail,
        "weather_display": weather_display,
        "venue_name": game.get("venue_name", ""),
    }


def _render_sim_tab(
    sides: list[dict],
    h_stat_lookup: dict[int, dict],
    k_samples_dict: dict[str, np.ndarray],
    bf_priors: pd.DataFrame,
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    gpk: int,
    game_context: dict | None = None,
    bb_samples_dict: dict[str, np.ndarray] | None = None,
    hr_samples_dict: dict[str, np.ndarray] | None = None,
    exit_model: object | None = None,
    pitcher_pc: pd.DataFrame | None = None,
    batter_pc: pd.DataFrame | None = None,
    tto_profiles: pd.DataFrame | None = None,
    exit_tendencies: pd.DataFrame | None = None,
    hitter_k_samples: dict[str, np.ndarray] | None = None,
    hitter_bb_samples: dict[str, np.ndarray] | None = None,
    hitter_hr_samples: dict[str, np.ndarray] | None = None,
    h_arch_lookup: dict[int, str] | None = None,
    pos_lookup: dict[int, str] | None = None,
    sides_all: list[dict] | None = None,
) -> None:
    """Multi-stat game simulator | pitcher and batter posteriors."""
    from lib.game_sim.simulator import simulate_game
    from lib.game_sim.batter_simulator import simulate_batter_game
    from lib.game_sim.exit_model import ExitModel
    from lib.game_sim.tto_model import build_all_tto_lifts
    from lib.game_sim.pitch_count_model import build_pitch_count_features
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    hitter_k_samples = hitter_k_samples or {}
    hitter_bb_samples = hitter_bb_samples or {}
    hitter_hr_samples = hitter_hr_samples or {}
    h_arch_lookup = h_arch_lookup or {}
    pos_lookup = pos_lookup or {}
    _all_sides = sides_all if sides_all is not None else sides

    if not k_samples_dict:
        st.info("K% posterior samples are not yet available.")
        return

    bb_samples_dict = bb_samples_dict or {}
    hr_samples_dict = hr_samples_dict or {}
    ctx = game_context or {}
    pitcher_pc = pitcher_pc if pitcher_pc is not None else pd.DataFrame()
    batter_pc = batter_pc if batter_pc is not None else pd.DataFrame()
    tto_profiles_df = tto_profiles if tto_profiles is not None else pd.DataFrame()
    exit_tend_df = exit_tendencies if exit_tendencies is not None else pd.DataFrame()

    # Ensure we have an ExitModel instance
    if exit_model is None:
        exit_model = ExitModel()

    # Game context bar
    context_parts = []
    venue = ctx.get("venue_name")
    if venue:
        context_parts.append(f'<span style="color:var(--tdd-cream);">{venue}</span>')
    ump_name = ctx.get("ump_name")
    if ump_name:
        ump_detail = ctx.get("ump_detail", "")
        ump_lift = ctx.get("ump_k_lift", 0.0)
        ump_color = POSITIVE if ump_lift > 0.02 else NEGATIVE if ump_lift < -0.02 else SLATE
        detail_html = f' <span style="color:{ump_color};">({ump_detail})</span>' if ump_detail else ""
        context_parts.append(f'HP: {ump_name}{detail_html}')
    wx_display = ctx.get("weather_display")
    if wx_display:
        wx_detail = ctx.get("wx_detail", "")
        wx_html = f' <span style="color:var(--tdd-slate);">({wx_detail})</span>' if wx_detail else ""
        context_parts.append(f'{wx_display}{wx_html}')

    if context_parts:
        st.markdown(
            f'<div style="background:transparent; border:none; '
            f'border-bottom:1px solid var(--tdd-dark-border); padding:0.6rem 0; margin-bottom:1rem; '
            f'font-size:0.85rem; color:var(--tdd-slate);">'
            f'{" &nbsp;|&nbsp; ".join(context_parts)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    ump_k_lift = ctx.get("ump_k_lift", 0.0)
    wx_k_lift = ctx.get("wx_k_lift", 0.0)

    # Stat display config
    _STAT_META = {
        "k":    {"label": "K",    "word": "strikeouts", "lines": (3.5, 10.5), "hi": 6, "vhi": 8},
        "bb":   {"label": "BB",   "word": "walks",      "lines": (1.5, 5.5),  "hi": 3, "vhi": 4},
        "h":    {"label": "H",    "word": "hits",       "lines": (3.5, 9.5),  "hi": 6, "vhi": 8},
        "hr":   {"label": "HR",   "word": "home runs",  "lines": (0.5, 2.5),  "hi": 1, "vhi": 2},
        "outs": {"label": "Outs", "word": "outs",       "lines": (11.5, 21.5), "hi": 17, "vhi": 20},
    }

    # Helper: fallback rate samples
    _rng = np.random.default_rng(99)

    def _fallback(rate: float, n: int = 4000) -> np.ndarray:
        r = np.clip(rate, 0.01, 0.99)
        return _rng.beta(r * 200, (1 - r) * 200, size=n).astype(np.float32)

    def _pitcher_avg_pitches(pid: int) -> float:
        if exit_tend_df.empty:
            return 88.0
        row = exit_tend_df[
            (exit_tend_df["pitcher_id"] == pid)
            & (exit_tend_df["season"] == PRIOR_SEASON)
        ]
        return float(row.iloc[0]["avg_pitches"]) if not row.empty else 88.0

    rendered = False
    for side_info in sides:
        pid = side_info["pitcher_id"]
        if not pid or str(pid) not in k_samples_dict:
            continue

        rendered = True
        pitcher_name = side_info["pitcher_name"]
        side_abbr = side_info["abbr"]
        opp_abbr = side_info["opp_abbr"]
        opp_lu = side_info["opp_lineup"]
        p_proj = side_info["pitcher_proj"]

        pid_str = str(pid)
        k_samp = k_samples_dict[pid_str]
        proj_bb = float(p_proj.get("projected_bb_rate", 0.08) or 0.08)
        proj_hr = float(p_proj.get("projected_hr_per_bf", 0.03) or 0.03)
        _bb = bb_samples_dict.get(pid_str)
        bb_samp = _bb if _bb is not None else _fallback(proj_bb)
        _hr = hr_samples_dict.get(pid_str)
        hr_samp = _hr if _hr is not None else _fallback(proj_hr)

        # Compute per-batter matchup lifts
        lineup_matchup_lifts: dict[str, np.ndarray] = {}
        per_batter_details: list[dict] = []
        lineup_batter_ids: list[int] = []
        has_lineup = False

        if not opp_lu.empty and not arsenal_df.empty and not vuln_df.empty:
            id_col = "batter_id" if "batter_id" in opp_lu.columns else "player_id"
            name_col = "batter_name" if "batter_name" in opp_lu.columns else "player_name"

            k_lifts, bb_lifts, hr_lifts = [], [], []
            for _, brow in opp_lu.head(9).iterrows():
                bid = int(brow[id_col]) if pd.notna(brow.get(id_col)) else None
                if bid:
                    lineup_batter_ids.append(bid)
                    k_m = score_matchup(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    kl = k_m.get("matchup_k_logit_lift", 0.0)
                    k_lifts.append(0.0 if np.isnan(kl) else kl)

                    bb_m = score_matchup_bb(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    bl = bb_m.get("matchup_bb_logit_lift", 0.0)
                    bb_lifts.append(0.0 if np.isnan(bl) else bl)

                    hr_m = score_matchup_hr(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    hl = hr_m.get("matchup_hr_logit_lift", 0.0)
                    hr_lifts.append(0.0 if np.isnan(hl) else hl)

                    k_m["batter_name"] = brow.get(name_col, "Unknown")
                    k_m["batting_order"] = int(brow["batting_order"])
                    per_batter_details.append(k_m)
                else:
                    lineup_batter_ids.append(0)
                    k_lifts.append(0.0)
                    bb_lifts.append(0.0)
                    hr_lifts.append(0.0)

            while len(k_lifts) < 9:
                k_lifts.append(0.0)
                bb_lifts.append(0.0)
                hr_lifts.append(0.0)
                lineup_batter_ids.append(0)

            has_lineup = True
            lineup_matchup_lifts = {
                "k": np.array(k_lifts[:9]),
                "bb": np.array(bb_lifts[:9]),
                "hr": np.array(hr_lifts[:9]),
            }

        # TTO lifts
        tto_lifts = build_all_tto_lifts(
            tto_profiles_df if not tto_profiles_df.empty else None,
            pid, PRIOR_SEASON,
        )

        # Pitch count features
        pitcher_ppa_adj = 0.0
        batter_ppa_adjs = np.zeros(9)
        if not pitcher_pc.empty and not batter_pc.empty and lineup_batter_ids:
            pitcher_ppa_adj, batter_ppa_adjs = build_pitch_count_features(
                pitcher_pc, batter_pc, pid, lineup_batter_ids[:9], PRIOR_SEASON,
            )

        avg_pitches = _pitcher_avg_pitches(pid)

        # Run PA-by-PA simulation
        result = simulate_game(
            pitcher_k_rate_samples=k_samp,
            pitcher_bb_rate_samples=bb_samp,
            pitcher_hr_rate_samples=hr_samp,
            lineup_matchup_lifts=lineup_matchup_lifts,
            tto_lifts=tto_lifts,
            pitcher_ppa_adj=pitcher_ppa_adj,
            batter_ppa_adjs=batter_ppa_adjs,
            exit_model=exit_model,
            pitcher_avg_pitches=avg_pitches,
            umpire_k_lift=ump_k_lift,
            weather_k_lift=wx_k_lift,
            n_sims=10_000,
            random_seed=42 + gpk + (0 if side_info["side"] == "away" else 1),
        )

        # Pitcher header
        arch = side_info.get("pitcher_arch")
        arch_tag = f" ({arch})" if arch else ""
        if has_lineup:
            lineup_tag = (
                f'vs <span class="tdd-team-abbr" data-team="{opp_abbr}">{opp_abbr}</span> lineup'
            )
        else:
            lineup_tag = "league-avg baseline"

        exp_parts = [
            f"E[K]: {np.mean(result.k_samples):.1f}",
            f"BB: {np.mean(result.bb_samples):.1f}",
            f"H: {np.mean(result.h_samples):.1f}",
            f"HR: {np.mean(result.hr_samples):.1f}",
            f"IP: {format_ip(np.mean(result.outs_samples) / 3.0)}",
        ]

        st.markdown(
            f'<div style="color:var(--tdd-gold); font-size:1rem; font-weight:600; '
            f'margin:0.8rem 0 0.3rem;">'
            f'<span class="tdd-team-abbr" data-team="{side_abbr}">{side_abbr}</span>'
            f' SP: {pitcher_name}{arch_tag}'
            f'<span style="color:var(--tdd-slate); font-size:0.85rem; font-weight:400;">'
            f' | {" | ".join(exp_parts)} | {lineup_tag}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # --- Inline toolbar: Stat + Line ---
        _side_key = f"{side_info['side']}_{gpk}"
        tb_stat, tb_line, _ = st.columns([1, 2, 3])
        with tb_stat:
            selected_stat_label = st.selectbox(
                "Stat", ["K", "BB", "H", "HR", "Outs"],
                key=f"sim_stat_{_side_key}", label_visibility="collapsed",
            )
        stat_key = selected_stat_label.lower()
        stat_samples = getattr(result, f"{stat_key}_samples")
        meta = _STAT_META[stat_key]

        # Build line options for this stat
        lo, hi = meta["lines"]
        line_opts = [x + 0.5 for x in range(int(lo - 0.5), int(hi - 0.5) + 1)]
        line_labels = [f"Over {v:.1f}" for v in line_opts]
        # Default to line closest to (but below) the mean
        _mean = float(np.mean(stat_samples))
        _default_idx = 0
        for i, v in enumerate(line_opts):
            if v <= _mean:
                _default_idx = i
        with tb_line:
            selected_line = st.selectbox(
                "Line", line_labels, index=_default_idx,
                key=f"sim_line_{_side_key}", label_visibility="collapsed",
            )
        threshold = line_opts[line_labels.index(selected_line)]

        # Compute probabilities
        p_over = float(np.mean(stat_samples > threshold))
        p_under = 1.0 - p_over

        # Signal classification
        if p_over > 0.65:
            signal, sig_color = "Strong Over", SAGE
        elif p_over > 0.55:
            signal, sig_color = "Lean Over", SAGE
        elif p_over < 0.35:
            signal, sig_color = "Strong Under", EMBER
        elif p_over < 0.45:
            signal, sig_color = "Lean Under", EMBER
        else:
            signal, sig_color = "Toss-up", SLATE

        # --- Two-column layout: chart + scouting notes ---
        chart_col, notes_col = st.columns([3, 2])

        with chart_col:
            import plotly.graph_objects as go
            cfg = _STAT_CHART_CONFIG.get(stat_key, _STAT_CHART_CONFIG["k"])
            bar_color = cfg["color"]
            max_val = int(stat_samples.max()) + 1
            bins = np.arange(0, max_val + 2)
            counts, edges = np.histogram(stat_samples, bins=bins - 0.5, density=True)

            over_counts = np.where(bins[:-1] > threshold, counts, 0)
            under_counts = np.where(bins[:-1] <= threshold, counts, 0)

            pfig = go.Figure()
            pfig.add_trace(go.Bar(
                x=bins[:-1], y=under_counts, name="Under",
                marker_color=f"rgba(123,143,166,0.53)", width=0.85,
                hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
            ))
            pfig.add_trace(go.Bar(
                x=bins[:-1], y=over_counts, name="Over",
                marker_color=bar_color, width=0.85,
                hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
            ))
            # Threshold line
            pfig.add_vline(
                x=threshold, line_dash="dash", line_color=GOLD, line_width=2,
                annotation_text=f"P(Over {threshold:.1f}) = {p_over:.1%}",
                annotation_position="top left",
                annotation_font=dict(color=GOLD, size=13),
            )
            pfig.update_layout(
                barmode="stack", showlegend=False, dragmode=False,
                xaxis_title=cfg["xlabel"], yaxis_title="",
                yaxis=dict(showticklabels=False, showgrid=False, fixedrange=True),
                xaxis=dict(dtick=1, gridcolor="rgba(0,0,0,0)", fixedrange=True),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=CREAM),
                margin=dict(l=10, r=10, t=30, b=40),
                height=300,
            )
            st.plotly_chart(pfig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

        with notes_col:
            mean_val = float(np.mean(stat_samples))
            std_val = float(np.std(stat_samples))
            p_hi = float((stat_samples >= meta["hi"]).sum() / len(stat_samples) * 100)
            p_vhi = float((stat_samples >= meta["vhi"]).sum() / len(stat_samples) * 100)

            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<span style="background:{sig_color}22; color:{sig_color}; '
                f'border:1px solid {sig_color}44; padding:4px 12px; border-radius:12px; '
                f'font-size:0.9rem; font-weight:700;">{signal}</span>'
                f'</div>'
                f'<div style="display:flex; gap:1.5rem; margin-bottom:1rem;">'
                f'<div>'
                f'<div style="color:var(--tdd-cream); font-size:1.4rem; font-weight:700;">{p_over:.1%}</div>'
                f'<div style="color:var(--tdd-slate); font-size:0.75rem;">P(Over {threshold:.1f})</div>'
                f'</div>'
                f'<div>'
                f'<div style="color:var(--tdd-cream); font-size:1.4rem; font-weight:700;">{p_under:.1%}</div>'
                f'<div style="color:var(--tdd-slate); font-size:0.75rem;">P(Under {threshold:.1f})</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-bullet">'
                f'<span class="dot" style="background:{GOLD};"></span>'
                f'Expect around <strong>{mean_val:.0f} {meta["word"]}</strong> '
                f'(\u00b1{std_val:.0f}).'
                f'</div>'
                f'<div class="insight-bullet">'
                f'<span class="dot" style="background:{SAGE};"></span>'
                f'<strong>{p_hi:.0f}%</strong> chance of {meta["hi"]}+ {meta["word"]}, '
                f'<strong>{p_vhi:.0f}%</strong> chance of {meta["vhi"]}+.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


        # --- Batter Projections (only when actual lineup is set) ---
        opp_lu = side_info["opp_lineup"]
        if not opp_lu.empty:
            name_col_b = "batter_name" if "batter_name" in opp_lu.columns else "player_name"
            id_col_b = "batter_id" if "batter_id" in opp_lu.columns else "player_id"

            # Find opposing pitcher for context
            opp_side = next((s for s in _all_sides if s["side"] != side_info["side"]), None)
            opp_pid = opp_side["pitcher_id"] if opp_side else None
            opp_proj = opp_side["pitcher_proj"] if opp_side else {}
            opp_pitcher_name = opp_side["pitcher_name"] if opp_side else "TBD"

            opp_k_rate = float(opp_proj.get("projected_k_rate", 0.22) or 0.22)
            opp_bb_rate = float(opp_proj.get("projected_bb_rate", 0.08) or 0.08)
            opp_hr_rate = float(opp_proj.get("projected_hr_per_bf", 0.03) or 0.03)

            opp_bf_mu, opp_bf_sigma = 22.0, 4.5
            if opp_pid and not bf_priors.empty:
                bp_row = bf_priors[bf_priors["pitcher_id"] == opp_pid]
                if not bp_row.empty:
                    bp_last = bp_row.sort_values("season").iloc[-1]
                    opp_bf_mu = float(bp_last["mu_bf"])
                    opp_bf_sigma = float(bp_last["sigma_bf"])

            # Build batter options
            batter_opts = []
            for _, brow in opp_lu.head(9).iterrows():
                bid = int(brow[id_col_b]) if pd.notna(brow.get(id_col_b)) else None
                bname = brow.get(name_col_b, "Unknown")
                order = int(brow["batting_order"])
                if bid:
                    batter_opts.append((f"{order}. {bname}", bid, order))

            if batter_opts:
                st.markdown(
                    f'<div class="tdd-section-hdr" style="margin-top:1rem;">'
                    f'<span class="tdd-team-abbr" data-team="{opp_abbr}">{opp_abbr}</span>'
                    f' Batters vs {pitcher_name}</div>',
                    unsafe_allow_html=True,
                )

                sel_batter = st.selectbox(
                    "Batter", [b[0] for b in batter_opts],
                    key=f"sim_batter_{_side_key}", label_visibility="collapsed",
                )
                sel_bid = next(b[1] for b in batter_opts if b[0] == sel_batter)
                sel_order = next(b[2] for b in batter_opts if b[0] == sel_batter)
                sel_bname = sel_batter.split(". ", 1)[1]

                bid_key = str(sel_bid)
                has_batter_samp = (
                    bid_key in hitter_k_samples
                    and bid_key in hitter_bb_samples
                    and bid_key in hitter_hr_samples
                    and opp_pid is not None
                )

                if has_batter_samp:
                    try:
                        # Matchup lifts
                        mk_lift = mb_lift = mh_lift = 0.0
                        if opp_pid and not arsenal_df.empty and not vuln_df.empty:
                            mk_lift = score_matchup(opp_pid, sel_bid, arsenal_df, vuln_df, baselines_pt).get("matchup_k_logit_lift", 0.0)
                            mb_lift = score_matchup_bb(opp_pid, sel_bid, arsenal_df, vuln_df, baselines_pt).get("matchup_bb_logit_lift", 0.0)
                            mh_lift = score_matchup_hr(opp_pid, sel_bid, arsenal_df, vuln_df, baselines_pt).get("matchup_hr_logit_lift", 0.0)

                        b_result = simulate_batter_game(
                            batter_k_rate_samples=hitter_k_samples[bid_key],
                            batter_bb_rate_samples=hitter_bb_samples[bid_key],
                            batter_hr_rate_samples=hitter_hr_samples[bid_key],
                            batting_order=sel_order,
                            starter_k_rate=opp_k_rate,
                            starter_bb_rate=opp_bb_rate,
                            starter_hr_rate=opp_hr_rate,
                            starter_bf_mu=opp_bf_mu,
                            starter_bf_sigma=opp_bf_sigma,
                            matchup_k_lift=mk_lift if not np.isnan(mk_lift) else 0.0,
                            matchup_bb_lift=mb_lift if not np.isnan(mb_lift) else 0.0,
                            matchup_hr_lift=mh_lift if not np.isnan(mh_lift) else 0.0,
                            bullpen_k_rate=0.23,
                            bullpen_bb_rate=0.09,
                            bullpen_hr_rate=0.03,
                            n_sims=5_000,
                            random_seed=gpk + sel_bid,
                        )

                        # Batter stat/line toolbar
                        _BATTER_STAT_META = {
                            "k":  {"label": "K",  "word": "strikeouts", "lines": (0.5, 3.5), "hi": 2, "vhi": 3},
                            "bb": {"label": "BB", "word": "walks",      "lines": (0.5, 2.5), "hi": 1, "vhi": 2},
                            "h":  {"label": "H",  "word": "hits",       "lines": (0.5, 3.5), "hi": 2, "vhi": 3},
                            "hr": {"label": "HR", "word": "home runs",  "lines": (0.5, 1.5), "hi": 1, "vhi": 2},
                        }

                        btb_stat, btb_line, _ = st.columns([1, 2, 3])
                        with btb_stat:
                            b_stat_label = st.selectbox(
                                "Stat", ["H", "HR", "K", "BB"],
                                key=f"bsim_stat_{_side_key}", label_visibility="collapsed",
                            )
                        b_stat_key = b_stat_label.lower()
                        b_samples = getattr(b_result, f"{b_stat_key}_samples")
                        b_meta = _BATTER_STAT_META[b_stat_key]

                        blo, bhi = b_meta["lines"]
                        b_line_opts = [x + 0.5 for x in range(int(blo - 0.5), int(bhi - 0.5) + 1)]
                        b_line_labels = [f"Over {v:.1f}" for v in b_line_opts]
                        b_mean = float(np.mean(b_samples))
                        b_def_idx = 0
                        for bi, bv in enumerate(b_line_opts):
                            if bv <= b_mean:
                                b_def_idx = bi
                        with btb_line:
                            b_sel_line = st.selectbox(
                                "Line", b_line_labels, index=b_def_idx,
                                key=f"bsim_line_{_side_key}", label_visibility="collapsed",
                            )
                        b_threshold = b_line_opts[b_line_labels.index(b_sel_line)]

                        b_p_over = float(np.mean(b_samples > b_threshold))
                        b_p_under = 1.0 - b_p_over

                        if b_p_over > 0.65:
                            b_signal, b_sig_color = "Strong Over", SAGE
                        elif b_p_over > 0.55:
                            b_signal, b_sig_color = "Lean Over", SAGE
                        elif b_p_over < 0.35:
                            b_signal, b_sig_color = "Strong Under", EMBER
                        elif b_p_over < 0.45:
                            b_signal, b_sig_color = "Lean Under", EMBER
                        else:
                            b_signal, b_sig_color = "Toss-up", SLATE

                        # Two-column: chart + notes
                        b_chart_col, b_notes_col = st.columns([3, 2])

                        with b_chart_col:
                            import plotly.graph_objects as go
                            b_cfg = _STAT_CHART_CONFIG.get(b_stat_key, _STAT_CHART_CONFIG["k"])
                            b_max_val = int(b_samples.max()) + 1
                            b_bins = np.arange(0, b_max_val + 2)
                            b_counts, _ = np.histogram(b_samples, bins=b_bins - 0.5, density=True)
                            b_over_c = np.where(b_bins[:-1] > b_threshold, b_counts, 0)
                            b_under_c = np.where(b_bins[:-1] <= b_threshold, b_counts, 0)

                            bfig = go.Figure()
                            bfig.add_trace(go.Bar(
                                x=b_bins[:-1], y=b_under_c, name="Under",
                                marker_color="rgba(123,143,166,0.53)", width=0.85,
                                hovertemplate="%{x} " + b_meta["label"] + ": %{y:.1%}<extra></extra>",
                            ))
                            bfig.add_trace(go.Bar(
                                x=b_bins[:-1], y=b_over_c, name="Over",
                                marker_color=b_cfg["color"], width=0.85,
                                hovertemplate="%{x} " + b_meta["label"] + ": %{y:.1%}<extra></extra>",
                            ))
                            bfig.add_vline(
                                x=b_threshold, line_dash="dash", line_color=GOLD, line_width=2,
                                annotation_text=f"P(Over {b_threshold:.1f}) = {b_p_over:.1%}",
                                annotation_position="top left",
                                annotation_font=dict(color=GOLD, size=13),
                            )
                            bfig.update_layout(
                                barmode="stack", showlegend=False, dragmode=False,
                                xaxis_title=b_cfg["xlabel"], yaxis_title="",
                                yaxis=dict(showticklabels=False, showgrid=False, fixedrange=True),
                                xaxis=dict(dtick=1, gridcolor="rgba(0,0,0,0)", fixedrange=True),
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=CREAM),
                                margin=dict(l=10, r=10, t=30, b=40),
                                height=280,
                            )
                            st.plotly_chart(bfig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

                        with b_notes_col:
                            b_std = float(np.std(b_samples))
                            b_p_hi = float((b_samples >= b_meta["hi"]).sum() / len(b_samples) * 100)
                            b_p_vhi = float((b_samples >= b_meta["vhi"]).sum() / len(b_samples) * 100)

                            st.markdown(
                                f'<div style="margin-bottom:1rem;">'
                                f'<span style="background:{b_sig_color}22; color:{b_sig_color}; '
                                f'border:1px solid {b_sig_color}44; padding:4px 12px; border-radius:12px; '
                                f'font-size:0.9rem; font-weight:700;">{b_signal}</span>'
                                f'</div>'
                                f'<div style="display:flex; gap:1.5rem; margin-bottom:1rem;">'
                                f'<div>'
                                f'<div style="color:var(--tdd-cream); font-size:1.4rem; font-weight:700;">{b_p_over:.1%}</div>'
                                f'<div style="color:var(--tdd-slate); font-size:0.75rem;">P(Over {b_threshold:.1f})</div>'
                                f'</div>'
                                f'<div>'
                                f'<div style="color:var(--tdd-cream); font-size:1.4rem; font-weight:700;">{b_p_under:.1%}</div>'
                                f'<div style="color:var(--tdd-slate); font-size:0.75rem;">P(Under {b_threshold:.1f})</div>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                f'<div class="insight-card">'
                                f'<div class="insight-bullet">'
                                f'<span class="dot" style="background:{GOLD};"></span>'
                                f'Expect around <strong>{b_mean:.1f} {b_meta["word"]}</strong> '
                                f'(\u00b1{b_std:.1f}).'
                                f'</div>'
                                f'<div class="insight-bullet">'
                                f'<span class="dot" style="background:{SAGE};"></span>'
                                f'<strong>{b_p_hi:.0f}%</strong> chance of {b_meta["hi"]}+ {b_meta["word"]}, '
                                f'<strong>{b_p_vhi:.0f}%</strong> chance of {b_meta["vhi"]}+.'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    except Exception:
                        st.info(f"Simulation data not available for {sel_bname}.")
                else:
                    st.info(f"No posterior samples available for {sel_bname}.")

        st.markdown("---")

    if not rendered:
        st.info("No K% samples available for this game's pitchers.")


def _render_game_browser() -> None:
    """Browse historical games grouped by game_pk with team and pitcher selectors."""
    from lib.matchup import score_matchup
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    st.caption(
        f"Browse {PRIOR_SEASON} games by team. Select a game, then view each pitcher's "
        "matchup breakdown against the opposing lineup."
    )

    game_logs_path = DASHBOARD_DIR / "pitcher_game_logs.parquet"
    lineups_path = DASHBOARD_DIR / "game_lineups.parquet"
    batter_ks_path = DASHBOARD_DIR / "game_batter_ks.parquet"

    if not all(p.exists() for p in [game_logs_path, lineups_path, batter_ks_path]):
        st.warning("Game browser data is not yet available.")
        return

    game_logs = pd.read_parquet(game_logs_path)
    all_lineups = pd.read_parquet(lineups_path)
    all_batter_ks = pd.read_parquet(batter_ks_path)

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    if arsenal_df.empty or vuln_df.empty:
        st.warning("Matchup profile data not found.")
        return

    @st.cache_data
    def _enrich_game_logs(_game_logs: pd.DataFrame) -> pd.DataFrame:
        game_info = load_game_info()
        if game_info.empty:
            return _game_logs
        return _game_logs.merge(game_info, on="game_pk", how="left")

    game_logs = _enrich_game_logs(game_logs)

    if "game_date" not in game_logs.columns:
        st.warning("Game info is not yet available.")
        return

    team_lookup = get_team_lookup()

    all_team_names = set()
    if "home_team_name" in game_logs.columns:
        all_team_names.update(game_logs["home_team_name"].dropna().unique())
    if "away_team_name" in game_logs.columns:
        all_team_names.update(game_logs["away_team_name"].dropna().unique())
    all_team_names = sorted(t for t in all_team_names if t)

    selected_team = st.selectbox(
        "Select team", all_team_names, key="gb_team"
    )

    team_games = game_logs[
        (game_logs["home_team_name"] == selected_team) |
        (game_logs["away_team_name"] == selected_team)
    ].copy()

    if team_games.empty:
        st.info("No games found for this team.")
        return

    game_summary = (
        team_games.groupby("game_pk")
        .agg(
            game_date=("game_date", "first"),
            home_team_name=("home_team_name", "first"),
            away_team_name=("away_team_name", "first"),
        )
        .reset_index()
        .sort_values("game_date", ascending=False)
    )

    game_options = {}
    for _, g in game_summary.iterrows():
        gpk = int(g["game_pk"])
        date_str = str(g["game_date"])[:10]
        home = g["home_team_name"] or "?"
        away = g["away_team_name"] or "?"
        label = f"{date_str} | {away} @ {home}"
        game_options[label] = gpk

    selected_game_label = st.selectbox(
        "Select Game", list(game_options.keys()), key="gb_game",
    )
    selected_gpk = game_options[selected_game_label]

    game_pitchers = game_logs[game_logs["game_pk"] == selected_gpk].copy()
    if game_pitchers.empty:
        st.info("No pitcher data for this game.")
        return

    game_meta = game_summary[game_summary["game_pk"] == selected_gpk].iloc[0]
    home_name = game_meta["home_team_name"] or "Home"
    away_name = game_meta["away_team_name"] or "Away"

    pitcher_opts = {}
    for _, pr in game_pitchers.sort_values(
        ["is_starter", "pitcher_name"], ascending=[False, True]
    ).iterrows():
        pid = int(pr["pitcher_id"])
        pname = pr["pitcher_name"]
        team = team_lookup.get(pid, "")
        role = "SP" if pr.get("is_starter") else "RP"
        ks = int(pr["strike_outs"]) if pd.notna(pr.get("strike_outs")) else 0
        ip = pr.get("innings_pitched", 0)
        dname = f"{pname} ({team}, {role}) | {ks} K, {ip} IP" if team else f"{pname} ({role}) | {ks} K, {ip} IP"
        pitcher_opts[dname] = pid

    selected_pitcher_display = st.selectbox(
        "Select Pitcher", list(pitcher_opts.keys()), key="gb_pitcher",
    )
    pitcher_id = pitcher_opts[selected_pitcher_display]

    game_row = game_pitchers[game_pitchers["pitcher_id"] == pitcher_id].iloc[0]
    game_lineups_this = all_lineups[all_lineups["game_pk"] == selected_gpk]

    if game_lineups_this.empty:
        st.warning("No lineup data found for this game.")
        return

    bk_game = all_batter_ks[
        (all_batter_ks["game_pk"] == selected_gpk) &
        (all_batter_ks["pitcher_id"] == pitcher_id)
    ]
    faced_batters = set(bk_game["batter_id"].tolist())
    home_tid = game_row.get("home_team_id")

    opposing_lineup = None
    opponent_name = ""
    for tid in game_lineups_this["team_id"].unique():
        team_lineup = game_lineups_this[
            game_lineups_this["team_id"] == tid
        ].sort_values("batting_order")
        lineup_batters = set(team_lineup["player_id"].tolist())
        if lineup_batters & faced_batters:
            opposing_lineup = team_lineup
            opponent_name = home_name if tid == home_tid else away_name
            break

    if opposing_lineup is None or opposing_lineup.empty:
        st.warning("Could not determine opposing lineup for this pitcher.")
        return

    actual_ks_game = bk_game
    actual_k_map = dict(zip(actual_ks_game["batter_id"], actual_ks_game["k"]))
    actual_pa_map = dict(zip(actual_ks_game["batter_id"], actual_ks_game["pa"]))

    baselines_pt = {
        pt: {"whiff_rate": vals.get("whiff_rate", 0.25)}
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    display_rows = []
    total_actual_k = int(game_row.get("strike_outs", 0))
    total_matchup_lift = 0.0
    n_scored = 0

    for _, brow in opposing_lineup.iterrows():
        bid = int(brow["player_id"])
        bname = brow.get("batter_name", "Unknown")
        order = int(brow["batting_order"])
        bteam = team_lookup.get(bid, "")

        matchup = score_matchup(
            pitcher_id=pitcher_id,
            batter_id=bid,
            pitcher_arsenal=arsenal_df,
            hitter_vuln=vuln_df,
            baselines_pt=baselines_pt,
        )

        lift = matchup.get("matchup_k_logit_lift", 0.0)
        if np.isnan(lift):
            lift = 0.0
        mwhiff = matchup.get("matchup_whiff_rate", np.nan)
        bwhiff = matchup.get("baseline_whiff_rate", np.nan)
        reliability = matchup.get("avg_reliability", 0.0)

        actual_k = actual_k_map.get(bid, 0)
        actual_pa = actual_pa_map.get(bid, 0)

        if not np.isnan(lift):
            total_matchup_lift += lift
            n_scored += 1

        row = {
            "#": order,
            "Batter": f"{bname} ({bteam})" if bteam else bname,
            "Matchup Whiff%": f"{mwhiff:.1%}" if pd.notna(mwhiff) else "--",
            "Baseline Whiff%": f"{bwhiff:.1%}" if pd.notna(bwhiff) else "--",
            "K Lift": f"{lift:+.3f}" if lift != 0 else "0.000",
            "Reliability": f"{reliability:.0%}",
            "PA": actual_pa,
            "K": actual_k,
        }
        display_rows.append(row)

    date_str = str(game_meta["game_date"])[:10]
    ip = game_row.get("innings_pitched", 0)
    bf = int(game_row.get("batters_faced", 0)) if pd.notna(game_row.get("batters_faced")) else 0
    avg_lift = total_matchup_lift / n_scored if n_scored > 0 else 0.0
    pitcher_name = game_row.get("pitcher_name", "Unknown")
    pitcher_team = team_lookup.get(pitcher_id, "")

    pitcher_team_tag = (
        f' (<span class="tdd-team-abbr" data-team="{pitcher_team}">{pitcher_team}</span>)'
        if pitcher_team else ""
    )
    gb_header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title">{pitcher_name}{pitcher_team_tag}</div>'
        f'<div class="brand-subtitle">{date_str} | {away_name} @ {home_name} | {ip} IP, {bf} BF</div>'
        f'</div>'
        f'<div style="font-size:1.2rem; font-weight:600;">'
        f'<span style="color:var(--tdd-gold);">{total_actual_k} K</span>'
        f'<span style="color:var(--tdd-slate);"> | Avg Lift: {avg_lift:+.3f}</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(gb_header_html, unsafe_allow_html=True)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
    )

    lineup_ks = sum(r["K"] for r in display_rows)
    lineup_pa = sum(r["PA"] for r in display_rows)
    k_rate_actual = lineup_ks / lineup_pa if lineup_pa > 0 else 0

    if avg_lift > 0.05:
        lift_color = POSITIVE
        lift_word = "favorable"
    elif avg_lift < -0.05:
        lift_color = NEGATIVE
        lift_word = "unfavorable"
    else:
        lift_color = SLATE
        lift_word = "neutral"

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-bullet">
            <span class="dot" style="background:{GOLD};"></span>
            Actual: <strong>{total_actual_k} K</strong> in {bf} BF
            ({k_rate_actual:.1%} K rate)
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{lift_color};"></span>
            Matchup model rated this lineup as
            <strong style="color:{lift_color};">{lift_word}</strong>
            for strikeouts (avg logit lift: {avg_lift:+.3f})
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SLATE};"></span>
            K Lift = matchup-driven K% advantage above baseline (logit scale).
            Positive = hitter is more vulnerable to this pitcher's arsenal.
            Negative = hitter handles it better than average.
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_schedule() -> None:
    """Schedule page | browse games by date with projections."""
    st.markdown(
        '<div class="section-header">Schedule</div>'
        '<div class="tdd-meta" style="margin-top:-0.5rem; margin-bottom:0.8rem;">'
        'Pregame analytics &amp; projections | not a live scores app.</div>',
        unsafe_allow_html=True,
    )

    # Use ET date as "today" (MLB games are scheduled in ET)
    meta = load_update_metadata()
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

    if is_today:
        # Today: use cached parquets if they match today's date
        schedule = load_todays_games()
        cached_date = (
            schedule["game_date"].iloc[0]
            if not schedule.empty and "game_date" in schedule.columns
            else None
        )
        if cached_date == today.isoformat():
            sims = load_game_props()
            lineups = load_todays_lineups()
            lineups = backfill_missing_lineups(schedule, lineups)
        else:
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

    _render_schedule_cards(schedule, sims, lineups, meta)
