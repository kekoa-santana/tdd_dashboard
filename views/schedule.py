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
    load_bf_priors,
    load_roster, load_game_props,
    fetch_live_schedule, fetch_live_lineups, fetch_live_boxscores,
    backfill_missing_lineups,
    load_batter_platoon_splits, load_batter_glicko, load_pitcher_glicko,
    load_pitcher_gb_pct, load_matchup_baselines,
    load_pitcher_game_sim_samples, load_batter_game_sim_samples,
)
from utils.helpers import format_ip, format_game_time, get_team_lookup
from components.diamond_rating import diamond_rating_html
from components.expandable_card import EXPANDABLE_CARD_CSS, expandable_card_html
from components.team_logo import team_logo_html
from components.headshot import headshot_html
from lib.fantasy_report import load_report_data, get_pitcher_scouting, ReportData


@st.cache_data(ttl=300)
def _get_scouting_report_data() -> ReportData:
    """Load scouting report data (cached 5 min)."""
    return load_report_data(DASHBOARD_DIR)


def _render_scouting_html(report) -> None:
    """Render a PitcherReport's scouting bullets as styled HTML."""
    from lib.fantasy_report import PitcherReport

    sections = [
        ("ADVANTAGES", SAGE, report.advantages),
        ("STRUGGLES", EMBER, report.struggles),
        ("BATTERS TO WATCH", GOLD, report.key_batters),
        ("BATTERS WHO WILL STRUGGLE", SLATE, report.struggling_batters),
        ("BULLPEN", SLATE, report.bullpen_notes),
        (f"{report.opp_abbr} BATTING OUTLOOK", GOLD, report.batting_outlook),
    ]

    all_bullets = []
    for title, color, bullets in sections:
        if not bullets:
            continue
        bullet_html = "".join(
            f'<li style="margin-bottom:0.25rem;">{b}</li>' for b in bullets
        )
        all_bullets.append(
            f'<div style="color:{color}; font-weight:600; font-size:0.7rem; '
            f'letter-spacing:0.5px; margin-bottom:0.2rem; margin-top:0.4rem;">'
            f'{title}</div>'
            f'<ul style="margin:0; padding-left:1.2rem; '
            f'color:var(--tdd-cream);">{bullet_html}</ul>'
        )

    if all_bullets:
        st.markdown(
            f'<div style="margin:0.5rem 0 1rem; padding:0.6rem 0.8rem; '
            f'border-left:2px solid {GOLD}; font-size:0.82rem; '
            f'color:var(--tdd-cream);">'
            f'<div style="color:{GOLD}; font-weight:600; '
            f'font-size:0.75rem; letter-spacing:0.5px; margin-bottom:0.3rem;">'
            f'SCOUTING REPORT</div>'
            f'{"".join(all_bullets)}</div>',
            unsafe_allow_html=True,
        )


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


def _lineup_hand_missing_html() -> str:
    return (
        '<span class="tdd-stat-label" style="min-width:1rem; text-align:center; '
        'margin-left:0.15rem; color:var(--tdd-dark-border);">—</span>'
    )


def _lineup_hand_html(hand: object) -> str:
    """L / R / S plate or mound handedness badge; unknown shows an em dash."""
    if hand is None:
        return _lineup_hand_missing_html()
    if isinstance(hand, (float, np.floating)) and pd.isna(hand):
        return _lineup_hand_missing_html()
    s = str(hand).strip().upper()
    if not s or s == "NAN":
        return _lineup_hand_missing_html()
    c = s[0]
    if c not in ("L", "R", "S"):
        return _lineup_hand_missing_html()
    ce = html.escape(c)
    return (
        f'<span class="tdd-stat-label" style="min-width:1rem; text-align:center; '
        f'margin-left:0.15rem; font-weight:700; color:var(--tdd-slate);">{ce}</span>'
    )


def _lineup_pitcher_hand_html(
    p_proj: dict,
    arsenal_df: pd.DataFrame | None,
    pid: int | None,
) -> str:
    """Throws (R/L) from arsenal table, else pitcher projection parquet."""
    hand: object = None
    if (
        pid
        and arsenal_df is not None
        and not arsenal_df.empty
        and "pitch_hand" in arsenal_df.columns
    ):
        sub = arsenal_df[arsenal_df["pitcher_id"] == pid]
        if not sub.empty:
            hand = sub["pitch_hand"].iloc[0]
    if hand is None or (isinstance(hand, float) and pd.isna(hand)):
        hand = (p_proj or {}).get("pitch_hand")
    return _lineup_hand_html(hand)


def _render_schedule_cards(
    schedule: pd.DataFrame,
    sims: pd.DataFrame,
    lineups: pd.DataFrame,
    meta: dict,
    *,
    live_stats: pd.DataFrame | None = None,
) -> None:
    """Render game cards from schedule + sim + lineup data."""
    # Cached lookup dicts (eliminates ~180 lines of iterrows on every rerender)
    _lookups = _build_schedule_lookups()
    _h_arch_lookup = _lookups["h_arch_lookup"]
    _p_arch_lookup = _lookups["p_arch_lookup"]
    _h_stat_lookup = _lookups["h_stat_lookup"]
    _pos_lookup = _lookups["pos_lookup"]
    proj_lookup = _lookups["proj_lookup"]
    _standings = _lookups["standings"]

    # Drilldown data is loaded lazily -- only when a game toggle is active.
    # This avoids ~20 cached-but-expensive data loads on every page render.
    _drilldown_data: dict | None = None

    def _get_drilldown_data() -> dict:
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

    # Pre-group pitcher sims by game_pk (one pass instead of N per-game filters)
    _pitcher_sims_by_gpk: dict[int, pd.DataFrame] = {}
    if not sims.empty and "game_pk" in sims.columns and "player_type" in sims.columns:
        _p_sims = sims[sims["player_type"] == "pitcher"]
        for _gpk_val, _grp in _p_sims.groupby("game_pk"):
            _pitcher_sims_by_gpk[int(_gpk_val)] = _grp

    # Sort games chronologically by start time
    if "game_datetime_utc" in schedule.columns:
        schedule = schedule.sort_values("game_datetime_utc", na_position="last")

    for _, game in schedule.iterrows():
        gpk = game["game_pk"]
        away_abbr = game.get("away_abbr", "?")
        home_abbr = game.get("home_abbr", "?")
        away_tid = game.get("away_team_id")
        home_tid = game.get("home_team_id")
        game_time = format_game_time(
            game.get("game_datetime_utc"), fallback=game.get("game_time", ""),
        )
        game_dt = game.get("game_date", "")
        status = game.get("status", "")

        game_sims = _pitcher_sims_by_gpk.get(int(gpk), pd.DataFrame())

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

        if st.toggle("Matchups & Projections", key=f"expand_{gpk}", value=False):
            _dd = _get_drilldown_data()
            _render_game_drilldown(
                game, lineups, _h_arch_lookup, _p_arch_lookup, _h_stat_lookup,
                _dd["bf_priors"], _dd["arsenal_df"], _dd["vuln_df"],
                proj_lookup, gpk,
                pos_lookup=_pos_lookup,
                str_df=_dd["str_df"],
                game_props=_dd["game_props"],
                live_stats=live_stats,
                batter_sims_df=_dd["batter_sims_df"],
                pitcher_sim_samples=_dd["pitcher_sim_samples"],
                batter_sim_samples=_dd["batter_sim_samples"],
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

    # --- Inline toolbar: Section ---
    _SECTIONS = ["Lineups", "Props Lab"]

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
            st.info(f"Lineup changed since last sim. {', '.join(parts)}. Sims will auto-update shortly.")

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

    elif section == "Props Lab":
        _render_props_section(
            gpk,
            game_props if game_props is not None else pd.DataFrame(),
            lineups_df=game_lu,
            live_stats_df=live_stats,
        )


_PITCHER_STAT_META = {
    "k":    {"label": "K",    "xlabel": "Pitcher Strikeouts", "color": SAGE,  "lines": (3.5, 10.5), "hi": 6, "vhi": 8},
    "bb":   {"label": "BB",   "xlabel": "Walks Issued",      "color": EMBER, "lines": (1.5, 5.5),  "hi": 3, "vhi": 4},
    "h":    {"label": "H",    "xlabel": "Hits Allowed",      "color": SLATE, "lines": (3.5, 9.5),  "hi": 6, "vhi": 8},
    "hr":   {"label": "HR",   "xlabel": "Home Runs Allowed", "color": GOLD,  "lines": (0.5, 2.5),  "hi": 1, "vhi": 2},
    "outs": {"label": "Outs", "xlabel": "Outs Recorded",     "color": SLATE, "lines": (11.5, 21.5), "hi": 17, "vhi": 20},
}

_BATTER_STAT_META = {
    "k":  {"label": "K",  "xlabel": "Batter Strikeouts", "color": EMBER, "lines": (0.5, 3.5), "hi": 2, "vhi": 3},
    "bb": {"label": "BB", "xlabel": "Batter Walks",      "color": SAGE,  "lines": (0.5, 2.5), "hi": 1, "vhi": 2},
    "h":  {"label": "H",  "xlabel": "Batter Hits",       "color": SAGE,  "lines": (0.5, 3.5), "hi": 2, "vhi": 3},
    "hr": {"label": "HR", "xlabel": "Batter Home Runs",  "color": GOLD,  "lines": (0.5, 1.5), "hi": 1, "vhi": 2},
}


@st.fragment
def _render_player_sim(
    samples_dict: dict[str, np.ndarray],
    stat_meta: dict[str, dict],
    stat_options: list[str],
    widget_key: str,
) -> None:
    """Render sim distribution chart + notes for one player.

    This is a nested fragment so that stat/line changes only rerun this
    player's chart, not the entire lineup.
    """
    import plotly.graph_objects as go

    tb_stat, tb_line, _ = st.columns([1, 2, 3])
    with tb_stat:
        stat_label = st.selectbox(
            "Stat", stat_options,
            key=f"psim_stat_{widget_key}", label_visibility="collapsed",
        )
    stat_key = stat_label.lower()
    stat_samples = samples_dict[stat_key]
    meta = stat_meta[stat_key]

    lo, hi = meta["lines"]
    line_opts = [x + 0.5 for x in range(int(lo - 0.5), int(hi - 0.5) + 1)]
    line_labels = [f"Over {v:.1f}" for v in line_opts]
    _mean = float(np.mean(stat_samples))
    _default_idx = 0
    for i, v in enumerate(line_opts):
        if v <= _mean:
            _default_idx = i
    with tb_line:
        sel_line = st.selectbox(
            "Line", line_labels, index=_default_idx,
            key=f"psim_line_{widget_key}", label_visibility="collapsed",
        )
    threshold = line_opts[line_labels.index(sel_line)]

    p_over = float(np.mean(stat_samples > threshold))
    p_under = 1.0 - p_over

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

    bar_color = meta.get("color", SAGE)
    max_val = int(stat_samples.max()) + 1
    bins = np.arange(0, max_val + 2)
    counts, _ = np.histogram(stat_samples, bins=bins - 0.5, density=True)
    over_counts = np.where(bins[:-1] > threshold, counts, 0)
    under_counts = np.where(bins[:-1] <= threshold, counts, 0)

    pfig = go.Figure()
    pfig.add_trace(go.Bar(
        x=bins[:-1], y=under_counts, name="Under",
        marker_color="rgba(123,143,166,0.53)", width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_trace(go.Bar(
        x=bins[:-1], y=over_counts, name="Over",
        marker_color=bar_color, width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_vline(
        x=threshold, line_dash="dash", line_color=GOLD, line_width=2,
        annotation_text=f"P(Over {threshold:.1f}) = {p_over:.1%}",
        annotation_position="top left",
        annotation_font=dict(color=GOLD, size=12),
    )
    pfig.update_layout(
        barmode="stack", showlegend=False, dragmode=False,
        xaxis_title=meta.get("xlabel", meta["label"]), yaxis_title="",
        yaxis=dict(showticklabels=False, showgrid=False, fixedrange=True),
        xaxis=dict(dtick=1, gridcolor="rgba(0,0,0,0)", fixedrange=True, rangemode="nonnegative"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CREAM),
        margin=dict(l=10, r=10, t=30, b=40),
        height=260,
    )
    st.plotly_chart(pfig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    mean_val = float(np.mean(stat_samples))
    std_val = float(np.std(stat_samples))
    p_hi = float((stat_samples >= meta["hi"]).sum() / len(stat_samples) * 100)
    p_vhi = float((stat_samples >= meta["vhi"]).sum() / len(stat_samples) * 100)

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:1rem; margin:0.5rem 0;">'
        f'<span style="background:{sig_color}22; color:{sig_color}; '
        f'border:1px solid {sig_color}44; padding:3px 10px; border-radius:12px; '
        f'font-size:0.82rem; font-weight:700;">{signal}</span>'
        f'<span style="color:var(--tdd-cream); font-size:1rem; font-weight:700;">'
        f'{p_over:.1%}</span>'
        f'<span style="color:var(--tdd-slate); font-size:0.75rem;">'
        f'Over {threshold:.1f}</span>'
        f'<span style="color:var(--tdd-cream); font-size:1rem; font-weight:700;">'
        f'{p_under:.1%}</span>'
        f'<span style="color:var(--tdd-slate); font-size:0.75rem;">'
        f'Under {threshold:.1f}</span>'
        f'</div>'
        f'<div style="color:var(--tdd-slate); font-size:0.78rem; margin-top:0.3rem;">'
        f'E[{meta["label"]}] {mean_val:.1f} (\u00b1{std_val:.1f})'
        f' | {p_hi:.0f}% chance {meta["hi"]}+'
        f' | {p_vhi:.0f}% chance {meta["vhi"]}+'
        f'</div>',
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
# Matchup advantage helpers
# ---------------------------------------------------------------------------

def _load_matchup_aux() -> dict:
    """Load auxiliary data for score_matchup_advantage (cached per session)."""
    platoon_df = load_batter_platoon_splits()
    pitcher_glicko_df = load_pitcher_glicko()
    batter_glicko_df = load_batter_glicko()
    pitcher_gb_df = load_pitcher_gb_pct()
    matchup_baselines = load_matchup_baselines()
    scales = None
    if matchup_baselines:
        scales = {
            "trajectory_scale": matchup_baselines.get("trajectory_scale", 5.0),
            "glicko_scale": matchup_baselines.get("glicko_scale", 0.001),
        }
    return {
        "platoon_df": platoon_df,
        "pitcher_glicko_df": pitcher_glicko_df,
        "batter_glicko_df": batter_glicko_df,
        "pitcher_gb_df": pitcher_gb_df,
        "matchup_baselines": matchup_baselines,
        "scales": scales,
    }


def _build_advantage_kwargs(
    aux: dict,
    pitcher_id: int,
    batter_id: int,
    pitcher_hand: str | None = None,
    str_df: pd.DataFrame | None = None,
    hitter_proj: pd.DataFrame | None = None,
) -> dict:
    """Build keyword arguments for score_matchup_advantage from aux data."""
    kwargs: dict = {}

    # Hitter strength
    if str_df is not None and not str_df.empty:
        kwargs["hitter_str"] = str_df

    # Pitcher hand
    if pitcher_hand:
        kwargs["pitcher_hand"] = pitcher_hand

    # Platoon splits
    platoon_df = aux["platoon_df"]
    if not platoon_df.empty:
        bp = platoon_df[platoon_df["batter_id"] == batter_id]
        if not bp.empty:
            d: dict = {}
            for _, row in bp.iterrows():
                d[row["pitch_hand"]] = {
                    "k_rate": float(row["k_rate"]),
                    "bb_rate": float(row["bb_rate"]),
                    "pa": int(row["pa"]),
                }
            d["overall_k_rate"] = float(bp["overall_k_rate"].iloc[0])
            d["overall_bb_rate"] = float(bp["overall_bb_rate"].iloc[0])
            kwargs["batter_platoon_splits"] = d

    # Glicko
    p_glicko_df = aux["pitcher_glicko_df"]
    if not p_glicko_df.empty:
        pg = p_glicko_df[p_glicko_df["pitcher_id"] == pitcher_id]
        if not pg.empty:
            kwargs["pitcher_glicko_mu"] = float(pg["mu"].iloc[0])
    b_glicko_df = aux["batter_glicko_df"]
    if not b_glicko_df.empty:
        bg = b_glicko_df[b_glicko_df["batter_id"] == batter_id]
        if not bg.empty:
            kwargs["batter_glicko_mu"] = float(bg["mu"].iloc[0])

    # Pitcher GB%
    pgb_df = aux["pitcher_gb_df"]
    if not pgb_df.empty:
        pgb = pgb_df[pgb_df["pitcher_id"] == pitcher_id]
        if not pgb.empty:
            kwargs["pitcher_gb_pct"] = float(pgb["gb_pct"].iloc[0])

    # Batter GB/FB from projections
    if hitter_proj is not None and not hitter_proj.empty:
        hp = hitter_proj[hitter_proj["batter_id"] == batter_id]
        if not hp.empty:
            if "gb_rate" in hp.columns:
                v = hp["gb_rate"].iloc[0]
                if pd.notna(v):
                    kwargs["batter_gb_rate"] = float(v)
            if "fb_rate" in hp.columns:
                v = hp["fb_rate"].iloc[0]
                if pd.notna(v):
                    kwargs["batter_fb_rate"] = float(v)

    # League baselines and scales
    if aux["matchup_baselines"]:
        kwargs["league_platoon"] = aux["matchup_baselines"]
    if aux["scales"]:
        kwargs["matchup_scales"] = aux["scales"]

    return kwargs


def _matchup_advantage_html(
    advantage: dict,
) -> tuple[str, float, float, float]:
    """Return (HTML badge + short reason, k_lift, bb_lift, hr_lift).

    Uses the unified score_matchup_advantage result dict.
    """
    bd = advantage["breakdown"]
    k_lift = bd["k_lift"]
    bb_lift = bd["bb_lift"]
    hr_lift = bd["hr_lift"]
    adv = advantage["advantage"]
    reason = advantage["reason"]

    if adv == "neutral":
        html_out = (
            f'<span style="color:var(--tdd-slate); font-size:0.68rem; '
            f'margin-left:0.3rem;" title="{reason}">'
            f'Even</span>'
        )
        return html_out, k_lift, bb_lift, hr_lift

    if adv == "pitcher":
        color_var = "--tdd-ember"
        label = "Pitcher"
    else:
        color_var = "--tdd-sage"
        label = "Hitter"

    html_out = (
        f'<span style="color:var({color_var}); font-size:0.68rem; '
        f'font-weight:600; margin-left:0.3rem;" title="{reason}">'
        f'{label} - '
        f'<span style="font-weight:400; font-size:0.62rem;">{reason}</span>'
        f'</span>'
    )
    return html_out, k_lift, bb_lift, hr_lift


def _matchup_lift_badge_html(k_lift: float, bb_lift: float, hr_lift: float) -> str:
    """Compact matchup badge from precomputed lift values."""
    # Positive k_lift = pitcher advantage (more K), negative = hitter advantage
    net = k_lift - 0.5 * bb_lift - 0.5 * hr_lift
    if net > 0.03:
        color_var, label = "--tdd-ember", "Pitcher"
    elif net < -0.03:
        color_var, label = "--tdd-sage", "Hitter"
    else:
        return (
            f'<span style="color:var(--tdd-slate); font-size:0.68rem; '
            f'margin-left:0.3rem;">Even</span>'
        )
    return (
        f'<span style="color:var({color_var}); font-size:0.68rem; '
        f'font-weight:600; margin-left:0.3rem;">{label} '
        f'<span style="font-weight:400; font-size:0.62rem;">'
        f'K {k_lift:+.2f}</span></span>'
    )


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
    from components.scouting import build_matchup_scouting_bullets

    if pos_lookup is None:
        pos_lookup = {}
    if bf_priors is None:
        bf_priors = pd.DataFrame()
    if batter_sims_df is None:
        batter_sims_df = pd.DataFrame()

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
            p_hand = _lineup_pitcher_hand_html(p_proj, arsenal_df, pid)
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

                # Pitcher game sim — prefer precomputed samples
                _p_samples: dict[str, np.ndarray] | None = None
                if pitcher_sim_samples is not None:
                    _p_samples = pitcher_sim_samples.get(gpk, pid)

                if _p_samples is not None and "k" in _p_samples:
                    _render_player_sim(
                        _p_samples,
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

                # Headshot + diamond + matchup badge
                composite = stats.get("tdd_value_score")
                _top_parts = []
                if bid:
                    _top_parts.append(headshot_html(bid, size=32))
                if pd.notna(composite):
                    _top_parts.append(diamond_rating_html(0, size="sm", precomputed=composite))
                if abs(k_lift) > 0.01 or abs(bb_lift) > 0.01:
                    _top_parts.append(_matchup_lift_badge_html(k_lift, bb_lift, hr_lift))
                if _top_parts:
                    detail_parts.append(
                        f'<div style="display:flex; align-items:center; gap:0.5rem;">'
                        + "".join(_top_parts) + '</div>'
                    )

                # Scouting bullets
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

                # Grades
                grades_html = _hitter_grades_html(stats)
                if grades_html:
                    detail_parts.append(grades_html)

                if detail_parts:
                    st.markdown("".join(detail_parts), unsafe_allow_html=True)

                # Batter game sim — prefer precomputed samples
                _b_precomp: dict[str, np.ndarray] | None = None
                if bid and batter_sim_samples is not None:
                    _b_precomp = batter_sim_samples.get(gpk, bid)

                if _b_precomp is not None and "k" in _b_precomp:
                    _render_player_sim(
                        _b_precomp,
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
