"""Schedule page | date-based game browser with projections and live data."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
)
from utils.alerts import tdd_info
from services.data_loader import (
    load_todays_games, load_todays_lineups, load_todays_batter_sims,
    load_update_metadata,
    load_projections, load_counting, load_hitter_archetypes, load_pitcher_archetypes,
    load_roster, load_game_props, load_prop_attribution,
    load_hitters_daily_standouts, load_pitchers_daily_standouts,
    fetch_live_schedule, fetch_live_lineups, backfill_missing_lineups,
)
from utils.helpers import format_game_time
from utils.html import esc, esc_attr
from components.attribution import build_attribution_panel
from components.projection_table import (
    MODE_FINAL,
    MODE_LIVE,
    MODE_PROJECTION,
    game_block,
    lineup_tag,
    with_outcome_ranges,
)
from components.diamond_rating import diamond_rating_html
from components.headshot import headshot_html


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
    _cols = ["projected_k_rate", "pitcher_name", "projected_bb_rate",
             "projected_hr_per_bf", "pitch_hand"]
    _avail = [c for c in _cols if c in proj.columns]
    lookup = {}
    for pid, *vals in zip(proj["pitcher_id"].astype(int), *[proj[c] for c in _avail]):
        lookup[pid] = dict(zip(_avail, vals))
    return lookup


@st.cache_data(ttl=timedelta(minutes=5))
def _build_schedule_lookups() -> dict:
    """Build all lookup dicts for schedule cards (cached).

    Uses vectorized pandas operations instead of iterrows() for speed.
    """
    from lib.diamond_rating import score_to_diamonds

    _h_arch = load_hitter_archetypes()
    _p_arch = load_pitcher_archetypes()
    _h_proj = load_projections("hitter")
    _h_count = load_counting("hitter")

    # Archetype lookups (vectorized)
    h_arch_lookup: dict[int, str] = {}
    if not _h_arch.empty and "batter_id" in _h_arch.columns:
        h_arch_lookup = dict(zip(
            _h_arch["batter_id"].astype(int), _h_arch["archetype_name"],
        ))

    p_arch_lookup: dict[int, str] = {}
    if not _p_arch.empty and "pitcher_id" in _p_arch.columns:
        p_arch_lookup = dict(zip(
            _p_arch["pitcher_id"].astype(int), _p_arch["archetype_name"],
        ))

    from services.data_loader import load_standings, load_rankings
    standings = load_standings()

    _h_rankings = load_rankings("hitters")
    _p_rankings = load_rankings("pitchers")

    # Diamond lookup (vectorized)
    diamond_lookup: dict[int, float] = {}
    if not _h_rankings.empty and "tdd_value_score" in _h_rankings.columns:
        _valid = _h_rankings[_h_rankings["tdd_value_score"].notna()]
        diamond_lookup.update({
            int(bid): score_to_diamonds(score)
            for bid, score in zip(_valid["batter_id"], _valid["tdd_value_score"])
        })
    if not _p_rankings.empty and "tdd_value_score" in _p_rankings.columns:
        _valid = _p_rankings[_p_rankings["tdd_value_score"].notna()]
        diamond_lookup.update({
            int(pid): score_to_diamonds(score)
            for pid, score in zip(_valid["pitcher_id"], _valid["tdd_value_score"])
        })

    # Hitter stat lookup (vectorized)
    h_stat_lookup: dict[int, dict] = {}
    if not _h_proj.empty and "batter_id" in _h_proj.columns:
        for bid, k_rate, bb_rate, bat_hand in zip(
            _h_proj["batter_id"].astype(int),
            _h_proj.get("projected_k_rate", pd.Series(dtype=float)),
            _h_proj.get("projected_bb_rate", pd.Series(dtype=float)),
            _h_proj.get("batter_stand", pd.Series(dtype=str)),
        ):
            h_stat_lookup[bid] = {
                "k_rate": k_rate if pd.notna(k_rate) else None,
                "bb_rate": bb_rate if pd.notna(bb_rate) else None,
                "tdd_value_score": diamond_lookup.get(bid),
                "bat_hand": bat_hand if pd.notna(bat_hand) else None,
            }

    # Counting stats merge
    if not _h_count.empty and "batter_id" in _h_count.columns:
        _count_cols = {"total_hr_mean": "hr", "total_k_mean": "total_k",
                       "total_bb_mean": "total_bb"}
        _avail = [c for c in _count_cols if c in _h_count.columns]
        for bid_val, *vals in zip(
            _h_count["batter_id"].astype(int),
            *[_h_count[c] for c in _avail],
        ):
            counting = dict(zip([_count_cols[c] for c in _avail], vals))
            counting["total_hr"] = counting.get("hr")
            if bid_val in h_stat_lookup:
                h_stat_lookup[bid_val].update(counting)
            else:
                h_stat_lookup[bid_val] = {
                    "k_rate": None, "bb_rate": None,
                    "tdd_value_score": diamond_lookup.get(bid_val),
                    **counting,
                }

    # Inject scouting grades (vectorized)
    _h_grade_cols = ["grade_hit", "grade_power", "grade_speed", "grade_discipline", "grade_fielding"]
    if not _h_rankings.empty and "batter_id" in _h_rankings.columns:
        _avail_g = [c for c in _h_grade_cols if c in _h_rankings.columns]
        if _avail_g:
            for bid_val, *gvals in zip(
                _h_rankings["batter_id"].astype(int),
                *[_h_rankings[c] for c in _avail_g],
            ):
                grades = {c: int(v) for c, v in zip(_avail_g, gvals) if pd.notna(v)}
                if bid_val in h_stat_lookup:
                    h_stat_lookup[bid_val].update(grades)
                else:
                    h_stat_lookup[bid_val] = {
                        "k_rate": None, "bb_rate": None,
                        "tdd_value_score": diamond_lookup.get(bid_val),
                        **grades,
                    }

    # Pitcher grade lookup (vectorized)
    p_grade_lookup: dict[int, dict] = {}
    _p_grade_cols = ["grade_stuff", "grade_command", "grade_durability"]
    if not _p_rankings.empty and "pitcher_id" in _p_rankings.columns:
        _avail_pg = [c for c in _p_grade_cols if c in _p_rankings.columns]
        if _avail_pg:
            for pid_val, *gvals in zip(
                _p_rankings["pitcher_id"].astype(int),
                *[_p_rankings[c] for c in _avail_pg],
            ):
                grades = {c: int(v) for c, v in zip(_avail_pg, gvals) if pd.notna(v)}
                grades["tdd_value_score"] = diamond_lookup.get(pid_val)
                p_grade_lookup[pid_val] = grades

    # Grade confidence intervals (vectorized)
    from services.data_loader import load_hitter_grade_ci, load_pitcher_grade_ci
    _h_ci = load_hitter_grade_ci()
    _p_ci = load_pitcher_grade_ci()
    _h_ci_cols = [
        "grade_hit_lo", "grade_hit_hi", "grade_power_lo", "grade_power_hi",
        "grade_speed_lo", "grade_speed_hi", "grade_discipline_lo", "grade_discipline_hi",
        "grade_fielding_lo", "grade_fielding_hi",
    ]
    if not _h_ci.empty and "player_id" in _h_ci.columns:
        _avail_ci = [c for c in _h_ci_cols if c in _h_ci.columns]
        if _avail_ci:
            for bid_val, *cvals in zip(
                _h_ci["player_id"].astype(int),
                *[_h_ci[c] for c in _avail_ci],
            ):
                ci_vals = {c: int(v) for c, v in zip(_avail_ci, cvals) if pd.notna(v)}
                if bid_val in h_stat_lookup:
                    h_stat_lookup[bid_val].update(ci_vals)
                else:
                    h_stat_lookup[bid_val] = ci_vals

    _p_ci_cols = [
        "grade_stuff_lo", "grade_stuff_hi", "grade_command_lo", "grade_command_hi",
        "grade_durability_lo", "grade_durability_hi",
    ]
    if not _p_ci.empty and "player_id" in _p_ci.columns:
        _avail_pci = [c for c in _p_ci_cols if c in _p_ci.columns]
        if _avail_pci:
            for pid_val, *cvals in zip(
                _p_ci["player_id"].astype(int),
                *[_p_ci[c] for c in _avail_pci],
            ):
                ci_vals = {c: int(v) for c, v in zip(_avail_pci, cvals) if pd.notna(v)}
                if pid_val in p_grade_lookup:
                    p_grade_lookup[pid_val].update(ci_vals)
                else:
                    p_grade_lookup[pid_val] = ci_vals

    # Position lookup (vectorized)
    _roster = load_roster()
    pos_lookup: dict[int, str] = {}
    if not _roster.empty and "primary_position" in _roster.columns:
        pos_lookup = dict(zip(
            _roster["player_id"].astype(int), _roster["primary_position"],
        ))
    _prospects = load_rankings("prospect")
    if not _prospects.empty and "primary_position" in _prospects.columns:
        _id_col = "player_id" if "player_id" in _prospects.columns else "batter_id"
        for pid_val, pos_val in zip(
            _prospects[_id_col].astype(int), _prospects["primary_position"],
        ):
            if pid_val and pid_val not in pos_lookup:
                pos_lookup[pid_val] = pos_val

    # Projection lookup with injected grades/diamonds
    proj_lookup = _build_projection_lookup()
    for pid, pinfo in proj_lookup.items():
        pinfo["tdd_value_score"] = diamond_lookup.get(pid)
        if pid in p_grade_lookup:
            pinfo.update(p_grade_lookup[pid])

    # Name lookup (vectorized)
    name_lookup: dict[int, str] = {}
    if not _h_proj.empty and "batter_name" in _h_proj.columns:
        name_lookup.update(dict(zip(
            _h_proj["batter_id"].astype(int), _h_proj["batter_name"],
        )))
    _pp = load_projections("pitcher")
    if not _pp.empty and "pitcher_name" in _pp.columns:
        name_lookup.update(dict(zip(
            _pp["pitcher_id"].astype(int), _pp["pitcher_name"],
        )))

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
# Projected stat lines
# ---------------------------------------------------------------------------

_STAT_TO_ACTUAL = {
    "K": "actual_K", "H": "actual_H", "HR": "actual_HR",
    "BB": "actual_BB", "TB": "actual_TB", "Outs": "actual_Outs",
    "R": "actual_R", "RBI": "actual_RBI",
}


def _overlay_live_actuals(game_df: pd.DataFrame, game_live: pd.DataFrame) -> pd.DataFrame:
    """Fill in results from the live boxscore for a game in progress."""
    if game_live.empty:
        return game_df
    live_lookup = {int(row["player_id"]): row for _, row in game_live.iterrows()}
    for idx, row in game_df.iterrows():
        live = live_lookup.get(int(row["player_id"]))
        if live is None:
            continue
        stat = row["stat"]
        if stat == "HRR":
            parts = [live.get(c) for c in ("actual_H", "actual_R", "actual_RBI")]
            if any(pd.notna(p) for p in parts):
                game_df.at[idx, "actual"] = sum(float(p) for p in parts if pd.notna(p))
        else:
            column = _STAT_TO_ACTUAL.get(stat)
            if column and column in live.index and pd.notna(live[column]):
                game_df.at[idx, "actual"] = float(live[column])
        game_df.at[idx, "game_status"] = live.get("game_status", row.get("game_status"))
    return game_df


def _render_props_section(
    gpk: int,
    props_df: pd.DataFrame,
    lineups_df: pd.DataFrame | None = None,
    live_stats_df: pd.DataFrame | None = None,
) -> None:
    """Render projected stat lines for a single game.

    When confirmed lineups are available, only shows players who are in
    the starting lineup (pitchers always included).  When lineups have
    not been released yet, shows all projected players with a warning
    banner.
    """
    game_df = (
        props_df[props_df["game_pk"] == gpk].copy()
        if not props_df.empty
        else pd.DataFrame()
    )
    if game_df.empty:
        st.markdown(
            '<div class="tdd-meta">No projection data available for this game.</div>',
            unsafe_allow_html=True,
        )
        return

    game_lu = (
        lineups_df[lineups_df["game_pk"] == gpk]
        if lineups_df is not None and not lineups_df.empty
        else pd.DataFrame()
    )
    lineup_confirmed = not game_lu.empty

    if lineup_confirmed:
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

    for col in ("actual", "game_status"):
        if col not in game_df.columns:
            game_df[col] = None

    game_live = (
        live_stats_df[live_stats_df["game_pk"] == gpk]
        if live_stats_df is not None and not live_stats_df.empty
        else pd.DataFrame()
    )
    game_df = _overlay_live_actuals(game_df, game_live)

    statuses = {str(s).lower() for s in game_df["game_status"].dropna()}
    if any("final" in s or "game over" in s for s in statuses):
        mode = MODE_FINAL
    elif game_df["actual"].notna().any():
        mode = MODE_LIVE
    else:
        mode = MODE_PROJECTION

    st.markdown(
        '<div class="tdd-props-header">Projected Stat Lines</div>',
        unsafe_allow_html=True,
    )
    teams = [t for t in game_df["team"].dropna().unique()]
    tag = lineup_tag(lineup_confirmed) if mode == MODE_PROJECTION else ""
    st.markdown(
        game_block(with_outcome_ranges(game_df), [(t, tag) for t in teams], mode),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tdd-meta">Each cell shows the projected average with the range '
        'covering the middle 80% of simulated outcomes.</div>',
        unsafe_allow_html=True,
    )

    # "Why this number" -- K projection attribution for the game's starters
    _attr = load_prop_attribution()
    if not _attr.empty:
        _pit = game_df[game_df["player_type"] == "pitcher"]
        _name_by_pid = dict(zip(_pit["player_id"].astype(int), _pit["player_name"]))
        _pids = list(dict.fromkeys(_pit["player_id"].astype(int)))
        _panels = []
        for _pid in _pids:
            _ar = _attr[
                (_attr["game_pk"] == gpk)
                & (_attr["player_id"] == _pid)
                & (_attr["player_type"] == "pitcher")
                & (_attr["stat"] == "K")
            ]
            if not _ar.empty:
                _panels.append(build_attribution_panel(
                    _ar.iloc[0], name=_name_by_pid.get(_pid),
                ))
        if _panels:
            st.markdown(
                '<div class="tdd-props-header">Why This Number</div>',
                unsafe_allow_html=True,
            )
            st.markdown("".join(_panels), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Game Drill-Down (Phase 2: Game Center)
# ---------------------------------------------------------------------------

@st.fragment
def _render_game_drilldown(
    game: pd.Series,
    lineups: pd.DataFrame,
    gpk: int,
    game_props: pd.DataFrame | None = None,
    live_stats: pd.DataFrame | None = None,
    batter_sims_df: pd.DataFrame | None = None,
) -> None:
    """Game drill-down: props and game simulator."""
    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty else pd.DataFrame()

    _batter_sims = batter_sims_df if batter_sims_df is not None else pd.DataFrame()

    # Surface a notice when the posted lineup has drifted from the sim inputs
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

    _render_props_section(
        gpk,
        game_props if game_props is not None else pd.DataFrame(),
        lineups_df=game_lu,
        live_stats_df=live_stats,
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


def _schedule_masthead_html(
    selected_date, n_games: int, n_hitters: int, n_starters: int = 0,
) -> str:
    """Redesigned masthead using .tdd-sched-masthead classes."""
    d = selected_date
    date_str = d.strftime("%a, %b %d")
    year_str = d.strftime("%Y")
    full_date = d.strftime("%B %d").replace(" 0", " ")
    stats = [
        (str(n_games), "Games"),
        (str(n_hitters), "Hitters"),
        (str(n_starters), "Starters"),
    ]
    stats_html = "".join(
        f'<div class="stat"><div class="v">{esc(v)}</div>'
        f'<div class="l">{esc(l)}</div></div>'
        for v, l in stats
    )
    return (
        '<div class="tdd-sched-masthead">'
        '<div class="title-block">'
        f'<div class="eyebrow">MLB Schedule</div>'
        f'<h1>Games <span class="date">{esc(full_date)}</span></h1>'
        f'<div class="sub">{esc(year_str)} Regular Season</div>'
        '</div>'
        f'<div class="slate-stats">{stats_html}</div>'
        '</div>'
    )


def _yesterday_performers_html(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> str:
    """Render yesterday's top performers widget above the game list."""
    if hitters.empty and pitchers.empty:
        return ""

    items: list[dict] = []

    if not hitters.empty and "daily_standout_score" in hitters.columns:
        top_h = hitters.sort_values("daily_standout_score", ascending=False).head(3)
        for _, row in top_h.iterrows():
            name = str(row.get("batter_name", ""))
            if not name:
                continue
            team = str(row.get("team_abbr", "")) or ""
            h_val = int(row["hits"]) if pd.notna(row.get("hits")) else 0
            ab_val = int(row["ab"]) if pd.notna(row.get("ab")) else 0
            hr_val = int(row["hr"]) if pd.notna(row.get("hr")) else 0
            stat = f"{h_val}-{ab_val}"
            if hr_val > 0:
                stat += f", {hr_val}HR"
            items.append({"badge": "Hitter", "name": name, "team": team, "stat": stat})

    if not pitchers.empty and "daily_standout_score" in pitchers.columns:
        top_p = pitchers.sort_values("daily_standout_score", ascending=False).head(2)
        for _, row in top_p.iterrows():
            name = str(row.get("pitcher_name", ""))
            if not name:
                continue
            team = str(row.get("team_abbr", "")) or ""
            ip_val = float(row["ip"]) if pd.notna(row.get("ip")) else 0.0
            k_val = int(row["k"]) if pd.notna(row.get("k")) else 0
            bb_val = int(row["bb"]) if pd.notna(row.get("bb")) else 0
            h_val = int(row["hits"]) if pd.notna(row.get("hits")) else 0
            stat = f"{ip_val:.1f} IP, {k_val}K, {bb_val}BB, {h_val}H"
            items.append({"badge": "Pitcher", "name": name, "team": team, "stat": stat})

    if not items:
        return ""

    header = (
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'padding: 0.4rem 0.25rem 0.3rem; border-top: 1px solid var(--tdd-dark-border);">'
        '<span style="font-size:0.65rem; letter-spacing:1.2px; text-transform:uppercase; '
        'font-weight:700; color:var(--tdd-slate); font-family:var(--tdd-font-heading);">'
        "Yesterday's Top Performers</span>"
        '<a href="?page=the_diamond_daily" target="_self" '
        'style="font-size:0.7rem; color:var(--tdd-gold); text-decoration:none; '
        'font-weight:600; letter-spacing:0.3px;">'
        'View Diamond Daily &rsaquo;'
        '</a>'
        '</div>'
    )

    tiles = ""
    for item in items:
        tiles += (
            '<div class="e">'
            f'<div class="tier">{esc(item["badge"])}</div>'
            f'<div class="nm">{esc(item["name"])}</div>'
            f'<div class="line">{esc(item["team"])}</div>'
            f'<div class="stat-v">{esc(item["stat"])}</div>'
            '</div>'
        )

    return f'{header}<div class="tdd-top-performers" style="border-top:none;">{tiles}</div>'


def _game_row_html(
    game: pd.Series,
    gpk: int,
    pitcher_line_lookup: dict[tuple[int, int], str],
    game_proj: dict[int, dict],
    game_highlights: dict[int, dict],
    is_open: bool,
) -> str:
    """Render a single game as a .tdd-sg 5-column grid row."""
    away = game.get("away_abbr", "?")
    home = game.get("home_abbr", "?")
    time_str = format_game_time(
        game.get("game_datetime_utc"), fallback=game.get("game_time", ""),
    )
    status_raw = game.get("status", "")
    away_sp = game.get("away_pitcher_name", "") or "TBD"
    home_sp = game.get("home_pitcher_name", "") or "TBD"
    away_pid_raw = game.get("away_pitcher_id")
    home_pid_raw = game.get("home_pitcher_id")
    away_pid = int(away_pid_raw) if pd.notna(away_pid_raw) else None
    home_pid = int(home_pid_raw) if pd.notna(home_pid_raw) else None

    # Status
    if "Progress" in status_raw or "Live" in status_raw:
        status_html = '<div class="status live"><span class="dot"></span> LIVE</div>'
    elif "Final" in status_raw:
        status_html = '<div class="status final">FINAL</div>'
    else:
        status_html = f'<div class="status scheduled">{esc(time_str)}</div>'

    # Pitcher hand hints
    away_hand = ""
    home_hand = ""

    # Col 1: Time
    col_time = (
        '<div class="col-time">'
        f'<div class="time">{esc(time_str)}</div>'
        f'{status_html}'
        '</div>'
    )

    # Col 2: Teams + pitchers
    col_teams = (
        '<div class="col-teams">'
        '<div class="tm">'
        f'<span class="abbr" data-team="{esc_attr(away)}">{esc(away)}</span>'
        f'<span class="pname">{esc(away_sp)}</span>'
        '</div>'
        '<div class="tm">'
        f'<span class="abbr" data-team="{esc_attr(home)}">{esc(home)}</span>'
        f'<span class="pname">{esc(home_sp)}</span>'
        '</div>'
        '</div>'
    )

    # Col 3: Projections (R, H, HR, K)
    proj = game_proj.get(gpk, {})
    proj_r = f"{proj.get('R', 0):.1f}" if proj.get("R") else ""
    proj_h = f"{proj.get('H', 0):.1f}" if proj.get("H") else ""
    proj_hr = f"{proj.get('HR', 0):.1f}" if proj.get("HR") else ""
    proj_k = f"{proj.get('K', 0):.1f}" if proj.get("K") else ""

    col_proj = (
        '<div class="col-proj">'
        '<div class="lbl">Projected</div>'
        '<div class="row">'
        f'<div class="m"><div class="v">{esc(proj_r)}</div><div class="k">R</div></div>'
        f'<div class="m"><div class="v">{esc(proj_h)}</div><div class="k">H</div></div>'
        f'<div class="m"><div class="v">{esc(proj_hr)}</div><div class="k">HR</div></div>'
        f'<div class="m dim"><div class="v">{esc(proj_k)}</div><div class="k">K</div></div>'
        '</div>'
        '</div>'
    )

    # Col 4: top projected batter
    highlight = game_highlights.get(gpk, {})
    if highlight:
        top_html = (
            f'<div class="nm">{esc(highlight["name"])}</div>'
            f'<div class="count"><b>{highlight["value"]:.1f}</b> {esc(highlight["stat"])} projected</div>'
        )
    else:
        top_html = '<div class="count zero">No projections</div>'

    col_top = (
        '<div class="col-top">'
        f'{top_html}'
        '</div>'
    )

    # Col 5: CTA
    expanded_cls = " expanded" if is_open else ""
    col_cta = (
        '<div class="col-cta">'
        'Details <span class="arrow">&rsaquo;</span>'
        '</div>'
    )

    return (
        f'<div class="tdd-sg{expanded_cls}">'
        f'{col_time}{col_teams}{col_proj}{col_top}{col_cta}'
        '</div>'
    )


@st.cache_data(ttl=300)
def _build_game_projections(sims: pd.DataFrame) -> dict[int, dict]:
    """Aggregate game-level R/H/HR/K projections from game_props.

    Uses batter rows for H/HR/R/K (summed across lineup) and
    pitcher rows for Outs. Avoids double-counting.
    """
    result: dict[int, dict] = {}
    if sims.empty or "game_pk" not in sims.columns:
        return result

    # Filter to today's games only
    today_gpks = sims["game_pk"].unique()

    # Batter stats: sum H, R, K across both lineups
    batters = sims[sims["player_type"] == "batter"] if "player_type" in sims.columns else sims
    for gpk, grp in batters.groupby("game_pk"):
        gpk = int(gpk)
        totals: dict[str, float] = {}
        for _, row in grp.iterrows():
            stat = row.get("stat", "")
            exp = row.get("expected")
            if pd.notna(exp) and stat in ("H", "R", "K", "HRR"):
                totals[stat] = totals.get(stat, 0) + float(exp)
        result[gpk] = totals

    # Pitcher stats: sum HR allowed across both starters
    pitchers = sims[sims["player_type"] == "pitcher"] if "player_type" in sims.columns else pd.DataFrame()
    if not pitchers.empty:
        for gpk, grp in pitchers.groupby("game_pk"):
            gpk = int(gpk)
            for _, row in grp.iterrows():
                stat = row.get("stat", "")
                exp = row.get("expected")
                if pd.notna(exp) and stat == "HR":
                    if gpk not in result:
                        result[gpk] = {}
                    result[gpk]["HR"] = result[gpk].get("HR", 0) + float(exp)

    return result


@st.cache_data(ttl=300)
def _build_game_highlights(sims: pd.DataFrame) -> dict[int, dict]:
    """The highest projected batter in each game, by total bases."""
    result: dict[int, dict] = {}
    if sims.empty or "game_pk" not in sims.columns:
        return result

    batters = sims[(sims.get("player_type") == "batter") & (sims.get("stat") == "TB")]
    for gpk, grp in batters.groupby("game_pk"):
        gpk = int(gpk)
        grp = grp[grp["expected"].notna()]
        if grp.empty:
            result[gpk] = {}
            continue
        best = grp.loc[grp["expected"].idxmax()]
        result[gpk] = {
            "name": str(best.get("player_name", "")),
            "value": float(best["expected"]),
            "stat": "TB",
        }
    return result


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
    """List-view layout with 5-column game rows, top performers rail, and drilldown."""
    n_games = len(schedule)
    # game_props and the batter sims both cover today and tomorrow, so the
    # slate counts have to be scoped to the games actually on screen.
    slate_pks = set(schedule["game_pk"]) if "game_pk" in schedule.columns else set()
    n_hitters = (
        batter_sims[batter_sims["game_pk"].isin(slate_pks)]["batter_id"].nunique()
        if not batter_sims.empty and slate_pks
        else 0
    )

    # Pre-compute game projections and per-game highlights from sims
    game_proj = _build_game_projections(sims)
    game_highlights = _build_game_highlights(sims)
    n_starters = (
        sims[(sims["player_type"] == "pitcher") & sims["game_pk"].isin(slate_pks)]["player_id"].nunique()
        if not sims.empty and "player_type" in sims.columns and slate_pks
        else 0
    )

    # Lookups
    _lookups = _build_schedule_lookups()

    # Masthead
    st.markdown(
        _schedule_masthead_html(selected_date, n_games, n_hitters, n_starters),
        unsafe_allow_html=True,
    )

    # Yesterday's top performers widget
    _yp_html = _yesterday_performers_html(
        load_hitters_daily_standouts(),
        load_pitchers_daily_standouts(),
    )
    if _yp_html:
        st.markdown(_yp_html, unsafe_allow_html=True)

    if schedule.empty:
        tdd_info("No games scheduled for this date.")
        return

    # Sort chronologically
    if "game_datetime_utc" in schedule.columns:
        schedule = schedule.sort_values("game_datetime_utc", na_position="last")

    # Build pitcher projected line lookup
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
            "batter_sims_df": load_todays_batter_sims(),
            "game_props": _game_props,
        }
        return _drilldown_data

    # Button styles for drilldown toggle
    st.markdown(
        '<style>'
        '.sched-toggle [data-testid="stButton"] button {'
        '  background: transparent !important;'
        '  border: 1px solid var(--tdd-dark-border) !important;'
        '  border-radius: 0 !important;'
        '  margin-top: -0.5rem !important;'
        '  padding: 0.3rem 1rem !important;'
        '  font-size: 0.7rem !important;'
        '  color: var(--tdd-slate) !important;'
        '  font-weight: 500 !important;'
        '  letter-spacing: 0.3px !important;'
        '  transition: border-color 0.15s, color 0.15s !important;'
        '}'
        '.sched-toggle [data-testid="stButton"] button:hover {'
        '  border-color: var(--tdd-gold) !important;'
        '  color: var(--tdd-gold) !important;'
        '  background: rgba(255,255,255,0.03) !important;'
        '}'
        '</style>',
        unsafe_allow_html=True,
    )

    # Track expanded games
    if "sched_a_open" not in st.session_state:
        st.session_state["sched_a_open"] = set()

    # Render game list
    st.markdown('<div class="tdd-sg-list">', unsafe_allow_html=True)

    for _, game in schedule.iterrows():
        gpk = int(game["game_pk"])
        is_open = gpk in st.session_state["sched_a_open"]

        # Game row
        st.markdown(
            _game_row_html(game, gpk, _pitcher_line_lookup, game_proj, game_highlights, is_open),
            unsafe_allow_html=True,
        )

        # Toggle button
        with st.container():
            st.markdown('<div class="sched-toggle">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)

        # Drilldown
        if is_open:
            st.markdown('<div class="tdd-sg-drill">', unsafe_allow_html=True)
            st.markdown(
                f'<a href="?page=game_analysis&game_pk={gpk}" target="_self" '
                f'style="color:var(--tdd-gold); font-size:0.75rem; '
                f'text-decoration:none; font-weight:600; display:inline-block; '
                f'margin-bottom:0.5rem;">Full Analysis &#8594;</a>',
                unsafe_allow_html=True,
            )
            _dd = _get_dd()
            _render_game_drilldown(
                game, lineups, gpk,
                game_props=_dd["game_props"],
                batter_sims_df=_dd["batter_sims_df"],
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


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
            live_sched = fetch_live_schedule(today.isoformat(), include_weather=False)
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
