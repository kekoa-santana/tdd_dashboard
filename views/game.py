"""Game Analysis page -- full-screen deep dive for a single game."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DASHBOARD_DIR
from services.data_loader import (
    load_todays_games, load_todays_lineups, load_todays_batter_sims,
    load_update_metadata, load_pitcher_arsenal, load_hitter_vulnerability,
    load_hitter_strength, load_projections, load_hitter_archetypes,
    load_pitcher_archetypes, load_bf_priors,
    load_pitcher_game_sim_samples, load_batter_game_sim_samples,
    fetch_live_schedule, fetch_live_lineups, backfill_missing_lineups,
)
from components.team_logo import team_logo_html
from components.headshot import headshot_html
from components.grades import pitcher_grades_html, hitter_grades_html
from components.sim_chart import render_player_sim, PITCHER_STAT_META, BATTER_STAT_META
from components.scouting import render_scouting_html, compute_matchup_xwoba_edge
from lib.fantasy_report import load_report_data, get_pitcher_scouting, ReportData
from utils.helpers import format_ip, format_game_time


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _get_scouting_report_data() -> ReportData:
    return load_report_data(DASHBOARD_DIR)


@st.cache_data(ttl=300)
def _build_lookups() -> dict:
    """Build cached lookup dicts for archetypes, stats, and projections."""
    h_arch = {}
    ha = load_hitter_archetypes()
    if not ha.empty and "batter_id" in ha.columns and "archetype_name" in ha.columns:
        h_arch = dict(zip(ha["batter_id"].astype(int), ha["archetype_name"]))

    p_arch = {}
    pa = load_pitcher_archetypes()
    if not pa.empty and "pitcher_id" in pa.columns and "archetype_name" in pa.columns:
        p_arch = dict(zip(pa["pitcher_id"].astype(int), pa["archetype_name"]))

    h_stat = {}
    hp = load_projections("hitter")
    if not hp.empty and "batter_id" in hp.columns:
        for _, r in hp.iterrows():
            h_stat[int(r["batter_id"])] = r.to_dict()

    proj = {}
    pp = load_projections("pitcher")
    if not pp.empty and "pitcher_id" in pp.columns:
        for _, r in pp.iterrows():
            proj[int(r["pitcher_id"])] = r.to_dict()

    try:
        from services.data_loader import load_standings
        standings = load_standings() or {}
    except (ImportError, Exception):
        standings = {}

    return {
        "h_arch": h_arch, "p_arch": p_arch,
        "h_stat": h_stat, "proj": proj, "standings": standings,
    }


def _find_game(game_pk: int, schedule: pd.DataFrame) -> pd.Series | None:
    """Find a game row by game_pk."""
    if schedule.empty or "game_pk" not in schedule.columns:
        return None
    match = schedule[schedule["game_pk"] == game_pk]
    return match.iloc[0] if not match.empty else None


# ---------------------------------------------------------------------------
# Game header
# ---------------------------------------------------------------------------

def _render_game_header(
    game: pd.Series,
    lookups: dict,
) -> None:
    """Team logos, pitcher info, game context."""
    away_abbr = game.get("away_abbr", "?")
    home_abbr = game.get("home_abbr", "?")
    away_tid = game.get("away_team_id")
    home_tid = game.get("home_team_id")
    standings = lookups["standings"]
    proj = lookups["proj"]
    p_arch = lookups["p_arch"]

    away_logo = team_logo_html(int(away_tid), size=100) if pd.notna(away_tid) else ""
    home_logo = team_logo_html(int(home_tid), size=100) if pd.notna(home_tid) else ""

    away_record = f'{standings[away_abbr][0]}-{standings[away_abbr][1]}' if away_abbr in standings else ""
    home_record = f'{standings[home_abbr][0]}-{standings[home_abbr][1]}' if home_abbr in standings else ""

    game_time = format_game_time(
        game.get("game_datetime_utc"), fallback=game.get("game_time", ""),
    )
    game_date = game.get("game_date", "")

    # Context line
    ctx_parts = []
    venue = game.get("venue_name", "")
    if venue:
        ctx_parts.append(venue)
    hp_ump = game.get("hp_umpire_name", "")
    if hp_ump:
        ctx_parts.append(f"HP: {hp_ump}")
    wx_temp = game.get("weather_temp", "")
    wx_cond = game.get("weather_condition", "")
    if wx_temp:
        wx_str = f"{wx_temp} F"
        if wx_cond:
            wx_str += f", {wx_cond}"
        ctx_parts.append(wx_str)

    # Team header
    st.markdown(
        f'<div style="text-align:center; margin-bottom:1rem;">'
        f'<div class="tdd-meta">{game_date} · {game_time}</div>'
        f'<div style="display:flex; justify-content:center; align-items:center; gap:1.5rem; margin:0.8rem 0;">'
        f'{away_logo}'
        f'<div style="text-align:center;">'
        f'<span class="tdd-team-abbr" data-team="{away_abbr}" style="font-size:1.8rem;">{away_abbr}</span>'
        f'<div class="tdd-meta">{away_record}</div>'
        f'</div>'
        f'<span style="color:var(--tdd-slate); font-size:1.5rem; margin:0 0.5rem;">@</span>'
        f'<div style="text-align:center;">'
        f'<span class="tdd-team-abbr" data-team="{home_abbr}" style="font-size:1.8rem;">{home_abbr}</span>'
        f'<div class="tdd-meta">{home_record}</div>'
        f'</div>'
        f'{home_logo}'
        f'</div>'
        f'{"<div class=" + chr(34) + "tdd-meta" + chr(34) + ">" + " · ".join(ctx_parts) + "</div>" if ctx_parts else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )



# ---------------------------------------------------------------------------
# Pitcher duel
# ---------------------------------------------------------------------------

def _render_pitcher_duel(
    game: pd.Series,
    lookups: dict,
    pitcher_sim_samples,
) -> None:
    """Side-by-side starting pitcher comparison."""
    proj = lookups["proj"]
    p_arch = lookups["p_arch"]
    gpk = game["game_pk"]

    col_away, col_home = st.columns(2)

    for side, col in [("away", col_away), ("home", col_home)]:
        with col:
            pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
            pid_raw = game.get(f"{side}_pitcher_id")
            pid = int(pid_raw) if pd.notna(pid_raw) else None
            side_abbr = game.get(f"{side}_abbr", "?")
            opp_abbr = game.get(f"{'home' if side == 'away' else 'away'}_abbr", "?")

            # Headshot + name
            hs = headshot_html(pid, size=80) if pid else ""
            arch = p_arch.get(pid, "") if pid else ""
            arch_tag = f'<span class="tdd-meta"> · {arch}</span>' if arch else ""

            st.markdown(
                f'<div style="text-align:center; margin-bottom:0.5rem;">'
                f'{hs}'
                f'<div class="tdd-player-name" style="font-size:1rem;">{pitcher_name}</div>'
                f'<div class="tdd-meta">{side_abbr}{arch_tag}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Grades
            p_proj = proj.get(pid, {}) if pid else {}
            grades = pitcher_grades_html(p_proj)
            if grades:
                st.markdown(
                    f'<div style="text-align:center; margin-bottom:0.5rem;">{grades}</div>',
                    unsafe_allow_html=True,
                )

            # Key projections
            parts = []
            if p_proj.get("projected_k_rate"):
                parts.append(f'K% {p_proj["projected_k_rate"]*100:.1f}%')
            if p_proj.get("projected_bb_rate"):
                parts.append(f'BB% {p_proj["projected_bb_rate"]*100:.1f}%')
            if parts:
                st.markdown(
                    f'<div style="text-align:center;" class="tdd-meta">'
                    f'{" · ".join(parts)}</div>',
                    unsafe_allow_html=True,
                )

            # Sim distribution
            if pid and pitcher_sim_samples is not None:
                samples = pitcher_sim_samples.get(gpk, pid)
                if samples is not None and "k" in samples:
                    render_player_sim(
                        samples, PITCHER_STAT_META,
                        ["K", "BB", "H", "HR", "Outs"],
                        f"gp_p_{side}_{gpk}",
                    )

            # Scouting report
            if pid:
                scouting_data = _get_scouting_report_data()
                report = get_pitcher_scouting(
                    pid, pitcher_name, side_abbr, opp_abbr, scouting_data,
                )
                render_scouting_html(report)


# ---------------------------------------------------------------------------
# Key matchups spotlight
# ---------------------------------------------------------------------------

def _render_key_matchups(
    game: pd.Series,
    batter_sims: pd.DataFrame,
    lookups: dict,
) -> None:
    """Spotlight the most lopsided pitcher-batter matchups."""
    gpk = game["game_pk"]
    if batter_sims.empty or "game_pk" not in batter_sims.columns:
        return

    game_bs = batter_sims[batter_sims["game_pk"] == gpk].copy()
    if game_bs.empty or "matchup_k_lift" not in game_bs.columns:
        return

    game_bs["abs_lift"] = game_bs["matchup_k_lift"].abs()
    top = game_bs.nlargest(4, "abs_lift")
    if top.empty:
        return

    h_stat = lookups["h_stat"]
    p_arch = lookups["p_arch"]

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    str_df = load_hitter_strength(career=True)

    st.markdown(
        '<div class="section-subheader" style="margin-top:1.5rem;">Key Matchups</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(top), 4))
    for i, (_, row) in enumerate(top.iterrows()):
        bid = int(row["batter_id"])
        batter_name = row.get("batter_name", str(bid))
        opp_starter = int(row["opp_starter_id"]) if pd.notna(row.get("opp_starter_id")) else None

        # Compute advantage via odds-ratio xwOBA
        badge = ""
        if opp_starter and not arsenal_df.empty and not vuln_df.empty:
            _p_ars = arsenal_df[arsenal_df["pitcher_id"] == opp_starter]
            _h_vul = vuln_df[vuln_df["batter_id"] == bid]
            _h_str = str_df[str_df["batter_id"] == bid] if not str_df.empty else pd.DataFrame()
            _ph = str(_p_ars["pitch_hand"].iloc[0]) if not _p_ars.empty and "pitch_hand" in _p_ars.columns else None
            _bh = str(_h_vul["batter_stand"].iloc[0]) if not _h_vul.empty and "batter_stand" in _h_vul.columns else None
            if not _p_ars.empty and not _h_vul.empty:
                _edge_result = compute_matchup_xwoba_edge(
                    _p_ars, _h_vul, _h_str,
                    pitcher_hand=_ph, batter_hand=_bh,
                )
                _adv_label = _edge_result["advantage"]
                _xw = _edge_result["matchup_xwoba"]
                if _adv_label == "pitcher":
                    badge = (
                        f'<span style="color:var(--tdd-ember); font-size:0.68rem; '
                        f'font-weight:600;">Pitcher Edge '
                        f'<span style="font-weight:400; font-size:0.6rem;">'
                        f'.{int(_xw*1000):03d}</span></span>'
                    )
                elif _adv_label == "hitter":
                    badge = (
                        f'<span style="color:var(--tdd-sage); font-size:0.68rem; '
                        f'font-weight:600;">Hitter Edge '
                        f'<span style="font-weight:400; font-size:0.6rem;">'
                        f'.{int(_xw*1000):03d}</span></span>'
                    )
                else:
                    badge = (
                        f'<span style="color:var(--tdd-slate); font-size:0.68rem;">'
                        f'Even <span style="font-size:0.6rem;">'
                        f'.{int(_xw*1000):03d}</span></span>'
                    )

        hs = headshot_html(bid, size=50)
        team = row.get("team_abbr", "")

        with cols[i % len(cols)]:
            st.markdown(
                f'<div style="text-align:center; padding:0.5rem; '
                f'border:1px solid var(--tdd-slate-20); border-radius:8px; '
                f'margin-bottom:0.5rem;">'
                f'{hs}'
                f'<div class="tdd-player-name" style="font-size:0.85rem;">{batter_name}</div>'
                f'<div class="tdd-meta">{team}</div>'
                f'{badge}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Lineup matchups (reuses schedule.py machinery)
# ---------------------------------------------------------------------------

def _render_lineup_matchups(
    game: pd.Series,
    lineups: pd.DataFrame,
    lookups: dict,
    batter_sims: pd.DataFrame,
    pitcher_sim_samples,
    batter_sim_samples,
) -> None:
    """Full lineup matchup view, reusing schedule.py's side-by-side renderer."""
    from views.schedule import (
        _render_matchup_tab_sidebyside,
        _detect_lineup_changes,
    )

    gpk = game["game_pk"]
    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty else pd.DataFrame()

    h_arch = lookups["h_arch"]
    p_arch = lookups["p_arch"]
    h_stat = lookups["h_stat"]
    proj = lookups["proj"]

    pos_lookup = {}
    if not lineups.empty and "batter_id" in lineups.columns:
        for _, r in game_lu.iterrows():
            bid = int(r.get("batter_id", 0))
            pos = r.get("game_position", "")
            if bid and pos:
                pos_lookup[bid] = pos

    sides = []
    for side, opp_side_label in [("away", "home"), ("home", "away")]:
        pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
        pid_raw = game.get(f"{side}_pitcher_id")
        pid = int(pid_raw) if pd.notna(pid_raw) else None
        side_team_id = game.get(f"{side}_team_id")
        opp_team_id = game.get(f"{opp_side_label}_team_id")
        opp_abbr = game.get(f"{opp_side_label}_abbr", "?")
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
            "pitcher_arch": p_arch.get(pid) if pid else None,
            "pitcher_proj": proj.get(pid, {}) if pid else {},
            "opp_lineup": opp_lu,
            "own_lineup": own_lu,
        })

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    str_df = load_hitter_strength(career=True)
    bf_priors = load_bf_priors()

    changes = _detect_lineup_changes(game_lu, batter_sims, gpk)
    if changes["changed"]:
        n_new = len(changes["new_batters"])
        n_miss = len(changes["missing_batters"])
        parts = []
        if n_new:
            parts.append(f"{n_new} new batter(s)")
        if n_miss:
            parts.append(f"{n_miss} removed")
        st.info(f"Lineup changed since last sim. {', '.join(parts)}.")

    _render_matchup_tab_sidebyside(
        sides, h_arch, h_stat,
        arsenal_df, vuln_df, gpk,
        pos_lookup=pos_lookup,
        str_df=str_df,
        bf_priors=bf_priors,
        batter_sims_df=batter_sims,
        pitcher_sim_samples=pitcher_sim_samples,
        batter_sim_samples=batter_sim_samples,
    )


# ---------------------------------------------------------------------------
# Game selector (when no game_pk provided)
# ---------------------------------------------------------------------------

def _render_game_selector() -> int | None:
    """Date picker + game list, returns selected game_pk or None."""
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    today = et_now.date()

    dates = [today + timedelta(days=d) for d in range(-7, 8)]
    date_labels = []
    default_idx = 0
    for i, d in enumerate(dates):
        label = d.strftime("%a, %b %d")
        if d == today:
            label += "  (Today)"
            default_idx = i
        date_labels.append(label)

    selected_label = st.selectbox(
        "Game Date", date_labels, index=default_idx,
        key="game_page_date", label_visibility="collapsed",
    )
    selected_date = dates[date_labels.index(selected_label)]
    is_today = selected_date == today

    if is_today:
        schedule = load_todays_games()
        cached_date = (
            schedule["game_date"].iloc[0]
            if not schedule.empty and "game_date" in schedule.columns
            else None
        )
        if cached_date != today.isoformat():
            schedule = fetch_live_schedule(today.isoformat())
    else:
        schedule = fetch_live_schedule(selected_date.isoformat())

    if schedule.empty:
        st.info("No games found for this date.")
        return None

    if "game_datetime_utc" in schedule.columns:
        schedule = schedule.sort_values("game_datetime_utc", na_position="last")

    game_labels = []
    game_pks = []
    for _, g in schedule.iterrows():
        away = g.get("away_abbr", "?")
        home = g.get("home_abbr", "?")
        t = format_game_time(g.get("game_datetime_utc"), fallback=g.get("game_time", ""))
        game_labels.append(f"{away} @ {home} - {t}")
        game_pks.append(int(g["game_pk"]))

    sel = st.selectbox("Select Game", game_labels, key="game_page_game", label_visibility="collapsed")
    return game_pks[game_labels.index(sel)]


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def page_game() -> None:
    """Game Analysis page."""
    st.markdown(
        '<div class="section-header">Game Analysis</div>',
        unsafe_allow_html=True,
    )

    # Get game_pk from query params or show selector
    gpk_param = st.query_params.get("game_pk", "")
    gpk = int(gpk_param) if gpk_param.isdigit() else None

    if gpk is None:
        gpk = _render_game_selector()
        if gpk is None:
            return

    # Resolve game data
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    today = et_now.date()

    schedule = load_todays_games()
    game = _find_game(gpk, schedule)
    is_today_game = game is not None

    if game is None:
        # Try fetching from API for historical/future dates
        for delta in range(-7, 8):
            d = today + timedelta(days=delta)
            live_sched = fetch_live_schedule(d.isoformat())
            game = _find_game(gpk, live_sched)
            if game is not None:
                schedule = live_sched
                break

    if game is None:
        st.warning("Game not found. It may be outside the available date range.")
        if st.button("Back to Schedule"):
            st.query_params["page"] = "schedule"
            st.rerun()
        return

    lookups = _build_lookups()

    # Load sim data (only available for today's games)
    batter_sims = load_todays_batter_sims() if is_today_game else pd.DataFrame()
    pitcher_sim_samples = load_pitcher_game_sim_samples() if is_today_game else None
    batter_sim_samples = load_batter_game_sim_samples() if is_today_game else None

    # Load lineups
    if is_today_game:
        lineups = load_todays_lineups()
        lineups = backfill_missing_lineups(schedule, lineups)
    else:
        lineups = fetch_live_lineups(schedule) if not schedule.empty else pd.DataFrame()

    # Back link
    st.markdown(
        '<a href="?page=schedule" target="_self" style="color:var(--tdd-gold); '
        'font-size:0.8rem; text-decoration:none;">&#8592; Back to Schedule</a>',
        unsafe_allow_html=True,
    )

    # --- Section 1: Game Header ---
    _render_game_header(game, lookups)

    if not is_today_game:
        st.markdown(
            '<div class="tdd-meta" style="text-align:center; margin:0.5rem 0;">'
            'Simulations and projections are only available for today\'s games.</div>',
            unsafe_allow_html=True,
        )

    # --- Section 2: Pitcher Duel ---
    st.markdown(
        '<div class="section-subheader" style="margin-top:1.5rem;">Starting Pitchers</div>',
        unsafe_allow_html=True,
    )
    _render_pitcher_duel(game, lookups, pitcher_sim_samples)

    # --- Section 3: Key Matchups ---
    if is_today_game and not batter_sims.empty:
        _render_key_matchups(game, batter_sims, lookups)

    # --- Section 4: Lineup Matchups ---
    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty and "game_pk" in lineups.columns else pd.DataFrame()
    if not game_lu.empty:
        st.markdown(
            '<div class="section-subheader" style="margin-top:1.5rem;">Lineup Matchups</div>',
            unsafe_allow_html=True,
        )
        _render_lineup_matchups(
            game, lineups, lookups, batter_sims,
            pitcher_sim_samples, batter_sim_samples,
        )
    else:
        st.markdown(
            '<div class="tdd-meta" style="margin-top:1rem;">Lineups not yet available for this game.</div>',
            unsafe_allow_html=True,
        )
