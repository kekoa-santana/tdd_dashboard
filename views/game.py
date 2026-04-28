"""Game Analysis page -- dashboard variant with 12-col grid layout."""
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
    fetch_live_schedule, fetch_live_lineups, backfill_missing_lineups,
)
from components.team_logo import team_logo_html
from components.headshot import headshot_html
from components.grades import pitcher_grades_html, hitter_grades_html
from utils.alerts import tdd_info, tdd_warn
from utils.html import esc, esc_attr
from components.sim_chart import render_player_sim_from_props, PITCHER_STAT_META, BATTER_STAT_META
from components.scouting import render_scouting_html, compute_matchup_xwoba_edge
from lib.fantasy_report import load_report_data, get_pitcher_scouting, ReportData
from utils.helpers import format_ip, format_game_time
from utils.team_names import team_full


# ---------------------------------------------------------------------------
# Data helpers (unchanged)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _get_scouting_report_data() -> ReportData:
    return load_report_data(DASHBOARD_DIR)


@st.cache_data(ttl=300)
def _build_lookups() -> dict:
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
    if schedule.empty or "game_pk" not in schedule.columns:
        return None
    match = schedule[schedule["game_pk"] == game_pk]
    return match.iloc[0] if not match.empty else None


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def _stub_section(title: str) -> str:
    """Placeholder for sections without data yet."""
    return (
        '<div class="tdd-stub">'
        f'<div class="stub-title">{esc(title)}</div>'
        'Coming soon'
        '</div>'
    )


def _render_hero_html(game: pd.Series, lookups: dict) -> str:
    """Game hero scoreboard with teams, projected scores, and context."""
    away = game.get("away_abbr", "?")
    home = game.get("home_abbr", "?")
    away_name = game.get("away_team_name", away)
    home_name = game.get("home_team_name", home)
    standings = lookups["standings"]
    away_rec = f'{standings[away][0]}-{standings[away][1]}' if away in standings else ""
    home_rec = f'{standings[home][0]}-{standings[home][1]}' if home in standings else ""

    game_time = format_game_time(
        game.get("game_datetime_utc"), fallback=game.get("game_time", ""),
    )
    game_date = game.get("game_date", "")
    venue = game.get("venue_name", "")
    status_raw = game.get("status", "")

    # Status badge
    if "Progress" in status_raw or "Live" in status_raw:
        status_html = '<div class="gh-status" style="color:var(--tdd-sage);border-color:var(--tdd-sage);background:rgba(107,163,142,0.1)">LIVE</div>'
    elif "Final" in status_raw:
        status_html = '<div class="gh-status" style="color:var(--tdd-slate);border-color:var(--tdd-slate);background:rgba(123,143,166,0.1)">FINAL</div>'
    else:
        status_html = f'<div class="gh-status">{esc(game_time)}</div>'

    # Context line
    ctx_parts = []
    if venue:
        ctx_parts.append(venue)
    hp_ump = game.get("hp_umpire_name", "")
    if hp_ump:
        ctx_parts.append(f"HP: {hp_ump}")

    ctx_html = f'<div style="color:var(--tdd-slate);font-size:0.7rem;text-align:center;margin-top:0.5rem">{" / ".join(ctx_parts)}</div>' if ctx_parts else ""

    return (
        '<div class="tdd-ghero">'
        # Date row
        '<div class="gh-date">'
        f'<span class="eyebrow">Game Analysis</span>'
        f'<span>{esc(game_date)} / {esc(game_time)}</span>'
        '</div>'
        # Score row
        '<div class="gh-score">'
        # Away team
        '<div class="gh-team">'
        '<div class="gh-t-meta">'
        f'<div class="gh-city">{esc(away_name)}</div>'
        f'<div class="gh-name" data-team="{esc_attr(away)}">{esc(team_full(away))}</div>'
        f'<div class="gh-rec">{esc(away_rec)}</div>'
        '</div>'
        '</div>'
        # VS
        '<div class="gh-vs">'
        '<div class="gh-at">@</div>'
        f'{status_html}'
        '</div>'
        # Home team
        '<div class="gh-team home">'
        '<div class="gh-t-meta right">'
        f'<div class="gh-city">{esc(home_name)}</div>'
        f'<div class="gh-name" data-team="{esc_attr(home)}">{esc(team_full(home))}</div>'
        f'<div class="gh-rec">{esc(home_rec)}</div>'
        '</div>'
        '</div>'
        '</div>'
        f'{ctx_html}'
        '</div>'
    )


def _render_pitcher_duel_html(game: pd.Series, lookups: dict) -> str:
    """Pitcher duel in .tdd-pduel layout."""
    proj = lookups["proj"]
    p_arch = lookups["p_arch"]
    cards = ""

    for side in ("away", "home"):
        pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
        pid_raw = game.get(f"{side}_pitcher_id")
        pid = int(pid_raw) if pd.notna(pid_raw) else None
        side_abbr = game.get(f"{side}_abbr", "?")
        p_proj = proj.get(pid, {}) if pid else {}
        arch = p_arch.get(pid, "") if pid else ""

        hs = headshot_html(pid, size=50) if pid else ""
        meta_parts = [side_abbr]
        if arch:
            meta_parts.append(arch)
        if p_proj.get("pitch_hand"):
            meta_parts.append(f"{'LHP' if p_proj['pitch_hand'] == 'L' else 'RHP'}")

        # Vitals
        vitals = ""
        stat_pairs = [
            ("K%", p_proj.get("projected_k_rate")),
            ("BB%", p_proj.get("projected_bb_rate")),
            ("HR/BF", p_proj.get("projected_hr_per_bf")),
        ]
        for label, val in stat_pairs:
            if pd.notna(val):
                vitals += (
                    '<div class="pd-v">'
                    f'<div class="pd-vv">{float(val)*100:.1f}%</div>'
                    f'<div class="pd-vl">{label}</div>'
                    '</div>'
                )

        cards += (
            '<div class="pd-card">'
            '<div class="pd-head">'
            f'{hs}'
            '<div class="pd-nm">'
            f'<div class="pd-name">{esc(pitcher_name)}</div>'
            f'<div class="pd-meta">{esc(" / ".join(meta_parts))}</div>'
            '</div>'
            '</div>'
            f'<div class="pd-vitals">{vitals}</div>'
            '</div>'
        )

    return f'<div class="tdd-pduel">{cards}</div>'


def _render_edge_call_html(game: pd.Series, game_props: pd.DataFrame) -> str:
    """Diamond Edge Call box with headline pick + top 3 edges."""
    if game_props.empty or "expected" not in game_props.columns or "line" not in game_props.columns:
        return (
            '<div class="tdd-edgecall">'
            '<div class="ec-eyebrow">Diamond Edge</div>'
            '<div class="ec-primary">No projection data available</div>'
            '</div>'
        )

    work = game_props.copy()
    work["edge"] = work["expected"] - work["line"]
    work["abs_edge"] = work["edge"].abs()
    work = work.dropna(subset=["edge"])
    if work.empty:
        return (
            '<div class="tdd-edgecall">'
            '<div class="ec-eyebrow">Diamond Edge</div>'
            '<div class="ec-primary">No significant edges found</div>'
            '</div>'
        )

    top3 = work.nlargest(3, "abs_edge")
    best = top3.iloc[0]
    name = best.get("player_name", str(best.get("player_id", "")))
    stat = best.get("stat", "")
    edge = best["edge"]
    direction = "Over" if edge > 0 else "Under"
    edge_str = f"+{edge:.1f}" if edge > 0 else f"{edge:.1f}"
    line_val = best.get("line", 0)
    p_over = best.get("p_over", 0.5)

    primary = f"{name} {stat} {direction} {line_val:.1f}"
    secondary = f"Model projects {best.get('expected', 0):.1f} vs line {line_val:.1f}. Edge: {edge_str}."

    n_edges = len(work[(work["p_over"] >= 0.63) | (work["p_over"] <= 0.37)])

    # Runner-up edges (2nd and 3rd)
    runners_html = ""
    if len(top3) > 1:
        runner_rows = ""
        for _, row in top3.iloc[1:].iterrows():
            r_name = row.get("player_name", str(row.get("player_id", "")))
            r_stat = row.get("stat", "")
            r_edge = row["edge"]
            r_dir = "O" if r_edge > 0 else "U"
            r_edge_str = f"+{r_edge:.1f}" if r_edge > 0 else f"{r_edge:.1f}"
            r_line = row.get("line", 0)
            r_p_over = row.get("p_over", 0.5)
            r_color = "var(--tdd-sage)" if r_edge > 0 else "var(--tdd-ember)"
            runner_rows += (
                '<div style="display:flex;justify-content:space-between;align-items:baseline;'
                'padding:0.4rem 0;border-bottom:1px solid var(--tdd-dark-border-faint)">'
                '<div>'
                f'<span style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);'
                f'font-weight:600;font-size:0.85rem">{esc(r_name)}</span>'
                f'<span style="color:var(--tdd-slate);font-size:0.72rem;margin-left:0.5rem">'
                f'{esc(r_stat)} {r_dir} {r_line:.1f}</span>'
                '</div>'
                '<div style="display:flex;gap:0.8rem;align-items:baseline">'
                f'<span style="color:var(--tdd-slate);font-size:0.68rem">{r_p_over:.0%}</span>'
                f'<span style="color:{r_color};font-family:var(--tdd-font-heading);'
                f'font-weight:700;font-size:0.85rem">{esc(r_edge_str)}</span>'
                '</div>'
                '</div>'
            )
        runners_html = (
            f'<div style="margin-top:0.6rem;padding-top:0.6rem;'
            f'border-top:1px solid var(--tdd-dark-border)">'
            f'{runner_rows}'
            f'</div>'
        )

    return (
        '<div class="tdd-edgecall">'
        '<div class="ec-eyebrow">Diamond Edge</div>'
        f'<div class="ec-primary">{esc(primary)}</div>'
        f'<div class="ec-secondary">{esc(secondary)}</div>'
        '<div class="ec-meta">'
        f'<span>P(over) <b>{p_over:.0%}</b></span>'
        f'<span>Edges <b>{n_edges}</b></span>'
        f'<span>Best <b>{edge_str}</b></span>'
        '</div>'
        f'{runners_html}'
        '</div>'
    )


def _render_key_matchups_html(
    game: pd.Series,
    batter_sims: pd.DataFrame,
    lookups: dict,
) -> str:
    """Key matchups as HTML rows."""
    gpk = game["game_pk"]
    if batter_sims.empty or "game_pk" not in batter_sims.columns:
        return ""

    game_bs = batter_sims[batter_sims["game_pk"] == gpk].copy()
    if game_bs.empty or "matchup_k_lift" not in game_bs.columns:
        return ""

    game_bs["abs_lift"] = game_bs["matchup_k_lift"].abs()
    top = game_bs.nlargest(6, "abs_lift")
    if top.empty:
        return ""

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    str_df = load_hitter_strength(career=True)

    rows = ""
    for _, row in top.iterrows():
        bid = int(row["batter_id"])
        batter_name = row.get("batter_name", str(bid))
        team = row.get("team_abbr", "")
        opp_starter = int(row["opp_starter_id"]) if pd.notna(row.get("opp_starter_id")) else None

        badge = ""
        if opp_starter and not arsenal_df.empty and not vuln_df.empty:
            _p_ars = arsenal_df[arsenal_df["pitcher_id"] == opp_starter]
            _h_vul = vuln_df[vuln_df["batter_id"] == bid]
            _h_str = str_df[str_df["batter_id"] == bid] if not str_df.empty else pd.DataFrame()
            _ph = str(_p_ars["pitch_hand"].iloc[0]) if not _p_ars.empty and "pitch_hand" in _p_ars.columns else None
            _bh = str(_h_vul["batter_stand"].iloc[0]) if not _h_vul.empty and "batter_stand" in _h_vul.columns else None
            if not _p_ars.empty and not _h_vul.empty:
                _edge = compute_matchup_xwoba_edge(_p_ars, _h_vul, _h_str, pitcher_hand=_ph, batter_hand=_bh)
                _adv = _edge["advantage"]
                _xw = _edge["matchup_xwoba"]
                if _adv == "pitcher":
                    badge = f'<span style="color:var(--tdd-ember);font-size:0.68rem;font-weight:600">Pitcher .{int(_xw*1000):03d}</span>'
                elif _adv == "hitter":
                    badge = f'<span style="color:var(--tdd-sage);font-size:0.68rem;font-weight:600">Hitter .{int(_xw*1000):03d}</span>'
                else:
                    badge = f'<span style="color:var(--tdd-slate);font-size:0.68rem">Even .{int(_xw*1000):03d}</span>'

        rows += (
            '<div style="display:flex;gap:10px;align-items:center;'
            'padding:0.5rem 0;border-bottom:1px solid var(--tdd-dark-border-faint)">'
            f'{headshot_html(bid, size=36)}'
            '<div style="flex:1;min-width:0">'
            f'<div style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);font-weight:700;font-size:0.85rem">{esc(batter_name)}</div>'
            f'<div style="color:var(--tdd-slate);font-size:0.68rem">{esc(team)}</div>'
            '</div>'
            f'<div>{badge}</div>'
            '</div>'
        )

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
        '<div class="gsec-head">Key Matchups</div>'
        f'{rows}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Game selector (unchanged)
# ---------------------------------------------------------------------------

def _render_game_selector() -> int | None:
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
        tdd_info("No games found for this date.")
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
    """Game Analysis page -- dashboard variant."""

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
        for delta in range(-7, 8):
            d = today + timedelta(days=delta)
            live_sched = fetch_live_schedule(d.isoformat())
            game = _find_game(gpk, live_sched)
            if game is not None:
                schedule = live_sched
                break

    if game is None:
        tdd_warn("Game not found. It may be outside the available date range.")
        if st.button("Back to Schedule"):
            st.query_params["page"] = "schedule"
            st.rerun()
        return

    lookups = _build_lookups()

    # Load sim data
    batter_sims = load_todays_batter_sims() if is_today_game else pd.DataFrame()
    if is_today_game:
        from services.data_loader import load_game_props
        _gp_all = load_game_props()
        game_props_df = (
            _gp_all[_gp_all["game_pk"] == gpk] if not _gp_all.empty else pd.DataFrame()
        )
    else:
        game_props_df = pd.DataFrame()

    # Load lineups
    if is_today_game:
        lineups = load_todays_lineups()
        lineups = backfill_missing_lineups(schedule, lineups)
    else:
        lineups = fetch_live_lineups(schedule) if not schedule.empty else pd.DataFrame()

    # Back link
    st.markdown(
        '<a href="?page=schedule" target="_self" style="color:var(--tdd-gold);'
        'font-size:0.8rem;text-decoration:none;">&#8592; Back to Schedule</a>',
        unsafe_allow_html=True,
    )

    # === DASHBOARD LAYOUT ========================================
    # Build all HTML sections, then render as single blob per grid row
    # to avoid Streamlit container gaps.

    parts: list[str] = ['<div class="tdd-game">']

    # Hero scoreboard
    parts.append(_render_hero_html(game, lookups))

    # Row 1: Edge Call + Key Matchups (grid 6+6)
    edge_html = _render_edge_call_html(game, game_props_df)
    matchups_html = _render_key_matchups_html(game, batter_sims, lookups) if is_today_game else _stub_section("Key Matchups")

    parts.append(
        '<div class="section grid12">'
        f'<div class="col-6">{edge_html}</div>'
        f'<div class="col-6">{matchups_html}</div>'
        '</div>'
    )

    # Row 2: Pitcher duel (full width)
    parts.append(
        '<div class="section">'
        '<div class="gsec-head">Starting Pitchers</div>'
        + _render_pitcher_duel_html(game, lookups) +
        '</div>'
    )

    # Row 3: Weather + Umpire stubs (grid 6+6)
    parts.append(
        '<div class="section grid12">'
        f'<div class="col-6">{_stub_section("Weather + Park")}</div>'
        f'<div class="col-6">{_stub_section("Home Plate Umpire")}</div>'
        '</div>'
    )

    # Row 4: Bullpen + H2H + News stubs (grid 4+4+4)
    parts.append(
        '<div class="section grid12">'
        f'<div class="col-4">{_stub_section("Bullpen Away")}</div>'
        f'<div class="col-4">{_stub_section("Bullpen Home")}</div>'
        f'<div class="col-4">{_stub_section("H2H Record")}</div>'
        '</div>'
    )

    parts.append('</div>')  # close .tdd-game

    # Render the HTML grid
    st.markdown("".join(parts), unsafe_allow_html=True)

    # === INTERACTIVE SECTIONS (need Streamlit widgets) ============
    # These use st.columns, st.plotly_chart, etc. so they can't be
    # part of the single HTML blob above.

    # Pitcher sim distributions (Plotly charts)
    if is_today_game and not game_props_df.empty:
        st.markdown(
            '<div style="margin-top:1rem">'
            '<div class="gsec-head" style="color:var(--tdd-gold);font-family:var(--tdd-font-heading);'
            'font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;'
            'margin-bottom:6px">Pitcher Projections</div></div>',
            unsafe_allow_html=True,
        )
        col_a, col_h = st.columns(2)
        for side, col in [("away", col_a), ("home", col_h)]:
            pid_raw = game.get(f"{side}_pitcher_id")
            pid = int(pid_raw) if pd.notna(pid_raw) else None
            if pid:
                with col:
                    pitcher_name = game.get(f"{side}_pitcher_name", "")
                    st.markdown(
                        f'<div style="text-align:center;color:var(--tdd-cream);'
                        f'font-family:var(--tdd-font-heading);font-weight:700;'
                        f'font-size:0.85rem;margin-bottom:0.3rem">{esc(pitcher_name)}</div>',
                        unsafe_allow_html=True,
                    )
                    _p_rows = game_props_df[
                        (game_props_df["player_id"] == pid)
                        & (game_props_df.get("player_type", "pitcher") == "pitcher")
                    ]
                    if not _p_rows.empty:
                        render_player_sim_from_props(
                            _p_rows, PITCHER_STAT_META,
                            ["K", "BB", "H", "HR", "Outs"],
                            f"gp_p_{side}_{gpk}",
                        )

    # Lineup matchups (uses schedule.py machinery with st.expanders)
    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty and "game_pk" in lineups.columns else pd.DataFrame()
    if not game_lu.empty:
        st.markdown(
            '<div style="margin-top:1rem">'
            '<div class="gsec-head" style="color:var(--tdd-gold);font-family:var(--tdd-font-heading);'
            'font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;'
            'margin-bottom:6px">Lineup Matchups</div></div>',
            unsafe_allow_html=True,
        )
        from views.schedule import (
            _render_matchup_tab_sidebyside,
            _detect_lineup_changes,
        )

        h_arch = lookups["h_arch"]
        p_arch = lookups["p_arch"]
        h_stat = lookups["h_stat"]
        proj_lookup = lookups["proj"]

        pos_lookup = {}
        for _, r in game_lu.iterrows():
            bid = int(r.get("batter_id", 0))
            pos = r.get("game_position", "")
            if bid and pos:
                pos_lookup[bid] = pos

        sides = []
        for side, opp_side in [("away", "home"), ("home", "away")]:
            pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
            pid_raw = game.get(f"{side}_pitcher_id")
            pid = int(pid_raw) if pd.notna(pid_raw) else None
            side_team_id = game.get(f"{side}_team_id")
            opp_team_id = game.get(f"{opp_side}_team_id")
            opp_abbr = game.get(f"{opp_side}_abbr", "?")
            side_abbr = game.get(f"{side}_abbr", "?")

            opp_lu = (
                game_lu[game_lu["team_id"] == opp_team_id].sort_values("batting_order")
                if pd.notna(opp_team_id) else pd.DataFrame()
            )
            own_lu = (
                game_lu[game_lu["team_id"] == side_team_id].sort_values("batting_order")
                if pd.notna(side_team_id) else pd.DataFrame()
            )

            sides.append({
                "side": side,
                "abbr": side_abbr,
                "opp_abbr": opp_abbr,
                "pitcher_name": pitcher_name,
                "pitcher_id": pid,
                "pitcher_arch": p_arch.get(pid) if pid else None,
                "pitcher_proj": proj_lookup.get(pid, {}) if pid else {},
                "opp_lineup": opp_lu,
                "own_lineup": own_lu,
            })

        arsenal_df = load_pitcher_arsenal()
        vuln_df = load_hitter_vulnerability(career=True)
        str_df = load_hitter_strength(career=True)
        bf_priors_df = load_bf_priors()

        _render_matchup_tab_sidebyside(
            sides, h_arch, h_stat,
            arsenal_df, vuln_df, gpk,
            pos_lookup=pos_lookup,
            str_df=str_df,
            bf_priors=bf_priors_df,
            batter_sims_df=batter_sims,
        )
    elif not is_today_game:
        st.markdown(
            '<div class="tdd-meta" style="margin-top:1rem">'
            'Simulations and projections are only available for today\'s games.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="tdd-meta" style="margin-top:1rem">Lineups not yet available.</div>',
            unsafe_allow_html=True,
        )
