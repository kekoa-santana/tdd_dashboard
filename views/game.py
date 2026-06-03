"""Game Analysis page -- dashboard variant with 12-col grid layout."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DASHBOARD_DIR
from services.data_loader import (
    load_todays_games, load_todays_lineups, load_todays_batter_sims,
    load_update_metadata, load_pitcher_arsenal,
    load_projections, load_hitter_archetypes,
    load_pitcher_archetypes,
    fetch_live_schedule, fetch_live_lineups, backfill_missing_lineups,
    load_park_factors, load_hr_park_factors, load_umpire_tendencies,
    load_reliever_rankings, load_roster,
)
from components.team_logo import team_logo_html
from components.headshot import headshot_html
from components.grades import pitcher_grades_html, hitter_grades_html
from utils.alerts import tdd_info, tdd_warn
from utils.html import esc, esc_attr
from components.sim_chart import render_player_sim_from_props, PITCHER_STAT_META
from utils.helpers import format_ip, format_game_time
from utils.team_names import team_full


# ---------------------------------------------------------------------------
# Data helpers (unchanged)
# ---------------------------------------------------------------------------

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


def _render_weather_park_html(game: pd.Series) -> str:
    """Weather conditions + park factor card."""
    venue = game.get("venue_name", "")
    venue_id = game.get("venue_id")

    # Look up venue name from hr_park_factors if not in game data
    if not venue and pd.notna(venue_id):
        hpf = load_hr_park_factors()
        if not hpf.empty and "venue_id" in hpf.columns and "venue_name" in hpf.columns:
            match = hpf[hpf["venue_id"] == int(venue_id)]
            if not match.empty:
                venue = str(match.iloc[0]["venue_name"])
    temp = game.get("weather_temp")
    wind_speed = game.get("weather_wind_speed")
    wind_dir = game.get("weather_wind_direction", "")
    condition = game.get("weather_condition", "")

    # Weather row
    weather_parts = []
    if pd.notna(temp):
        weather_parts.append(f"{int(temp)}&deg;F")
    if pd.notna(wind_speed) and wind_dir:
        weather_parts.append(f"{int(wind_speed)} mph {esc(wind_dir)}")
    if condition:
        weather_parts.append(esc(condition))
    weather_line = " &middot; ".join(weather_parts) if weather_parts else "Not available"

    # Park factors
    pf_html = ""
    if pd.notna(venue_id):
        pf_df = load_park_factors()
        if not pf_df.empty and "venue_id" in pf_df.columns:
            venue_pf = pf_df[pf_df["venue_id"] == int(venue_id)]
            if not venue_pf.empty:
                pf_items = ""
                for _, row in venue_pf.iterrows():
                    stat = row.get("stat", "")
                    pf_val = row.get("park_factor_regressed", 1.0)
                    if pd.isna(pf_val):
                        continue
                    pf_pct = (pf_val - 1) * 100
                    if abs(pf_pct) < 1:
                        pf_color = "var(--tdd-slate)"
                        pf_label = "neutral"
                    elif pf_pct > 0:
                        pf_color = "var(--tdd-ember)"
                        pf_label = f"+{pf_pct:.0f}%"
                    else:
                        pf_color = "var(--tdd-sage)"
                        pf_label = f"{pf_pct:.0f}%"
                    pf_items += (
                        '<div class="pd-v">'
                        f'<div class="pd-vv" style="color:{pf_color}">{esc(pf_label)}</div>'
                        f'<div class="pd-vl">{esc(stat)}</div>'
                        '</div>'
                    )
                if pf_items:
                    pf_html = f'<div class="pd-vitals" style="margin-top:0.6rem">{pf_items}</div>'

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
        '<div class="gsec-head">Weather + Park</div>'
        f'<div style="color:var(--tdd-cream);font-size:0.85rem;margin-bottom:0.3rem">{esc(venue)}</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.78rem">{weather_line}</div>'
        f'{pf_html}'
        '</div>'
    )


def _render_umpire_html(game: pd.Series) -> str:
    """Home plate umpire tendency card."""
    ump_name = game.get("hp_umpire_name", "")
    if not ump_name:
        return (
            '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
            '<div class="gsec-head">Home Plate Umpire</div>'
            '<div style="color:var(--tdd-slate);font-size:0.78rem">Not yet assigned</div>'
            '</div>'
        )

    ump_df = load_umpire_tendencies()
    if ump_df.empty or "hp_umpire_name" not in ump_df.columns:
        return (
            '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
            '<div class="gsec-head">Home Plate Umpire</div>'
            f'<div style="color:var(--tdd-cream);font-size:0.85rem">{esc(ump_name)}</div>'
            '<div style="color:var(--tdd-slate);font-size:0.78rem">No tendency data</div>'
            '</div>'
        )

    match = ump_df[ump_df["hp_umpire_name"] == ump_name]
    if match.empty:
        # Fall back to league-average tendencies
        items = ""
        for label in ["K Tendency", "BB Tendency", "HR Tendency"]:
            items += (
                '<div class="pd-v">'
                '<div class="pd-vv" style="color:var(--tdd-slate)">neutral</div>'
                f'<div class="pd-vl">{esc(label)}</div>'
                '</div>'
            )
        return (
            '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
            '<div class="gsec-head">Home Plate Umpire</div>'
            f'<div style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);font-weight:700;font-size:0.85rem">{esc(ump_name)}</div>'
            '<div style="color:var(--tdd-slate);font-size:0.72rem;margin-bottom:0.4rem">Using league-average umpire tendencies</div>'
            f'<div class="pd-vitals">{items}</div>'
            '</div>'
        )

    u = match.iloc[0]
    games = int(u.get("games", 0))

    # Build tendency items
    items = ""
    for stat, lift_col, label in [
        ("K", "k_logit_lift", "K Tendency"),
        ("BB", "bb_logit_lift", "BB Tendency"),
        ("HR", "hr_logit_lift", "HR Tendency"),
    ]:
        lift = u.get(lift_col, 0)
        if pd.isna(lift):
            continue
        lift = float(lift)
        if abs(lift) < 0.02:
            color = "var(--tdd-slate)"
            desc = "neutral"
        elif lift > 0:
            color = "var(--tdd-ember)"
            desc = f"+{lift:.2f}"
        else:
            color = "var(--tdd-sage)"
            desc = f"{lift:.2f}"
        items += (
            '<div class="pd-v">'
            f'<div class="pd-vv" style="color:{color}">{esc(desc)}</div>'
            f'<div class="pd-vl">{esc(label)}</div>'
            '</div>'
        )

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
        '<div class="gsec-head">Home Plate Umpire</div>'
        f'<div style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);font-weight:700;font-size:0.85rem">{esc(ump_name)}</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.72rem;margin-bottom:0.4rem">{games} career games</div>'
        f'<div class="pd-vitals">{items}</div>'
        '</div>'
    )


def _render_bullpen_html(game: pd.Series, side: str) -> str:
    """Bullpen card for one team showing top relievers."""
    team_id = game.get(f"{side}_team_id")
    team_abbr = game.get(f"{side}_abbr", "?")

    if pd.isna(team_id):
        return _stub_section(f"Bullpen {team_abbr}")

    team_id = int(team_id)

    # Get relievers for this team via roster
    roster_df = load_roster()
    rr_df = load_reliever_rankings()

    if roster_df.empty or rr_df.empty:
        return _stub_section(f"Bullpen {team_abbr}")

    # Map pitcher_id → team_abbr from roster
    team_pitchers = roster_df[roster_df["team_abbr"] == team_abbr]
    if team_pitchers.empty:
        return _stub_section(f"Bullpen {team_abbr}")

    team_pids = set(team_pitchers["player_id"].astype(int))
    team_rr = rr_df[rr_df["pitcher_id"].isin(team_pids)].copy()

    if team_rr.empty:
        return _stub_section(f"Bullpen {team_abbr}")

    # Sort by role priority then value
    role_order = {"CL": 0, "SU": 1, "MR": 2, "LR": 3}
    team_rr["_role_ord"] = team_rr["role"].map(role_order).fillna(4)
    team_rr = team_rr.sort_values(["_role_ord", "tdd_value_score"], ascending=[True, False])

    # Get names from roster
    name_map = dict(zip(
        team_pitchers["player_id"].astype(int),
        team_pitchers["player_name"],
    ))

    rows = ""
    for _, r in team_rr.head(5).iterrows():
        pid = int(r["pitcher_id"])
        name = name_map.get(pid, str(pid))
        # Shorten long names
        if len(name) > 16:
            parts = name.split()
            name = f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name[:16]
        role = r.get("role", "?")
        hand = r.get("pitch_hand", "?")
        k_pct = r.get("k_pct", 0)
        bb_pct = r.get("bb_pct", 0)

        role_color = {
            "CL": "var(--tdd-ember)", "SU": "var(--tdd-gold)",
            "MR": "var(--tdd-slate)", "LR": "var(--tdd-slate)",
        }.get(role, "var(--tdd-slate)")

        rows += (
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:0.35rem 0;border-bottom:1px solid var(--tdd-dark-border-faint)">'
            '<div style="display:flex;gap:0.4rem;align-items:baseline;flex:1;min-width:0">'
            f'{headshot_html(pid, size=24)}'
            f'<span style="color:var(--tdd-cream);font-size:0.78rem;font-weight:600">{esc(name)}</span>'
            f'<span style="color:var(--tdd-slate);font-size:0.65rem">{esc(hand)}HP</span>'
            '</div>'
            '<div style="display:flex;gap:0.6rem;align-items:baseline">'
            f'<span style="color:var(--tdd-slate);font-size:0.65rem">K:{k_pct*100 if pd.notna(k_pct) else 0:.0f}%</span>'
            f'<span style="color:{role_color};font-family:var(--tdd-font-heading);'
            f'font-weight:700;font-size:0.72rem">{esc(role)}</span>'
            '</div>'
            '</div>'
        )

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
        f'<div class="gsec-head">Bullpen {esc(team_abbr)}</div>'
        f'{rows}'
        '</div>'
    )


def _describe_arsenal(pid: int) -> tuple[str, str]:
    """Return (primary weapon description, top pitch for K narrative).

    E.g. ("changeup-heavy LHP", "changeup (49% whiff)")
    """
    arsenal_df = load_pitcher_arsenal()
    if arsenal_df.empty:
        return "", ""
    pa = arsenal_df[arsenal_df["pitcher_id"] == pid].copy()
    if pa.empty:
        return "", ""

    pa = pa.sort_values("usage_pct", ascending=False)
    hand = str(pa.iloc[0].get("pitch_hand", "R"))
    hand_label = "LHP" if hand == "L" else "RHP"

    # Top whiff pitch (min 15% usage)
    whiff_candidates = pa[pa["usage_pct"] >= 0.15].copy()
    if whiff_candidates.empty:
        whiff_candidates = pa.head(3)
    best_whiff = whiff_candidates.sort_values("whiff_rate", ascending=False).iloc[0]
    whiff_pt = str(best_whiff.get("pitch_type", ""))
    whiff_rate = float(best_whiff.get("whiff_rate", 0))

    _PT_NAMES = {
        "FF": "four-seam", "SI": "sinker", "FC": "cutter",
        "SL": "slider", "CU": "curveball", "KC": "knuckle-curve",
        "CH": "changeup", "FS": "splitter", "ST": "sweeper",
        "SV": "slurve", "KN": "knuckleball",
    }
    whiff_name = _PT_NAMES.get(whiff_pt, whiff_pt)

    # Arsenal shape description
    primary = pa.iloc[0]
    primary_pt = str(primary.get("pitch_type", ""))
    primary_name = _PT_NAMES.get(primary_pt, primary_pt)
    primary_usage = float(primary.get("usage_pct", 0))

    # Count pitch families
    breaking = pa[pa["pitch_type"].isin(["SL", "CU", "KC", "ST", "SV"])]["usage_pct"].sum()
    offspeed = pa[pa["pitch_type"].isin(["CH", "FS"])]["usage_pct"].sum()

    if primary_usage >= 0.55:
        shape = f"{primary_name}-heavy {hand_label}"
    elif breaking >= 0.35:
        shape = f"breaking ball-heavy {hand_label}"
    elif offspeed >= 0.25:
        shape = f"{hand_label} with a strong offspeed mix"
    else:
        shape = f"{hand_label} with a balanced arsenal"

    whiff_desc = f"{whiff_name} ({whiff_rate*100:.0f}% whiff)" if whiff_rate > 0.20 else whiff_name

    return shape, whiff_desc


def _render_game_plan_html(
    game: pd.Series,
    game_props: pd.DataFrame,
    batter_sims: pd.DataFrame,
    lookups: dict,
) -> str:
    """Auto-generated narrative game plan for each side.

    Generates pitcher-specific scouting narratives with arsenal context,
    matchup dynamics, and environment notes.
    """
    gpk = game["game_pk"]

    # Gather both sides' pitcher data for game-level framing
    side_data = {}
    for side in ["away", "home"]:
        pitch_side = "home" if side == "away" else "away"
        pid_raw = game.get(f"{pitch_side}_pitcher_id")
        pid = int(pid_raw) if pd.notna(pid_raw) else None
        proj = lookups["proj"]
        p_proj = proj.get(pid, {}) if pid else {}
        side_data[side] = {
            "pid": pid,
            "pitcher_name": game.get(f"{pitch_side}_pitcher_name") or "TBD",
            "k_rate": float(p_proj.get("projected_k_rate", 0)) if p_proj.get("projected_k_rate") else 0,
            "bb_rate": float(p_proj.get("projected_bb_rate", 0)) if p_proj.get("projected_bb_rate") else 0,
        }

    # Game-level framing
    away_k = side_data["away"]["k_rate"]
    home_k = side_data["home"]["k_rate"]
    away_bb = side_data["away"]["bb_rate"]
    home_bb = side_data["home"]["bb_rate"]
    avg_k = (away_k + home_k) / 2 if (away_k > 0 and home_k > 0) else 0
    avg_bb = (away_bb + home_bb) / 2 if (away_bb > 0 and home_bb > 0) else 0

    game_frame = ""
    if avg_k > 0:
        if avg_k >= 0.26:
            game_frame = "Both starters miss bats — expect a low-contact, high-strikeout game."
        elif avg_k <= 0.19:
            game_frame = "Neither starter is a big K threat — expect balls in play and a higher-scoring game."
        elif abs(away_k - home_k) >= 0.06:
            high_side = "away" if away_k > home_k else "home"
            high_name = side_data[high_side]["pitcher_name"]
            game_frame = f"Pitching mismatch: {esc(high_name)} has a clear stuff advantage in this one."
        if avg_bb >= 0.09:
            game_frame += " Both pitchers walk hitters — free baserunners will be in play." if game_frame else "Walk-prone starters on both sides — free baserunners will drive this game."

    blocks = ""

    for side in ["away", "home"]:
        pitch_side = "home" if side == "away" else "away"
        pitcher_name = side_data[side]["pitcher_name"]  # pitcher facing THIS side's batters
        # Wait — side_data[side] has the pitcher facing the OTHER side's batters
        # We need the pitcher that THIS side's batters face
        pitcher_name = game.get(f"{pitch_side}_pitcher_name") or "TBD"
        pid = side_data[side]["pid"]  # this is the pitcher the other side faces
        # Fix: the pitcher facing side's batters is on pitch_side
        pid_raw = game.get(f"{pitch_side}_pitcher_id")
        pid = int(pid_raw) if pd.notna(pid_raw) else None
        batting_abbr = game.get(f"{side}_abbr", "?")

        if pid is None:
            continue

        proj = lookups["proj"]
        p_proj = proj.get(pid, {}) if pid else {}
        k_rate = float(p_proj.get("projected_k_rate", 0)) if p_proj.get("projected_k_rate") else 0
        bb_rate = float(p_proj.get("projected_bb_rate", 0)) if p_proj.get("projected_bb_rate") else 0

        # Pitcher game-level props
        p_props = game_props[
            (game_props["player_id"] == pid)
            & (game_props["player_type"] == "pitcher")
        ] if not game_props.empty and "player_type" in game_props.columns else pd.DataFrame()

        exp_k, exp_bb, exp_ip = 0.0, 0.0, 0.0
        ump_k, wx_k = 0.0, 0.0
        if not p_props.empty:
            stat_map = {}
            for _, row in p_props.iterrows():
                stat_map[row["stat"]] = float(row["expected"]) if pd.notna(row.get("expected")) else 0
            exp_k = stat_map.get("K", 0)
            exp_bb = stat_map.get("BB", 0)
            exp_ip = float(p_props.iloc[0].get("expected_ip", 0)) if pd.notna(p_props.iloc[0].get("expected_ip")) else 0
            ump_k = float(p_props.iloc[0].get("umpire_k_lift", 0)) if pd.notna(p_props.iloc[0].get("umpire_k_lift")) else 0
            wx_k = float(p_props.iloc[0].get("weather_k_lift", 0)) if pd.notna(p_props.iloc[0].get("weather_k_lift")) else 0

        # Batter matchup data
        side_batters = batter_sims[
            (batter_sims["game_pk"] == gpk)
            & (batter_sims["team_abbr"] == batting_abbr)
        ] if not batter_sims.empty and "team_abbr" in batter_sims.columns else pd.DataFrame()

        # Guard: skip if no projection data at all
        if k_rate == 0 and side_batters.empty:
            continue

        # Arsenal info
        arsenal_shape, top_whiff_pitch = _describe_arsenal(pid) if pid else ("", "")

        # === Pitcher scouting narrative ===
        narrative_parts = []

        # Guard for missing projections
        if k_rate == 0:
            narrative_parts.append(
                f"{esc(pitcher_name)} — limited projection data available. "
                f"Approach with standard game plan."
            )
        else:
            # Varied pitcher profile sentence based on archetype
            if k_rate >= 0.28 and bb_rate <= 0.06:
                profile = f"an elite arm who dominates the zone — {k_rate*100:.0f}% K rate with pinpoint command ({bb_rate*100:.0f}% BB)"
            elif k_rate >= 0.28 and bb_rate >= 0.09:
                profile = f"a high-K arm with wildness — {k_rate*100:.0f}% K rate but walks {bb_rate*100:.0f}% of batters"
            elif k_rate >= 0.23 and bb_rate <= 0.06:
                profile = f"a strike-thrower who limits free passes — {k_rate*100:.0f}% K, {bb_rate*100:.0f}% BB"
            elif k_rate >= 0.23:
                profile = f"solid strikeout ability ({k_rate*100:.0f}% K) with some walk risk ({bb_rate*100:.0f}% BB)"
            elif k_rate <= 0.18:
                profile = f"a contact manager — {k_rate*100:.0f}% K rate means balls in play, and that's where the offense can attack"
            else:
                profile = f"a mid-rotation arm — {k_rate*100:.0f}% K rate, {bb_rate*100:.0f}% BB rate"

            if arsenal_shape:
                profile = f"{arsenal_shape}, {profile}"
            else:
                profile = f"{esc(pitcher_name)} is {profile}"
                arsenal_shape = ""

            if arsenal_shape:
                narrative_parts.append(f"{esc(pitcher_name)} is a {profile}.")
            else:
                narrative_parts.append(f"{profile}.")

            # Game projection line (when available)
            if exp_k > 0:
                narrative_parts.append(
                    f"Game projection: <b>{exp_k:.1f} K</b>, "
                    f"<b>{exp_bb:.1f} BB</b>, <b>{exp_ip:.1f} IP</b>."
                )

            # Arsenal-specific approach advice
            if top_whiff_pitch and k_rate >= 0.22:
                narrative_parts.append(
                    f"His {top_whiff_pitch} is the primary out pitch — "
                    f"batters who can lay off it have the edge."
                )
            elif bb_rate >= 0.09:
                narrative_parts.append(
                    f"Be patient at the plate — he walks {bb_rate*100:.0f}% of hitters "
                    f"and will put himself in hitter's counts."
                )

        # Environment notes (only when meaningful)
        env_parts = []
        if abs(ump_k) >= 0.03:
            env_parts.append("K-friendly zone" if ump_k > 0 else "hitter-friendly zone")
        if abs(wx_k) >= 0.02:
            env_parts.append("weather boosts Ks" if wx_k > 0 else "weather suppresses Ks")
        if env_parts:
            narrative_parts.append(f"Environment: {', '.join(env_parts)}.")

        # === Lineup threats (projection-based) ===
        dynamics = []
        if not side_batters.empty:
            sb = side_batters.copy()

            # Walk upside (only for walk-prone pitchers)
            if bb_rate >= 0.08 and "expected_bb" in sb.columns:
                high_bb = sb[sb["expected_bb"] >= 0.5]
                if len(high_bb) >= 2:
                    names = ", ".join(high_bb.nlargest(2, "expected_bb")["batter_name"].tolist())
                    dynamics.append(
                        f'Walk upside for <b>{esc(names)}</b> — this pitcher puts '
                        f'runners on at a {bb_rate*100:.0f}% clip'
                    )

            # HR threat
            if "expected_hr" in sb.columns:
                hr_threats = sb[sb["expected_hr"] >= 0.20]
                if not hr_threats.empty:
                    best_hr = hr_threats.nlargest(1, "expected_hr").iloc[0]
                    dynamics.append(
                        f'<b>{esc(best_hr["batter_name"])}</b> carries the most '
                        f'HR upside ({best_hr["expected_hr"]:.2f} expected)'
                    )

        # === Render ===
        narrative_html = ""
        for p in narrative_parts:
            narrative_html += (
                f'<div style="color:var(--tdd-cream);font-size:0.82rem;'
                f'line-height:1.5;margin-bottom:0.4rem">{p}</div>'
            )

        dynamics_html = ""
        if dynamics:
            for d in dynamics[:4]:
                dynamics_html += (
                    f'<div style="color:var(--tdd-cream);font-size:0.78rem;'
                    f'padding:0.25rem 0;border-bottom:1px solid var(--tdd-dark-border-faint)">'
                    f'<span style="color:var(--tdd-gold);margin-right:0.4rem">&#9670;</span>{d}'
                    f'</div>'
                )

        blocks += (
            f'<div class="col-6">'
            '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);padding:0.8rem 1rem">'
            f'<div class="gsec-head">{esc(batting_abbr)} Game Plan vs {esc(pitcher_name)}</div>'
            f'{narrative_html}'
            f'{dynamics_html}'
            '</div>'
            '</div>'
        )

    if not blocks:
        return ""

    # Game-level framing above the two side panels
    frame_html = ""
    if game_frame:
        frame_html = (
            f'<div style="color:var(--tdd-cream);font-size:0.85rem;font-weight:600;'
            f'text-align:center;padding:0.5rem;margin-bottom:0.3rem;'
            f'background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border)">'
            f'{game_frame}</div>'
        )

    return (
        '<div class="section">'
        f'{frame_html}'
        f'<div class="grid12">{blocks}</div>'
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

    # Row 1: Pitcher duel (full width)
    parts.append(
        '<div class="section">'
        '<div class="gsec-head">Starting Pitchers</div>'
        + _render_pitcher_duel_html(game, lookups) +
        '</div>'
    )

    # Row 2: Game Plan narratives (grid 6+6)
    if is_today_game:
        game_plan_html = _render_game_plan_html(game, game_props_df, batter_sims, lookups)
        if game_plan_html:
            parts.append(game_plan_html)

    # Row 3: Weather + Umpire (grid 6+6)
    parts.append(
        '<div class="section grid12">'
        f'<div class="col-6">{_render_weather_park_html(game)}</div>'
        f'<div class="col-6">{_render_umpire_html(game)}</div>'
        '</div>'
    )

    # Row 4: Bullpen + H2H (grid 4+4+4)
    parts.append(
        '<div class="section grid12">'
        f'<div class="col-4">{_render_bullpen_html(game, "away")}</div>'
        f'<div class="col-4">{_render_bullpen_html(game, "home")}</div>'
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

    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty and "game_pk" in lineups.columns else pd.DataFrame()
    if game_lu.empty and not is_today_game:
        st.markdown(
            '<div class="tdd-meta" style="margin-top:1rem">'
            'Simulations and projections are only available for today\'s games.</div>',
            unsafe_allow_html=True,
        )
    elif game_lu.empty:
        st.markdown(
            '<div class="tdd-meta" style="margin-top:1rem">Lineups not yet available.</div>',
            unsafe_allow_html=True,
        )
