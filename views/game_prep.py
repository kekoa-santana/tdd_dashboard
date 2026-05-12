"""Game Prep Report -- MLB-style pregame preparation for coaching staff."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM
from services.data_loader import (
    load_todays_games, load_todays_lineups,
    load_pitcher_arsenal, load_pitcher_arsenal_by_stand,
    load_pitcher_putaway,
    load_hitter_vulnerability, load_hitter_strength,
    load_projections, load_pitcher_archetypes, load_hitter_archetypes,
    load_roster,
    backfill_missing_lineups,
    fetch_live_schedule, fetch_live_lineups,
)
from components.scouting import (
    build_attack_plan, compute_matchup_xwoba_edge, PITCH_DISPLAY,
)
from components.headshot import headshot_html
from components.team_logo import team_logo_html
from utils.html import esc, esc_attr
from utils.helpers import format_game_time


# ---------------------------------------------------------------------------
# Approach styling
# ---------------------------------------------------------------------------

_APPROACH_STYLE = {
    "hunt":       {"color": SAGE,  "bg": "rgba(107,163,142,0.12)", "label": "HUNT",       "icon": "&#9679;"},
    "aggressive": {"color": SAGE,  "bg": "rgba(107,163,142,0.08)", "label": "AGGRESSIVE", "icon": "&#9650;"},
    "neutral":    {"color": SLATE, "bg": "rgba(123,143,166,0.08)", "label": "NEUTRAL",    "icon": "&#9644;"},
    "defensive":  {"color": EMBER, "bg": "rgba(212,86,42,0.08)",  "label": "SELECTIVE",  "icon": "&#9660;"},
    "avoid":      {"color": EMBER, "bg": "rgba(212,86,42,0.12)",  "label": "AVOID",      "icon": "&#10005;"},
}


# ---------------------------------------------------------------------------
# PA weights by batting order slot (source: MLB historical averages)
# ---------------------------------------------------------------------------

_PA_WEIGHTS = [4.50, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85, 3.75, 3.55]

# wOBA-to-runs conversion: ~1.2 runs per .001 wOBA per PA over a season,
# but for a single game with ~4 PA, the scale is:
# delta_runs = delta_xwoba * PA * wOBA_scale
# where wOBA_scale ~ 1.15 (wOBA to runs conversion factor)
_WOBA_SCALE = 1.15


# ---------------------------------------------------------------------------
# Lineup optimization
# ---------------------------------------------------------------------------

def _optimize_lineup(
    batters: list[dict],
) -> tuple[list[dict], float, float]:
    """Find the optimal batting order given per-batter matchup xwOBA.

    Uses a heuristic that mirrors real MLB lineup construction:
      - Slot 1-2: highest OBP-type batters (xwOBA)
      - Slot 3-4: best power + xwOBA combo
      - Slot 5-9: descending xwOBA

    Parameters
    ----------
    batters : list of dicts with keys:
        batter_id, batter_name, team_abbr, matchup_xwoba, current_order,
        headshot_id, batter_label

    Returns
    -------
    (optimal_order, current_score, optimal_score)
        optimal_order: batters re-ordered with 'optimal_slot' assigned
        current_score: weighted xwOBA of current order
        optimal_score: weighted xwOBA of optimal order
    """
    if len(batters) < 2:
        return batters, 0.0, 0.0

    n = min(len(batters), 9)
    pa_w = _PA_WEIGHTS[:n]

    # Current order score
    current_sorted = sorted(batters, key=lambda b: b["current_order"])[:n]
    current_score = sum(
        b["matchup_xwoba"] * pa_w[i] for i, b in enumerate(current_sorted)
    )

    # Optimal: sort by matchup_xwoba descending, assign to slots with most PA
    # This maximizes the weighted sum (greedy optimal for linear objective)
    by_xwoba = sorted(batters[:n], key=lambda b: b["matchup_xwoba"], reverse=True)
    # Pair best hitters with highest-PA slots
    for i, b in enumerate(by_xwoba):
        b["optimal_slot"] = i + 1

    optimal_score = sum(
        b["matchup_xwoba"] * pa_w[i] for i, b in enumerate(by_xwoba)
    )

    return by_xwoba, current_score, optimal_score


def _render_lineup_optimization_html(
    batters: list[dict],
    bench: list[dict],
    pitcher_name: str,
    opp_abbr: str,
) -> str:
    """Render the recommended batting order card with bench options."""
    if len(batters) < 2:
        return ""

    optimal, _current_score, _optimal_score = _optimize_lineup(batters)

    n = min(len(batters), 9)

    # Build recommended order rows
    rows = ""
    for i, b in enumerate(optimal[:n]):
        xw = b["matchup_xwoba"]
        color = SAGE if xw > 0.325 else (EMBER if xw < 0.305 else SLATE)
        label = b.get("batter_label", "")
        pos = b.get("position", "")
        meta_parts = []
        if pos:
            meta_parts.append(pos)
        if label:
            meta_parts.append(label)
        meta_html = f' <span style="color:var(--tdd-slate);font-size:0.55rem">{esc(", ".join(meta_parts))}</span>' if meta_parts else ""

        rows += (
            '<div style="display:flex;align-items:center;padding:4px 0;'
            'border-bottom:1px solid var(--tdd-dark-border-faint);font-size:0.75rem">'
            f'<span style="color:var(--tdd-gold);width:1.4rem;text-align:right;'
            f'font-family:var(--tdd-font-heading);font-weight:700;font-size:0.8rem">{i+1}</span>'
            f'<span style="padding:0 0.4rem">{headshot_html(b["batter_id"], size=24)}</span>'
            f'<span style="color:var(--tdd-cream);flex:1;font-family:var(--tdd-font-heading);'
            f'font-weight:600">{esc(b["batter_name"])}{meta_html}</span>'
            f'<span style="color:{color};font-family:var(--tdd-font-mono);'
            f'font-weight:700;font-size:0.78rem">.{int(xw*1000):03d}</span>'
            '</div>'
        )

    # Bench players
    bench_html = ""
    if bench:
        bench_rows = ""
        for b in bench:
            xw = b["matchup_xwoba"]
            color = SAGE if xw > 0.325 else (EMBER if xw < 0.305 else SLATE)
            pos = b.get("position", "")
            pos_html = f' <span style="color:var(--tdd-slate);font-size:0.55rem">{esc(pos)}</span>' if pos else ""

            bench_rows += (
                '<div style="display:flex;align-items:center;padding:3px 0;'
                'border-bottom:1px solid var(--tdd-dark-border-faint);font-size:0.72rem;opacity:0.7">'
                f'<span style="color:var(--tdd-slate);width:1.4rem"></span>'
                f'<span style="padding:0 0.4rem">{headshot_html(b["batter_id"], size=22)}</span>'
                f'<span style="color:var(--tdd-cream);flex:1">{esc(b["batter_name"])}{pos_html}</span>'
                f'<span style="color:{color};font-family:var(--tdd-font-mono);'
                f'font-size:0.7rem">.{int(xw*1000):03d}</span>'
                '</div>'
            )

        bench_html = (
            '<div style="border-top:1px dashed var(--tdd-slate);margin-top:6px;'
            'padding-top:4px;opacity:0.7">'
            '<div style="color:var(--tdd-slate);font-size:0.55rem;letter-spacing:1px;'
            'margin-bottom:3px">BENCH</div>'
            f'{bench_rows}'
            '</div>'
        )

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);'
        'border-radius:6px;padding:0.8rem 1rem;margin-bottom:1rem">'
        '<div style="color:var(--tdd-gold);font-size:0.6rem;font-weight:700;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">'
        'Recommended Batting Order</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.6rem;margin-bottom:0.5rem">'
        f'{esc(opp_abbr)} vs {esc(pitcher_name)} -- ranked by matchup xwOBA</div>'
        f'{rows}'
        f'{bench_html}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Bullpen matchup matrix
# ---------------------------------------------------------------------------

def _compute_bullpen_matrix(
    relievers: list[dict],
    batters: list[dict],
    arsenal_df: pd.DataFrame,
    arsenal_by_stand_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
) -> list[dict]:
    """Compute matchup xwOBA for each reliever vs each batter.

    Returns list of dicts: {reliever info + 'matchups': {batter_id: xwoba}}
    """
    results = []
    for rp in relievers:
        pid = rp["pitcher_id"]
        p_hand = rp.get("pitch_hand")
        matchups: dict[int, float] = {}

        for b in batters:
            bid = b["batter_id"]
            h_vul = vuln_df[vuln_df["batter_id"] == bid]
            h_str = str_df[str_df["batter_id"] == bid] if not str_df.empty else pd.DataFrame()
            if h_vul.empty:
                continue

            b_hand = str(h_vul["batter_stand"].iloc[0]) if "batter_stand" in h_vul.columns else None

            # Use platoon arsenal if available
            p_ars = arsenal_df[arsenal_df["pitcher_id"] == pid]
            if b_hand and not arsenal_by_stand_df.empty:
                stand = b_hand.upper()[0] if b_hand else None
                if stand in ("L", "R"):
                    platoon = arsenal_by_stand_df[
                        (arsenal_by_stand_df["pitcher_id"] == pid)
                        & (arsenal_by_stand_df["batter_stand"] == stand)
                    ]
                    if not platoon.empty and len(platoon) >= 2:
                        p_ars = platoon

            if p_ars.empty:
                continue

            edge = compute_matchup_xwoba_edge(
                p_ars, h_vul, h_str,
                pitcher_hand=p_hand, batter_hand=b_hand,
            )
            matchups[bid] = edge["matchup_xwoba"]

        if matchups:
            avg_xw = sum(matchups.values()) / len(matchups)
            results.append({**rp, "matchups": matchups, "avg_xwoba": avg_xw})

    results.sort(key=lambda x: x["avg_xwoba"])
    return results


def _render_bullpen_matrix_html(
    matrix: list[dict],
    batters: list[dict],
) -> str:
    """Render bullpen matchup heatmap as HTML table."""
    if not matrix or not batters:
        return ""

    LG = 0.315

    # Header row with batter last names
    header = '<th style="text-align:left;padding:3px 6px;font-size:0.6rem;color:var(--tdd-gold)">Reliever</th>'
    for b in batters:
        parts = b["batter_name"].split()
        short = parts[-1] if len(parts) > 1 else parts[0]
        # Truncate long names
        if len(short) > 8:
            short = short[:7] + "."
        header += (
            f'<th style="text-align:center;padding:3px 2px;font-size:0.52rem;'
            f'color:var(--tdd-slate);min-width:32px;max-width:42px;'
            f'overflow:hidden;white-space:nowrap">{esc(short)}</th>'
        )
    header += '<th style="text-align:center;padding:3px 4px;font-size:0.55rem;color:var(--tdd-gold)">AVG</th>'

    # Body rows
    rows = ""
    for rp in matrix:
        hand = rp.get("pitch_hand", "")
        hand_str = f' {"L" if hand == "L" else "R"}HP' if hand else ""
        role = rp.get("role", "")
        role_html = f' <span style="color:var(--tdd-gold);font-size:0.5rem;font-weight:700">{esc(role)}</span>' if role else ""

        cells = (
            f'<td style="text-align:left;padding:3px 6px;font-size:0.65rem;'
            f'color:var(--tdd-cream);white-space:nowrap">'
            f'{esc(rp["pitcher_name"])}'
            f'<span style="color:var(--tdd-slate);font-size:0.5rem">{hand_str}</span>'
            f'{role_html}</td>'
        )

        for b in batters:
            bid = b["batter_id"]
            xw = rp["matchups"].get(bid)
            if xw is None:
                cells += '<td style="text-align:center;padding:3px 2px;font-size:0.6rem;color:var(--tdd-slate)">--</td>'
            else:
                # Color scale: green (low xwOBA = pitcher wins) to red (high = hitter wins)
                if xw <= LG - 0.030:
                    bg = "rgba(107,163,142,0.25)"
                    fg = SAGE
                elif xw <= LG - 0.010:
                    bg = "rgba(107,163,142,0.12)"
                    fg = SAGE
                elif xw >= LG + 0.030:
                    bg = "rgba(212,86,42,0.25)"
                    fg = EMBER
                elif xw >= LG + 0.010:
                    bg = "rgba(212,86,42,0.12)"
                    fg = EMBER
                else:
                    bg = "transparent"
                    fg = SLATE

                cells += (
                    f'<td style="text-align:center;padding:3px 2px;font-size:0.6rem;'
                    f'font-family:var(--tdd-font-mono);color:{fg};background:{bg}">'
                    f'.{int(xw*1000):03d}</td>'
                )

        # Average column
        avg = rp["avg_xwoba"]
        avg_color = SAGE if avg < LG - 0.010 else (EMBER if avg > LG + 0.010 else SLATE)
        cells += (
            f'<td style="text-align:center;padding:3px 4px;font-size:0.62rem;'
            f'font-family:var(--tdd-font-mono);font-weight:700;color:{avg_color}">'
            f'.{int(avg*1000):03d}</td>'
        )

        rows += f'<tr style="border-bottom:1px solid var(--tdd-dark-border-faint)">{cells}</tr>'

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);'
        'border-radius:6px;padding:0.8rem;margin-bottom:1rem;overflow-x:auto">'
        '<div style="color:var(--tdd-gold);font-size:0.6rem;font-weight:700;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">'
        'Bullpen Matchup Matrix</div>'
        '<div style="color:var(--tdd-slate);font-size:0.55rem;margin-bottom:6px">'
        'Matchup xwOBA per reliever vs each batter. Green = pitcher advantage, red = hitter advantage.</div>'
        '<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr style="border-bottom:1px solid var(--tdd-dark-border)">{header}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Putaway pitch helpers
# ---------------------------------------------------------------------------

_LOC_LABELS = {
    "up": "up in zone",
    "low": "down in zone",
    "off_plate": "off the plate",
}


def _build_putaway_html(
    pitcher_id: int,
    batter_stand: str | None,
    putaway_df: pd.DataFrame,
) -> str:
    """Build a compact 2-strike putaway summary for a batter's card."""
    if putaway_df.empty or not batter_stand:
        return ""

    stand = batter_stand.upper()[0] if batter_stand else None
    if stand not in ("L", "R"):
        return ""

    rows = putaway_df[
        (putaway_df["pitcher_id"] == pitcher_id)
        & (putaway_df["batter_stand"] == stand)
    ].sort_values("usage_2k_pct", ascending=False)

    if rows.empty:
        return ""

    top = rows.iloc[0]
    pt_name = PITCH_DISPLAY.get(top["pitch_type"], top["pitch_type"])
    usage_pct = top["usage_2k_pct"]
    whiff = top.get("whiff_rate_2k")

    # Primary location
    loc_pcts = {
        "up": top.get("loc_up_pct", 0) or 0,
        "low": top.get("loc_low_pct", 0) or 0,
        "off_plate": top.get("loc_off_plate_pct", 0) or 0,
    }
    best_loc = max(loc_pcts, key=loc_pcts.get)
    best_loc_pct = loc_pcts[best_loc]
    loc_label = _LOC_LABELS.get(best_loc, best_loc)

    # Secondary putaway if exists
    secondary = ""
    if len(rows) >= 2:
        r2 = rows.iloc[1]
        if r2["usage_2k_pct"] >= 0.20:
            pt2 = PITCH_DISPLAY.get(r2["pitch_type"], r2["pitch_type"]).lower()
            secondary = f', {r2["usage_2k_pct"]*100:.0f}% {pt2}'

    whiff_str = f", {whiff*100:.0f}% whiff" if pd.notna(whiff) and whiff > 0 else ""
    hand_label = "LHB" if stand == "L" else "RHB"

    return (
        '<div style="margin-top:6px;padding:6px 8px;'
        'border:1px solid var(--tdd-dark-border);border-radius:4px;'
        'background:rgba(200,169,110,0.06)">'
        '<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="color:var(--tdd-gold);font-size:0.58rem;font-weight:700;'
        f'letter-spacing:1px">2-STRIKE PUTAWAY vs {hand_label}</span>'
        '</div>'
        f'<div style="color:var(--tdd-cream);font-size:0.7rem;margin-top:2px">'
        f'{usage_pct*100:.0f}% {pt_name.lower()}{whiff_str}'
        f'{secondary}'
        f'</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.6rem;margin-top:1px">'
        f'{best_loc_pct*100:.0f}% located {loc_label}'
        f'</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def _render_pitch_plan_row(plan: dict, batter_label: str = "") -> str:
    """Render one pitch-type row in the attack plan."""
    style = _APPROACH_STYLE.get(plan["approach"], _APPROACH_STYLE["neutral"])
    xw = plan["matchup_xwoba"]
    whiff = plan["matchup_whiff"]
    usage = plan["usage"]
    velo = plan.get("velo")
    velo_str = f"{velo:.0f}" if velo else "--"

    # Usage label with batter hand context
    usage_label = f"{usage*100:.0f}%"
    if batter_label:
        usage_label += f" vs {batter_label}"

    # xwOBA bar
    bar_pct = min(xw / 0.500 * 100, 100)

    return (
        f'<div class="gp-pitch-row" style="border-left:3px solid {style["color"]};'
        f'background:{style["bg"]};padding:0.5rem 0.7rem;margin-bottom:4px;border-radius:0 4px 4px 0">'
        # Top line: pitch name + approach badge + xwOBA
        '<div style="display:flex;justify-content:space-between;align-items:center">'
        '<div style="display:flex;align-items:center;gap:0.5rem">'
        f'<span style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);'
        f'font-weight:700;font-size:0.82rem">{esc(plan["pitch_name"])}</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.65rem">{usage_label}</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.65rem">{velo_str} mph</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:0.5rem">'
        f'<span style="color:{style["color"]};font-size:0.6rem;font-weight:700;'
        f'letter-spacing:1px">{style["label"]}</span>'
        f'<span style="color:{style["color"]};font-family:var(--tdd-font-mono);'
        f'font-weight:700;font-size:0.8rem">.{int(xw*1000):03d}</span>'
        '</div>'
        '</div>'
        # Recommendation text
        f'<div style="color:var(--tdd-cream);font-size:0.72rem;margin-top:3px;'
        f'opacity:0.85">{esc(plan["recommendation"])}</div>'
        # Stats row
        '<div style="display:flex;gap:1rem;margin-top:4px">'
        f'<span style="color:var(--tdd-slate);font-size:0.6rem">Whiff {whiff*100:.0f}%</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.6rem">xwOBA con '
        f'.{int(plan["matchup_xwoba_contact"]*1000):03d}</span>'
        '</div>'
        '</div>'
    )


def _render_batter_attack_card(
    batter_id: int,
    batter_name: str,
    team_abbr: str,
    batting_order: int,
    plan: dict,
    edge: dict,
    putaway_html: str = "",
) -> str:
    """Render a full attack plan card for one batter."""
    xw = edge["matchup_xwoba"]
    adv = edge["advantage"]
    if adv == "hitter":
        edge_color = SAGE
        edge_label = "Hitter Edge"
    elif adv == "pitcher":
        edge_color = EMBER
        edge_label = "Pitcher Edge"
    else:
        edge_color = SLATE
        edge_label = "Even"

    platoon_html = ""
    if plan["platoon"] == "favorable":
        platoon_html = (
            '<span style="color:var(--tdd-sage);font-size:0.6rem;'
            'border:1px solid var(--tdd-sage);border-radius:3px;padding:1px 5px;'
            'margin-left:0.4rem">PLATOON +</span>'
        )
    elif plan["platoon"] == "unfavorable":
        platoon_html = (
            '<span style="color:var(--tdd-ember);font-size:0.6rem;'
            'border:1px solid var(--tdd-ember);border-radius:3px;padding:1px 5px;'
            'margin-left:0.4rem">PLATOON -</span>'
        )

    # Batter hand context label
    batter_label = plan.get("batter_label", "")
    hand_html = ""
    if batter_label:
        hand_html = (
            f'<span style="color:var(--tdd-slate);font-size:0.58rem;'
            f'margin-left:0.3rem">({batter_label})</span>'
        )

    # Pitch plan rows with usage threshold separator
    _bl = plan.get("batter_label", "") or ""
    pitch_rows = ""
    shown_separator = False
    for p in plan["pitch_plans"]:
        if not shown_separator and p.get("tier") in ("secondary", "rare"):
            shown_separator = True
            pitch_rows += (
                '<div style="border-top:1px dashed var(--tdd-slate);margin:6px 0;'
                'opacity:0.3"></div>'
                '<div style="color:var(--tdd-slate);font-size:0.55rem;'
                'letter-spacing:1px;margin-bottom:4px;opacity:0.5">'
                'SECONDARY</div>'
            )
        pitch_rows += _render_pitch_plan_row(p, batter_label=_bl)

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);'
        'border-radius:6px;padding:1rem;margin-bottom:0.8rem">'
        # Header: order + headshot + name + edge badge
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem">'
        f'<span style="color:var(--tdd-gold);font-family:var(--tdd-font-heading);'
        f'font-weight:700;font-size:1.1rem;min-width:1.4rem">{batting_order}</span>'
        f'{headshot_html(batter_id, size=40)}'
        '<div style="flex:1">'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap">'
        f'<span style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);'
        f'font-weight:700;font-size:0.95rem">{esc(batter_name)}</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.7rem;margin-left:0.4rem">{esc(team_abbr)}</span>'
        f'{hand_html}'
        f'{platoon_html}'
        '</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.65rem;margin-top:1px">'
        f'{esc(plan["summary"])}</div>'
        '</div>'
        f'<div style="text-align:right">'
        f'<div style="color:{edge_color};font-family:var(--tdd-font-mono);'
        f'font-weight:700;font-size:1rem">.{int(xw*1000):03d}</div>'
        f'<div style="color:{edge_color};font-size:0.55rem;letter-spacing:0.5px">{edge_label}</div>'
        '</div>'
        '</div>'
        # Pitch plans
        f'{pitch_rows}'
        # Putaway
        f'{putaway_html}'
        '</div>'
    )


def _render_game_prep_header(game: pd.Series, side: str) -> str:
    """Render the header for one side's game prep."""
    pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
    pid_raw = game.get(f"{side}_pitcher_id")
    side_abbr = game.get(f"{side}_abbr", "?")
    opp_side = "home" if side == "away" else "away"
    opp_abbr = game.get(f"{opp_side}_abbr", "?")

    return (
        '<div style="margin-bottom:1rem">'
        '<div style="color:var(--tdd-gold);font-size:0.6rem;font-weight:700;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">'
        f'{esc(opp_abbr)} Hitters vs {esc(pitcher_name)}</div>'
        '<div style="color:var(--tdd-slate);font-size:0.65rem">'
        f'Pitch-by-pitch approach plan for each batter in the {esc(opp_abbr)} lineup '
        f'against {esc(side_abbr)} starter {esc(pitcher_name)}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def page_game_prep() -> None:
    """Game Prep Report page."""

    # Header
    st.markdown(
        '<div class="pl-page">'
        '<header class="pl-page-head">'
        '<div>'
        '<div class="pl-page-eyebrow">The Data Diamond</div>'
        '<h1 class="pl-page-title">Game Prep Report</h1>'
        '<p class="pl-page-sub">'
        'Pitcher attack plans, matchup edges, and approach recommendations. '
        'What a hitting coach would tell the lineup before first pitch.'
        '</p>'
        '</div>'
        '</header>',
        unsafe_allow_html=True,
    )

    # Load schedule
    schedule = load_todays_games()
    if schedule.empty:
        st.markdown(
            '<div class="pl-empty">No games scheduled today.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Game selector
    game_labels = []
    game_pks = []
    for _, g in schedule.iterrows():
        away = g.get("away_abbr", "?")
        home = g.get("home_abbr", "?")
        t = format_game_time(g.get("game_datetime_utc"), fallback=g.get("game_time", ""))
        game_labels.append(f"{away} @ {home} - {t}")
        game_pks.append(int(g["game_pk"]))

    sel_idx = st.selectbox(
        "Select Game", range(len(game_labels)),
        format_func=lambda i: game_labels[i],
        key="gp_game_sel", label_visibility="collapsed",
    )
    gpk = game_pks[sel_idx]
    game = schedule[schedule["game_pk"] == gpk].iloc[0]

    # Load data
    lineups = load_todays_lineups()
    lineups = backfill_missing_lineups(schedule, lineups)
    arsenal_df = load_pitcher_arsenal()
    arsenal_by_stand_df = load_pitcher_arsenal_by_stand()
    putaway_df = load_pitcher_putaway()
    vuln_df = load_hitter_vulnerability(career=True)
    str_df = load_hitter_strength(career=True)

    if arsenal_df.empty or vuln_df.empty:
        st.markdown(
            '<div class="pl-empty">Matchup data not available.</div></div>',
            unsafe_allow_html=True,
        )
        return

    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty else pd.DataFrame()
    if game_lu.empty:
        st.markdown(
            '<div class="pl-empty">Lineups not yet available for this game.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Render both sides
    col_away, col_home = st.columns(2)

    for side, col in [("away", col_away), ("home", col_home)]:
        opp_side = "home" if side == "away" else "away"
        pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
        pid_raw = game.get(f"{side}_pitcher_id")
        pid = int(pid_raw) if pd.notna(pid_raw) else None
        side_abbr = game.get(f"{side}_abbr", "?")
        opp_abbr = game.get(f"{opp_side}_abbr", "?")
        opp_team_id = game.get(f"{opp_side}_team_id")

        if not pid:
            with col:
                st.markdown(
                    f'<div style="color:var(--tdd-slate);font-size:0.8rem;padding:1rem">'
                    f'{esc(opp_abbr)} lineup: Starting pitcher TBD</div>',
                    unsafe_allow_html=True,
                )
            continue

        p_ars = arsenal_df[arsenal_df["pitcher_id"] == pid]
        if p_ars.empty:
            with col:
                st.markdown(
                    f'<div style="color:var(--tdd-slate);font-size:0.8rem;padding:1rem">'
                    f'No arsenal data for {esc(pitcher_name)}</div>',
                    unsafe_allow_html=True,
                )
            continue

        p_hand = str(p_ars["pitch_hand"].iloc[0]) if "pitch_hand" in p_ars.columns else None

        # Get opposing lineup
        opp_lu = game_lu[game_lu["team_id"] == opp_team_id].sort_values("batting_order")

        with col:
            st.markdown(
                _render_game_prep_header(game, side),
                unsafe_allow_html=True,
            )

            if opp_lu.empty:
                st.markdown(
                    '<div style="color:var(--tdd-slate);font-size:0.8rem">'
                    'Lineup not available</div>',
                    unsafe_allow_html=True,
                )
                continue

            # --- First pass: compute matchup data for all batters ---
            batter_data: list[dict] = []
            for _, batter_row in opp_lu.iterrows():
                bid = int(batter_row.get("batter_id", 0))
                if not bid:
                    continue
                bname = batter_row.get("batter_name", str(bid))
                b_team = batter_row.get("team_abbr", opp_abbr)
                order = int(batter_row.get("batting_order", 0))

                h_vul = vuln_df[vuln_df["batter_id"] == bid]
                h_str = str_df[str_df["batter_id"] == bid] if not str_df.empty else pd.DataFrame()

                b_hand = str(h_vul["batter_stand"].iloc[0]) if not h_vul.empty and "batter_stand" in h_vul.columns else None

                # Use platoon-specific arsenal when available
                p_ars_plan = p_ars  # fallback to overall
                if b_hand and not arsenal_by_stand_df.empty:
                    stand = b_hand.upper()[0] if b_hand else None
                    if stand and stand in ("L", "R"):
                        platoon_ars = arsenal_by_stand_df[
                            (arsenal_by_stand_df["pitcher_id"] == pid)
                            & (arsenal_by_stand_df["batter_stand"] == stand)
                        ]
                        if not platoon_ars.empty and len(platoon_ars) >= 2:
                            p_ars_plan = platoon_ars

                plan = None
                edge = None
                if not h_vul.empty:
                    plan = build_attack_plan(
                        p_ars_plan, h_vul, h_str,
                        pitcher_hand=p_hand, batter_hand=b_hand,
                    )
                    edge = compute_matchup_xwoba_edge(
                        p_ars_plan, h_vul, h_str,
                        pitcher_hand=p_hand, batter_hand=b_hand,
                    )

                batter_data.append({
                    "batter_id": bid,
                    "batter_name": bname,
                    "team_abbr": b_team,
                    "current_order": order,
                    "matchup_xwoba": edge["matchup_xwoba"] if edge else 0.315,
                    "plan": plan,
                    "edge": edge,
                    "batter_label": plan.get("batter_label", "") if plan else "",
                    "batter_hand": b_hand,
                })

            # --- Split starters (order 1-9) vs bench (10+) ---
            # Build position lookup from roster
            roster_df = load_roster()
            pos_lookup: dict[int, str] = {}
            if not roster_df.empty:
                for _, r in roster_df.iterrows():
                    pos_lookup[int(r["player_id"])] = r.get("primary_position", "")

            # Add position to all batters
            for b in batter_data:
                b["position"] = pos_lookup.get(b["batter_id"], "")

            starters = [b for b in batter_data if b["current_order"] <= 9]
            bench_from_lineup = [b for b in batter_data if b["current_order"] > 9]

            bench_data: list[dict] = []
            for b in bench_from_lineup:
                bench_data.append({
                    "batter_id": b["batter_id"],
                    "batter_name": b["batter_name"],
                    "matchup_xwoba": b["matchup_xwoba"],
                    "position": pos_lookup.get(b["batter_id"], ""),
                })
            bench_data.sort(key=lambda x: x["matchup_xwoba"], reverse=True)

            # --- Lineup Optimization card ---
            scoreable = [b for b in starters if b["edge"] is not None]
            if len(scoreable) >= 4:
                opt_html = _render_lineup_optimization_html(
                    scoreable, bench_data, pitcher_name, opp_abbr,
                )
                if opt_html:
                    st.markdown(opt_html, unsafe_allow_html=True)

            # --- Attack plan cards (starters only) ---
            for b in starters:
                bid = b["batter_id"]
                order = b["current_order"]
                if b["plan"] is None:
                    st.markdown(
                        f'<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);'
                        f'border-radius:6px;padding:0.8rem 1rem;margin-bottom:0.8rem;'
                        f'display:flex;align-items:center;gap:0.6rem">'
                        f'<span style="color:var(--tdd-gold);font-family:var(--tdd-font-heading);'
                        f'font-weight:700;font-size:1.1rem;min-width:1.4rem">{order}</span>'
                        f'{headshot_html(bid, size=40)}'
                        f'<span style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);'
                        f'font-weight:700;font-size:0.95rem">{esc(b["batter_name"])}</span>'
                        f'<span style="color:var(--tdd-slate);font-size:0.7rem">No matchup data</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    pa_html = _build_putaway_html(
                        pid, b.get("batter_hand"), putaway_df,
                    )
                    card_html = _render_batter_attack_card(
                        bid, b["batter_name"], b["team_abbr"],
                        order, b["plan"], b["edge"],
                        putaway_html=pa_html,
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

            # --- Bullpen Matchup Matrix ---
            from services.data_loader import load_reliever_rankings
            rr = load_reliever_rankings()
            if not roster_df.empty and not rr.empty:
                # Get relievers for the pitcher's team (same side)
                side_rps = roster_df[
                    (roster_df["team_abbr"] == side_abbr)
                    & (roster_df["primary_position"] == "RP")
                    & (roster_df["roster_status"] == "active")
                ]
                rp_list: list[dict] = []
                for _, r in side_rps.iterrows():
                    rp_pid = int(r["player_id"])
                    rr_row = rr[rr["pitcher_id"] == rp_pid]
                    rp_hand = str(rr_row["pitch_hand"].iloc[0]) if not rr_row.empty and "pitch_hand" in rr_row.columns and pd.notna(rr_row["pitch_hand"].iloc[0]) else None
                    rp_role = str(rr_row["role"].iloc[0]) if not rr_row.empty and "role" in rr_row.columns else ""
                    rp_list.append({
                        "pitcher_id": rp_pid,
                        "pitcher_name": r["player_name"],
                        "pitch_hand": rp_hand,
                        "role": rp_role,
                    })

                if rp_list and starters:
                    matrix = _compute_bullpen_matrix(
                        rp_list, starters,
                        arsenal_df, arsenal_by_stand_df,
                        vuln_df, str_df,
                    )
                    if matrix:
                        bp_html = _render_bullpen_matrix_html(matrix, starters)
                        if bp_html:
                            st.markdown(bp_html, unsafe_allow_html=True)

    # Close page wrapper
    st.markdown('</div>', unsafe_allow_html=True)
