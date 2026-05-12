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
    load_prospect_comps_batters, load_prospect_comps_pitchers, load_milb_priors,
    load_pitcher_game_logs,
    load_pitcher_advanced_stats,
    load_pitcher_location_grid,
)
from components.scouting import (
    build_attack_plan, compute_matchup_xwoba_edge, PITCH_DISPLAY,
    build_comp_proxy_data, build_pitcher_comp_arsenal,
    assess_walk_strategy,
    get_pitcher_recent_form,
)
from components.headshot import headshot_html
from components.team_logo import team_logo_html
from lib.zone_charts import plot_pitcher_location_compact
from utils.chart_embed import fig_to_base64
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

    # Column header
    rows = (
        '<div style="display:flex;align-items:center;padding:3px 0;margin-bottom:2px;'
        'border-bottom:1px solid var(--tdd-dark-border)">'
        '<span style="width:1.6rem"></span>'
        '<span style="padding:0 0.5rem;width:28px"></span>'
        '<span style="color:var(--tdd-slate);flex:1;font-size:0.7rem;letter-spacing:0.5px"></span>'
        '<span style="color:var(--tdd-slate);font-size:0.7rem;width:3.5rem;text-align:center">K%</span>'
        '<span style="color:var(--tdd-slate);font-size:0.7rem;width:3.5rem;text-align:center">BB%</span>'
        '<span style="color:var(--tdd-slate);font-size:0.7rem;width:3.5rem;text-align:center">wOBA</span>'
        '<span style="color:var(--tdd-slate);font-size:0.7rem;width:4.5rem;text-align:center">Edge</span>'
        '<span style="color:var(--tdd-slate);font-size:0.7rem;width:10rem;text-align:left">Approach</span>'
        '<span style="color:var(--tdd-slate);font-size:0.7rem;width:4.5rem;text-align:right">Matchup</span>'
        '</div>'
    )

    # Build recommended order rows
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
        meta_html = f' <span style="color:var(--tdd-slate);font-size:0.7rem">{esc(", ".join(meta_parts))}</span>' if meta_parts else ""

        # Platoon badge
        platoon_html = ""
        plan = b.get("plan")
        if plan and plan.get("platoon") == "favorable":
            platoon_html = '<span style="color:var(--tdd-sage);font-size:0.6rem;border:1px solid var(--tdd-sage);border-radius:2px;padding:0 3px;margin-left:0.3rem">+</span>'
        elif plan and plan.get("platoon") == "unfavorable":
            platoon_html = '<span style="color:var(--tdd-ember);font-size:0.6rem;border:1px solid var(--tdd-ember);border-radius:2px;padding:0 3px;margin-left:0.3rem">&minus;</span>'

        # Edge label
        edge = b.get("edge")
        if edge and edge.get("advantage") == "hitter":
            edge_html = f'<span style="color:var(--tdd-sage);font-size:0.72rem">Hitter</span>'
        elif edge and edge.get("advantage") == "pitcher":
            edge_html = f'<span style="color:var(--tdd-ember);font-size:0.72rem">Pitcher</span>'
        else:
            edge_html = f'<span style="color:var(--tdd-slate);font-size:0.72rem">Even</span>'

        # Hunt/avoid summary
        approach_html = ""
        if plan:
            hunt_picks = [p["pitch_name"] for p in plan.get("pitch_plans", [])
                          if p.get("approach") == "hunt" and p.get("tier") == "primary"]
            avoid_picks = [p["pitch_name"] for p in plan.get("pitch_plans", [])
                           if p.get("approach") == "avoid" and p.get("tier") == "primary"]
            parts = []
            if hunt_picks:
                parts.append(f'<span style="color:var(--tdd-sage)">Hunt {", ".join(hunt_picks[:2])}</span>')
            if avoid_picks:
                parts.append(f'<span style="color:var(--tdd-ember)">Avoid {", ".join(avoid_picks[:1])}</span>')
            if parts:
                approach_html = f'<span style="font-size:0.72rem">{" · ".join(parts)}</span>'

        # Bayesian projections
        bproj = b.get("projections", {})
        k_pct = bproj.get("projected_k_rate")
        bb_pct = bproj.get("projected_bb_rate")
        woba = bproj.get("projected_woba")

        k_color = SAGE if k_pct and k_pct < 0.20 else (EMBER if k_pct and k_pct > 0.28 else SLATE)
        bb_color = SAGE if bb_pct and bb_pct > 0.10 else (EMBER if bb_pct and bb_pct < 0.06 else SLATE)
        woba_color = SAGE if woba and woba > 0.340 else (EMBER if woba and woba < 0.300 else SLATE)

        k_html = f'<span style="color:{k_color};font-family:var(--tdd-font-mono);font-size:0.8rem">{k_pct*100:.0f}</span>' if k_pct else '<span style="color:var(--tdd-slate);font-size:0.75rem">--</span>'
        bb_html = f'<span style="color:{bb_color};font-family:var(--tdd-font-mono);font-size:0.8rem">{bb_pct*100:.0f}</span>' if bb_pct else '<span style="color:var(--tdd-slate);font-size:0.75rem">--</span>'
        woba_html = f'<span style="color:{woba_color};font-family:var(--tdd-font-mono);font-size:0.8rem">.{int(woba*1000):03d}</span>' if woba else '<span style="color:var(--tdd-slate);font-size:0.75rem">--</span>'

        rows += (
            '<div style="display:flex;align-items:center;padding:5px 0;'
            'border-bottom:1px solid var(--tdd-dark-border-faint);font-size:0.9rem">'
            f'<span style="color:var(--tdd-gold);width:1.6rem;text-align:right;'
            f'font-family:var(--tdd-font-heading);font-weight:700;font-size:1rem">{i+1}</span>'
            f'<span style="padding:0 0.5rem">{headshot_html(b["batter_id"], size=28)}</span>'
            f'<span style="color:var(--tdd-cream);flex:1;font-family:var(--tdd-font-heading);'
            f'font-weight:600">{esc(b["batter_name"])}{meta_html}{platoon_html}</span>'
            f'<span style="width:3.5rem;text-align:center">{k_html}</span>'
            f'<span style="width:3.5rem;text-align:center">{bb_html}</span>'
            f'<span style="width:3.5rem;text-align:center">{woba_html}</span>'
            f'<span style="width:4.5rem;text-align:center">{edge_html}</span>'
            f'<span style="width:10rem;text-align:left">{approach_html}</span>'
            f'<span style="color:{color};font-family:var(--tdd-font-mono);'
            f'font-weight:700;font-size:0.95rem;width:4.5rem;text-align:right">.{int(xw*1000):03d}</span>'
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
        '<div style="color:var(--tdd-gold);font-size:0.75rem;font-weight:700;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">'
        'Recommended Batting Order</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.75rem;margin-bottom:0.5rem">'
        f'{esc(opp_abbr)} vs {esc(pitcher_name)} · '
        f'K%/BB%/wOBA = Bayesian projections · Matchup = pitch-type matchup quality (avg .315)</div>'
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
    comps_df: pd.DataFrame | None = None,
    milb_priors_df: pd.DataFrame | None = None,
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

            # Comp fallback for MiLB callups
            if h_vul.empty and comps_df is not None and not comps_df.empty:
                h_vul, h_str, _ = build_comp_proxy_data(
                    bid, comps_df, vuln_df, str_df, milb_priors_df,
                )

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


_APPEAR_THRESHOLD = 0.30  # ~49 projected games = likely to appear


def _render_bullpen_row(rp: dict, batters: list[dict], opacity: float = 1.0) -> str:
    """Render one reliever row in the bullpen matrix."""
    LG = 0.315
    hand = rp.get("pitch_hand", "")
    hand_str = f' {"L" if hand == "L" else "R"}HP' if hand else ""
    role = rp.get("role", "")
    role_html = f' <span style="color:var(--tdd-gold);font-size:0.5rem;font-weight:700">{esc(role)}</span>' if role else ""
    appear = rp.get("appear_pct", 0)
    appear_html = f' <span style="color:var(--tdd-slate);font-size:0.45rem">{appear*100:.0f}%</span>' if appear > 0 else ""

    cells = (
        f'<td style="text-align:left;padding:3px 6px;font-size:0.65rem;'
        f'color:var(--tdd-cream);white-space:nowrap;opacity:{opacity}">'
        f'{esc(rp["pitcher_name"])}'
        f'<span style="color:var(--tdd-slate);font-size:0.5rem">{hand_str}</span>'
        f'{role_html}{appear_html}</td>'
    )

    for b in batters:
        bid = b["batter_id"]
        xw = rp["matchups"].get(bid)
        if xw is None:
            cells += f'<td style="text-align:center;padding:3px 2px;font-size:0.6rem;color:var(--tdd-slate);opacity:{opacity}">--</td>'
        else:
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
                f'font-family:var(--tdd-font-mono);color:{fg};background:{bg};opacity:{opacity}">'
                f'.{int(xw*1000):03d}</td>'
            )

    avg = rp["avg_xwoba"]
    avg_color = SAGE if avg < LG - 0.010 else (EMBER if avg > LG + 0.010 else SLATE)
    cells += (
        f'<td style="text-align:center;padding:3px 4px;font-size:0.62rem;'
        f'font-family:var(--tdd-font-mono);font-weight:700;color:{avg_color};opacity:{opacity}">'
        f'.{int(avg*1000):03d}</td>'
    )

    return f'<tr style="border-bottom:1px solid var(--tdd-dark-border-faint)">{cells}</tr>'


def _render_bullpen_matrix_html(
    matrix: list[dict],
    batters: list[dict],
) -> str:
    """Render bullpen matchup heatmap with probable/unlikely split."""
    if not matrix or not batters:
        return ""

    # Header row
    header = '<th style="text-align:left;padding:3px 6px;font-size:0.6rem;color:var(--tdd-gold)">Reliever</th>'
    for b in batters:
        parts = b["batter_name"].split()
        short = parts[-1] if len(parts) > 1 else parts[0]
        if len(short) > 8:
            short = short[:7] + "."
        header += (
            f'<th style="text-align:center;padding:3px 2px;font-size:0.52rem;'
            f'color:var(--tdd-slate);min-width:32px;max-width:42px;'
            f'overflow:hidden;white-space:nowrap">{esc(short)}</th>'
        )
    header += '<th style="text-align:center;padding:3px 4px;font-size:0.55rem;color:var(--tdd-gold)">AVG</th>'

    # Split probable vs unlikely
    probable = [rp for rp in matrix if rp.get("appear_pct", 0) >= _APPEAR_THRESHOLD]
    unlikely = [rp for rp in matrix if rp.get("appear_pct", 0) < _APPEAR_THRESHOLD]

    rows = ""
    for rp in probable:
        rows += _render_bullpen_row(rp, batters, opacity=1.0)

    if unlikely:
        n_cols = len(batters) + 2
        rows += (
            f'<tr><td colspan="{n_cols}" style="padding:4px 6px;border-top:1px dashed var(--tdd-slate);'
            f'border-bottom:none">'
            f'<span style="color:var(--tdd-slate);font-size:0.5rem;letter-spacing:1px;opacity:0.5">'
            f'LESS LIKELY</span></td></tr>'
        )
        for rp in unlikely:
            rows += _render_bullpen_row(rp, batters, opacity=0.6)

    probable_label = f"{len(probable)} probable" if probable else ""
    unlikely_label = f", {len(unlikely)} depth" if unlikely else ""

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);'
        'border-radius:6px;padding:0.8rem;margin-bottom:1rem;overflow-x:auto">'
        '<div style="color:var(--tdd-gold);font-size:0.6rem;font-weight:700;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">'
        'Bullpen Matchup Matrix</div>'
        '<div style="color:var(--tdd-slate);font-size:0.55rem;margin-bottom:6px">'
        f'Matchup xwOBA per reliever vs each batter ({probable_label}{unlikely_label}). '
        'Green = pitcher advantage, red = hitter advantage.</div>'
        '<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr style="border-bottom:1px solid var(--tdd-dark-border)">{header}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Pinch-hit opportunities
# ---------------------------------------------------------------------------

def _find_pinch_hit_opportunities(
    starters: list[dict],
    bench: list[dict],
    min_xwoba_gain: float = 0.025,
) -> list[dict]:
    """Find bench bats that significantly outperform a starter in this matchup.

    Returns list of opportunities sorted by xwOBA gain, each with:
        starter, bench_player, xwoba_gain, reason
    """
    if not bench:
        return []

    opps: list[dict] = []
    for b in bench:
        b_xw = b["matchup_xwoba"]
        b_pos = b.get("position", "")
        b_name = b["batter_name"]
        b_id = b["batter_id"]

        for s in starters:
            s_xw = s["matchup_xwoba"]
            gain = b_xw - s_xw
            if gain < min_xwoba_gain:
                continue

            # Build reason
            reason_parts: list[str] = []
            s_hand = s.get("batter_label", "")
            b_hand = ""
            # Infer bench player hand from vuln data if available
            for bb in bench:
                if bb["batter_id"] == b_id:
                    # We don't have hand stored on bench - use position as proxy
                    break

            reason_parts.append(f"+{gain*1000:.0f} pts xwOBA")

            opps.append({
                "starter_name": s["batter_name"],
                "starter_id": s["batter_id"],
                "starter_order": s["current_order"],
                "starter_xwoba": s_xw,
                "starter_pos": s.get("position", ""),
                "bench_name": b_name,
                "bench_id": b_id,
                "bench_xwoba": b_xw,
                "bench_pos": b_pos,
                "xwoba_gain": gain,
            })

    # Sort by gain descending, deduplicate bench players (keep best opportunity)
    opps.sort(key=lambda x: x["xwoba_gain"], reverse=True)
    seen_bench: set[int] = set()
    deduped: list[dict] = []
    for o in opps:
        if o["bench_id"] not in seen_bench:
            deduped.append(o)
            seen_bench.add(o["bench_id"])
    return deduped


def _render_pinch_hit_html(opps: list[dict], pitcher_name: str) -> str:
    """Render pinch-hit opportunities card."""
    if not opps:
        return ""

    rows = ""
    for o in opps:
        gain_pts = o["xwoba_gain"] * 1000
        s_xw = o["starter_xwoba"]
        b_xw = o["bench_xwoba"]
        s_color = SAGE if s_xw > 0.325 else (EMBER if s_xw < 0.305 else SLATE)
        b_color = SAGE if b_xw > 0.325 else (EMBER if b_xw < 0.305 else SLATE)

        rows += (
            '<div style="display:flex;align-items:center;gap:6px;padding:5px 0;'
            'border-bottom:1px solid var(--tdd-dark-border-faint);font-size:0.72rem">'
            # Bench player (gains)
            f'{headshot_html(o["bench_id"], size=24)}'
            f'<div style="flex:1;min-width:0">'
            f'<span style="color:var(--tdd-cream);font-weight:600">{esc(o["bench_name"])}</span>'
            f' <span style="color:var(--tdd-slate);font-size:0.55rem">{esc(o["bench_pos"])}</span>'
            f'<span style="color:{b_color};font-family:var(--tdd-font-mono);'
            f'font-size:0.7rem;margin-left:0.4rem">.{int(b_xw*1000):03d}</span>'
            '</div>'
            # Arrow
            '<div style="color:var(--tdd-slate);font-size:0.6rem;padding:0 4px">'
            'for'
            '</div>'
            # Starter (replaced)
            f'{headshot_html(o["starter_id"], size=24)}'
            f'<div style="flex:1;min-width:0">'
            f'<span style="color:var(--tdd-cream);opacity:0.7">{esc(o["starter_name"])}</span>'
            f' <span style="color:var(--tdd-slate);font-size:0.55rem">#{o["starter_order"]}</span>'
            f'<span style="color:{s_color};font-family:var(--tdd-font-mono);'
            f'font-size:0.7rem;margin-left:0.4rem">.{int(s_xw*1000):03d}</span>'
            '</div>'
            # Gain badge
            f'<span style="color:var(--tdd-sage);font-family:var(--tdd-font-mono);'
            f'font-weight:700;font-size:0.72rem;min-width:3rem;text-align:right">'
            f'+{gain_pts:.0f} pts</span>'
            '</div>'
        )

    return (
        '<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);'
        'border-radius:6px;padding:0.8rem 1rem;margin-bottom:1rem">'
        '<div style="color:var(--tdd-gold);font-size:0.6rem;font-weight:700;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">'
        'Pinch-Hit Opportunities</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.55rem;margin-bottom:6px">'
        f'Bench bats with better matchup xwOBA vs {esc(pitcher_name)} than the starter they replace.</div>'
        f'{rows}'
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
        f'font-weight:700;font-size:0.95rem">{esc(plan["pitch_name"])}</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.78rem">{usage_label}</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.78rem">{velo_str} mph</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:0.5rem">'
        f'<span style="color:{style["color"]};font-size:0.72rem;font-weight:700;'
        f'letter-spacing:1px">{style["label"]}</span>'
        f'<span style="color:{style["color"]};font-family:var(--tdd-font-mono);'
        f'font-weight:700;font-size:0.95rem">.{int(xw*1000):03d}</span>'
        '</div>'
        '</div>'
        # Recommendation text
        f'<div style="color:var(--tdd-cream);font-size:0.85rem;margin-top:3px;'
        f'opacity:0.85">{esc(plan["recommendation"])}</div>'
        # Stats row
        '<div style="display:flex;gap:1rem;margin-top:4px">'
        f'<span style="color:var(--tdd-slate);font-size:0.72rem">Whiff {whiff*100:.0f}%</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.72rem">xwOBA con '
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
    comp_info: dict | None = None,
    zone_chart_b64: str | None = None,
) -> str:
    """Render a full attack plan card for one batter.

    When *comp_info* is provided the card is annotated with a MiLB badge,
    comp names, and (optionally) MiLB rate projections.
    """
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

    # MiLB comp badge
    milb_badge_html = ""
    if comp_info:
        milb_badge_html = (
            '<span style="color:var(--tdd-gold);font-size:0.55rem;font-weight:700;'
            'border:1px solid var(--tdd-gold);border-radius:3px;padding:1px 5px;'
            'margin-left:0.4rem;letter-spacing:0.5px">MiLB COMP</span>'
        )

    # Batter hand context label
    batter_label = plan.get("batter_label", "")
    hand_html = ""
    if batter_label:
        hand_html = (
            f'<span style="color:var(--tdd-slate);font-size:0.58rem;'
            f'margin-left:0.3rem">({batter_label})</span>'
        )

    # Comp + MiLB rates subtitle (below the summary line)
    comp_subtitle_html = ""
    if comp_info:
        names = comp_info.get("comp_names", [])
        sims = comp_info.get("comp_similarities", [])
        comp_parts = [
            f'{esc(n)} ({s:.0%})' for n, s in zip(names, sims)
        ]
        comp_line = ", ".join(comp_parts)

        # MiLB rate projections
        rates = comp_info.get("milb_rates", {})
        rate_parts: list[str] = []
        _RATE_LABELS = {"k_rate": "K%", "bb_rate": "BB%", "hr_rate": "HR%"}
        for stat_key, label in _RATE_LABELS.items():
            r = rates.get(stat_key)
            if r:
                mean_pct = r["mean"] * 100
                p10_pct = r["p10"] * 100
                p90_pct = r["p90"] * 100
                rate_parts.append(
                    f'{label} {mean_pct:.1f} '
                    f'<span style="opacity:0.5">[{p10_pct:.0f}-{p90_pct:.0f}]</span>'
                )
        rates_line = (
            f'<span style="margin-left:0.6rem">{" &nbsp; ".join(rate_parts)}</span>'
            if rate_parts else ""
        )

        comp_subtitle_html = (
            f'<div style="margin-top:2px;font-size:0.6rem;color:var(--tdd-slate);'
            f'opacity:0.8">'
            f'Based on: {comp_line}'
            f'{rates_line}'
            f'</div>'
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

    # Card border: gold accent for comp-based cards
    border_style = (
        'border:1px solid var(--tdd-gold);border-left:3px solid var(--tdd-gold)'
        if comp_info
        else 'border:1px solid var(--tdd-dark-border)'
    )

    # Body content: pitch plans + putaway
    body_content = f'{pitch_rows}{putaway_html}'

    if zone_chart_b64:
        # Two-column layout: plans on left, zone chart on right
        body_html = (
            '<div style="display:flex;gap:0.8rem;margin-top:0.2rem">'
            f'<div style="flex:2;min-width:0">{body_content}</div>'
            f'<div style="flex:0 0 280px;padding-top:0.3rem">'
            f'<div style="color:var(--tdd-slate);font-size:0.65rem;text-align:center;'
            f'margin-bottom:2px">Hunt pitch location</div>'
            f'<img src="{zone_chart_b64}" style="width:100%;border-radius:4px;'
            f'" />'
            f'</div>'
            '</div>'
        )
    else:
        body_html = body_content

    return (
        f'<div style="background:var(--tdd-dark-card);{border_style};'
        f'border-radius:6px;padding:1rem;margin-bottom:0.8rem">'
        # Header: order + headshot + name + edge badge
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem">'
        f'<span style="color:var(--tdd-gold);font-family:var(--tdd-font-heading);'
        f'font-weight:700;font-size:1.4rem;min-width:1.6rem">{batting_order}</span>'
        f'{headshot_html(batter_id, size=48)}'
        '<div style="flex:1">'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap">'
        f'<span style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);'
        f'font-weight:700;font-size:1.15rem">{esc(batter_name)}</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.8rem;margin-left:0.4rem">{esc(team_abbr)}</span>'
        f'{hand_html}'
        f'{platoon_html}'
        f'{milb_badge_html}'
        '</div>'
        f'<div style="color:var(--tdd-slate);font-size:0.78rem;margin-top:2px">'
        f'{esc(plan["summary"])}</div>'
        f'{comp_subtitle_html}'
        '</div>'
        f'<div style="text-align:right">'
        f'<div style="color:{edge_color};font-family:var(--tdd-font-mono);'
        f'font-weight:700;font-size:1.2rem">.{int(xw*1000):03d}</div>'
        f'<div style="color:{edge_color};font-size:0.65rem;letter-spacing:0.5px">{edge_label}</div>'
        '</div>'
        '</div>'
        # Body (with or without zone chart)
        f'{body_html}'
        '</div>'
    )


def _render_pitcher_overview(
    pitcher_name: str,
    pitch_hand: str | None,
    form: dict | None,
    walk_strat: dict | None,
    zone_b64: str | None,
) -> str:
    """Render the opposing starter overview card with form + zone chart."""
    hand_label = f"{'LHP' if pitch_hand == 'L' else 'RHP'}" if pitch_hand else ""

    # Form badge
    form_html = ""
    if form:
        era_color = SAGE if form["era"] < 3.50 else (EMBER if form["era"] > 4.50 else SLATE)
        form_html = (
            f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.5rem;'
            f'font-size:0.9rem;font-family:var(--tdd-font-mono)">'
            f'<span style="color:var(--tdd-slate)">Last {form["n_starts"]} starts:</span>'
            f'<span style="color:var(--tdd-cream);font-weight:600">{form["avg_ip"]} IP</span>'
            f'<span style="color:var(--tdd-cream);font-weight:600">{form["k_per_9"]} K/9</span>'
            f'<span style="color:var(--tdd-cream);font-weight:600">{form["bb_per_9"]} BB/9</span>'
            f'<span style="color:{era_color};font-weight:600">{form["era"]:.2f} ERA</span>'
            f'<span style="color:var(--tdd-slate)">{int(form["avg_pitches"])} pitch avg</span>'
            f'</div>'
        )

    # Walk strategy callout
    walk_html = ""
    if walk_strat:
        walk_html = (
            f'<div style="background:rgba(200,169,110,0.06);border:1px solid rgba(200,169,110,0.25);'
            f'border-radius:4px;padding:0.4rem 0.6rem;margin-top:0.5rem;'
            f'font-size:0.85rem;color:var(--tdd-gold)">'
            f'&#9888; {esc(walk_strat["note"])}'
            f'</div>'
        )

    # Zone chart image
    zone_html = ""
    if zone_b64:
        zone_html = (
            f'<div style="margin-top:0.8rem">'
            f'<div style="color:var(--tdd-slate);font-size:0.78rem;margin-bottom:0.3rem">'
            f'Pitch location density — where he throws each pitch type (darker = more frequent)</div>'
            f'<img src="{zone_b64}" style="width:100%;border-radius:4px;'
            f'" />'
            f'</div>'
        )

    return (
        f'<div style="background:var(--tdd-dark-card);border:1px solid var(--tdd-dark-border);'
        f'border-radius:6px;padding:1rem;margin-bottom:1rem">'
        f'<div style="display:flex;align-items:center;gap:0.6rem">'
        f'<div style="flex:1">'
        f'<div style="color:var(--tdd-gold);font-family:var(--tdd-font-heading);'
        f'font-weight:700;font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;'
        f'margin-bottom:0.3rem">OPPOSING STARTER</div>'
        f'<span style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);'
        f'font-weight:700;font-size:1.5rem">{esc(pitcher_name)}</span>'
        f'<span style="color:var(--tdd-slate);font-size:0.9rem;margin-left:0.5rem">'
        f'{hand_label}</span>'
        f'</div>'
        f'</div>'
        f'{form_html}'
        f'{walk_html}'
        f'{zone_html}'
        f'</div>'
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
        'Recommended lineups, pitcher attack plans, and bullpen matchup strategy. '
        'Who to start and how to approach each at-bat.'
        '</p>'
        '</div>'
        '</header>',
        unsafe_allow_html=True,
    )

    # Date selector: today + next 2 days
    from datetime import date
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    today = et_now.date()
    date_options = [today + timedelta(days=d) for d in range(3)]
    date_labels = []
    for d in date_options:
        if d == today:
            date_labels.append(f"{d.strftime('%a %b %d')} (Today)")
        else:
            date_labels.append(d.strftime("%a %b %d"))

    dcol1, dcol2 = st.columns([1, 3])
    with dcol1:
        sel_date_idx = st.selectbox(
            "Date", range(len(date_options)),
            format_func=lambda i: date_labels[i],
            key="gp_date_sel", label_visibility="collapsed",
        )
    sel_date = date_options[sel_date_idx]

    # Load schedule for selected date
    if sel_date == today:
        schedule = load_todays_games()
    else:
        schedule = fetch_live_schedule(sel_date.isoformat())

    if schedule.empty:
        st.markdown(
            f'<div class="pl-empty">No games scheduled for {sel_date.strftime("%b %d")}.</div></div>',
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
        ap = g.get("away_pitcher_name") or "TBD"
        hp = g.get("home_pitcher_name") or "TBD"
        game_labels.append(f"{away} ({ap}) @ {home} ({hp}) - {t}")
        game_pks.append(int(g["game_pk"]))

    with dcol2:
        sel_idx = st.selectbox(
            "Select Game", range(len(game_labels)),
            format_func=lambda i: game_labels[i],
            key="gp_game_sel", label_visibility="collapsed",
        )
    gpk = game_pks[sel_idx]
    game = schedule[schedule["game_pk"] == gpk].iloc[0]

    # Load lineups -- build from roster for all position players
    if sel_date == today:
        lineups = load_todays_lineups()
        lineups = backfill_missing_lineups(schedule, lineups)
    else:
        lineups = pd.DataFrame()
        lineups = backfill_missing_lineups(schedule, lineups)

    # If lineups are still empty (future games), build from roster
    if lineups.empty:
        roster_all = load_roster()
        if not roster_all.empty:
            rows = []
            for _, g in schedule.iterrows():
                _gpk = int(g["game_pk"])
                for side in ("away", "home"):
                    team_abbr = g.get(f"{side}_abbr", "")
                    team_id = g.get(f"{side}_team_id")
                    team_pos = roster_all[
                        (roster_all["team_abbr"] == team_abbr)
                        & (roster_all["roster_status"] == "active")
                        & (~roster_all["primary_position"].isin(["SP", "RP"]))
                    ]
                    for i, (_, r) in enumerate(team_pos.iterrows(), 1):
                        rows.append({
                            "game_pk": _gpk,
                            "team_id": team_id,
                            "team_abbr": team_abbr,
                            "batter_id": int(r["player_id"]),
                            "batter_name": r["player_name"],
                            "batting_order": i,
                            "lineup_source": "roster",
                        })
            if rows:
                lineups = pd.DataFrame(rows)
    arsenal_df = load_pitcher_arsenal()
    arsenal_by_stand_df = load_pitcher_arsenal_by_stand()
    putaway_df = load_pitcher_putaway()
    vuln_df = load_hitter_vulnerability(career=True)
    str_df = load_hitter_strength(career=True)

    # MiLB comp data for callups with no MLB pitch-type data
    comps_df = load_prospect_comps_batters()
    pitcher_comps_df = load_prospect_comps_pitchers()
    milb_priors_df = load_milb_priors()

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

    # Load additional data for v2 features
    location_df = load_pitcher_location_grid()
    game_logs_df = load_pitcher_game_logs()
    proj_df = load_projections("pitcher")
    hitter_proj_df = load_projections("hitter")
    adv_df = load_pitcher_advanced_stats()

    # Build tab labels: "{team} Hitters vs {opposing pitcher}"
    home_abbr = game.get("home_abbr", "?")
    away_abbr = game.get("away_abbr", "?")
    home_sp = game.get("home_pitcher_name") or "TBD"
    away_sp = game.get("away_pitcher_name") or "TBD"
    tab_labels = [
        f"{home_abbr} Hitters vs {away_sp}",
        f"{away_abbr} Hitters vs {home_sp}",
    ]
    tab_first, tab_second = st.tabs(tab_labels)

    # side="away" means away pitcher vs home hitters → first tab
    # side="home" means home pitcher vs away hitters → second tab
    for side, tab in [("away", tab_first), ("home", tab_second)]:
        opp_side = "home" if side == "away" else "away"
        pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
        pid_raw = game.get(f"{side}_pitcher_id")
        pid = int(pid_raw) if pd.notna(pid_raw) else None
        side_abbr = game.get(f"{side}_abbr", "?")
        opp_abbr = game.get(f"{opp_side}_abbr", "?")
        opp_team_id = game.get(f"{opp_side}_team_id")

        if not pid:
            with tab:
                st.markdown(
                    f'<div style="color:var(--tdd-slate);font-size:0.8rem;padding:1rem">'
                    f'{esc(opp_abbr)} lineup: Starting pitcher TBD</div>',
                    unsafe_allow_html=True,
                )
            continue

        p_ars = arsenal_df[arsenal_df["pitcher_id"] == pid]
        pitcher_comp_info = None
        if p_ars.empty and not pitcher_comps_df.empty:
            # MiLB callup pitcher: synthesize arsenal from comps
            p_ars, pitcher_comp_info = build_pitcher_comp_arsenal(
                pid, pitcher_comps_df, arsenal_df, milb_priors_df,
            )
        if p_ars.empty:
            with tab:
                st.markdown(
                    f'<div style="color:var(--tdd-slate);font-size:0.8rem;padding:1rem">'
                    f'No arsenal data for {esc(pitcher_name)}</div>',
                    unsafe_allow_html=True,
                )
            continue

        p_hand = str(p_ars["pitch_hand"].iloc[0]) if "pitch_hand" in p_ars.columns else None

        # --- Pitcher Overview (v2) ---
        form = get_pitcher_recent_form(pid, game_logs_df, n_starts=5)
        walk_strat = assess_walk_strategy(pid, proj_df, adv_df)

        # Zone heatmap for pitcher overview
        zone_overview_b64 = None
        if not location_df.empty:
            p_loc = location_df[location_df["pitcher_id"] == pid]
            if not p_loc.empty:
                try:
                    fig = plot_pitcher_location_compact(
                        p_loc, pitcher_name=pitcher_name, batter_stand=None,
                    )
                    zone_overview_b64 = fig_to_base64(fig, dpi=120)
                except Exception:
                    pass

        # Get opposing lineup
        opp_lu = game_lu[game_lu["team_id"] == opp_team_id].sort_values("batting_order")

        with tab:
            st.markdown(
                _render_game_prep_header(game, side),
                unsafe_allow_html=True,
            )

            # Pitcher overview with form + zone chart
            overview_html = _render_pitcher_overview(
                pitcher_name, p_hand, form, walk_strat, zone_overview_b64,
            )
            st.markdown(overview_html, unsafe_allow_html=True)

            # MiLB comp banner for pitcher
            if pitcher_comp_info:
                comp_names = pitcher_comp_info.get("comp_names", [])
                comp_sims = pitcher_comp_info.get("comp_similarities", [])
                comp_parts = [f'{esc(n)} ({s:.0%})' for n, s in zip(comp_names, comp_sims)]
                rates = pitcher_comp_info.get("milb_rates", {})
                rate_parts: list[str] = []
                for sk, sl in [("k_rate", "K%"), ("bb_rate", "BB%"), ("hr_per_bf", "HR%")]:
                    r = rates.get(sk)
                    if r:
                        rate_parts.append(
                            f'{sl} {r["mean"]*100:.1f} '
                            f'<span style="opacity:0.5">[{r["p10"]*100:.0f}-{r["p90"]*100:.0f}]</span>'
                        )
                st.markdown(
                    '<div style="background:rgba(200,169,110,0.08);border:1px solid var(--tdd-gold);'
                    'border-radius:4px;padding:0.5rem 0.7rem;margin-bottom:0.8rem;font-size:0.65rem">'
                    '<span style="color:var(--tdd-gold);font-weight:700;letter-spacing:0.5px">'
                    'MiLB COMP ARSENAL</span>'
                    f'<span style="color:var(--tdd-slate);margin-left:0.5rem">'
                    f'Based on: {", ".join(comp_parts)}</span>'
                    + (f'<div style="color:var(--tdd-cream);margin-top:3px">'
                       f'{" &nbsp; ".join(rate_parts)}</div>' if rate_parts else '')
                    + '</div>',
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
                b_comp_info = None
                if not h_vul.empty:
                    plan = build_attack_plan(
                        p_ars_plan, h_vul, h_str,
                        pitcher_hand=p_hand, batter_hand=b_hand,
                    )
                    edge = compute_matchup_xwoba_edge(
                        p_ars_plan, h_vul, h_str,
                        pitcher_hand=p_hand, batter_hand=b_hand,
                    )
                elif not comps_df.empty:
                    # MiLB callup: build synthetic profile from comps
                    comp_vuln, comp_str, b_comp_info = build_comp_proxy_data(
                        bid, comps_df, vuln_df, str_df, milb_priors_df,
                    )
                    if not comp_vuln.empty:
                        plan = build_attack_plan(
                            p_ars_plan, comp_vuln, comp_str,
                            pitcher_hand=p_hand, batter_hand=b_hand,
                        )
                        edge = compute_matchup_xwoba_edge(
                            p_ars_plan, comp_vuln, comp_str,
                            pitcher_hand=p_hand, batter_hand=b_hand,
                        )

                # Look up Bayesian projections for this batter
                b_proj: dict = {}
                if not hitter_proj_df.empty:
                    hp_row = hitter_proj_df[hitter_proj_df["batter_id"] == bid]
                    if not hp_row.empty:
                        for col in ("projected_k_rate", "projected_bb_rate", "projected_woba"):
                            val = hp_row[col].iloc[0] if col in hp_row.columns else None
                            if pd.notna(val):
                                b_proj[col] = float(val)

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
                    "comp_info": b_comp_info,
                    "projections": b_proj,
                })

            # --- Rank all position players by matchup xwOBA ---
            roster_df = load_roster()
            pos_lookup: dict[int, str] = {}
            if not roster_df.empty:
                for _, r in roster_df.iterrows():
                    pos_lookup[int(r["player_id"])] = r.get("primary_position", "")

            for b in batter_data:
                b["position"] = pos_lookup.get(b["batter_id"], "")

            # Always rank by matchup xwOBA: top 9 = recommended starters
            all_scoreable = sorted(
                [b for b in batter_data if b["edge"] is not None],
                key=lambda x: x["matchup_xwoba"], reverse=True,
            )
            starters = all_scoreable[:9]
            bench_pool = all_scoreable[9:]
            for i, b in enumerate(starters):
                b["current_order"] = i + 1

            bench_data: list[dict] = []
            for b in bench_pool:
                bench_data.append({
                    "batter_id": b["batter_id"],
                    "batter_name": b["batter_name"],
                    "matchup_xwoba": b["matchup_xwoba"],
                    "position": pos_lookup.get(b["batter_id"], ""),
                })
            bench_data.sort(key=lambda x: x["matchup_xwoba"], reverse=True)

            # --- Recommended Lineup card ---
            scoreable = [b for b in starters if b["edge"] is not None]
            if len(scoreable) >= 4:
                opt_html = _render_lineup_optimization_html(
                    scoreable, bench_data, pitcher_name, opp_abbr,
                )
                if opt_html:
                    st.markdown(opt_html, unsafe_allow_html=True)

            # --- Pinch-Hit Opportunities (placeholder for official lineups) ---
            if False and bench_data and scoreable:  # disabled until official lineups
                ph_opps = _find_pinch_hit_opportunities(scoreable, bench_data)
                if ph_opps:
                    ph_html = _render_pinch_hit_html(ph_opps, pitcher_name)
                    if ph_html:
                        st.markdown(ph_html, unsafe_allow_html=True)

            # --- Attack plan cards (starters only) ---
            for i, b in enumerate(starters):
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
                    # Zone chart for top 6 batters
                    zone_b64 = None
                    if i < 6 and not location_df.empty and b.get("batter_hand"):
                        p_loc = location_df[location_df["pitcher_id"] == pid]
                        if not p_loc.empty:
                            # Pick best primary hunt/aggressive pitch for zone chart
                            hunt_pt = None
                            for pp in b["plan"].get("pitch_plans", []):
                                if pp.get("tier") == "primary" and pp.get("approach") in ("hunt", "aggressive"):
                                    hunt_pt = pp["pitch_type"]
                                    break
                            if hunt_pt is None:
                                hunt_pt = b["plan"].get("hunt_pitch")
                            if hunt_pt:
                                try:
                                    bstand = b["batter_hand"].upper()[0] if b["batter_hand"] else None
                                    fig = plot_pitcher_location_compact(
                                        p_loc,
                                        pitch_types=[hunt_pt],
                                        batter_stand=bstand,
                                        figsize=(3.5, 3.5),
                                    )
                                    zone_b64 = fig_to_base64(fig, dpi=120)
                                except Exception:
                                    pass

                    pa_html = _build_putaway_html(
                        pid, b.get("batter_hand"), putaway_df,
                    )
                    card_html = _render_batter_attack_card(
                        bid, b["batter_name"], b["team_abbr"],
                        order, b["plan"], b["edge"],
                        putaway_html=pa_html,
                        comp_info=b.get("comp_info"),
                        zone_chart_b64=zone_b64,
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
                    # Appearance probability: projected games / 162
                    proj_games = float(rr_row["total_games_mean"].iloc[0]) if not rr_row.empty and "total_games_mean" in rr_row.columns and pd.notna(rr_row["total_games_mean"].iloc[0]) else 0.0
                    appear_pct = proj_games / 162.0
                    rp_list.append({
                        "pitcher_id": rp_pid,
                        "pitcher_name": r["player_name"],
                        "pitch_hand": rp_hand,
                        "role": rp_role,
                        "appear_pct": appear_pct,
                    })

                if rp_list and starters:
                    matrix = _compute_bullpen_matrix(
                        rp_list, starters,
                        arsenal_df, arsenal_by_stand_df,
                        vuln_df, str_df,
                        comps_df=comps_df,
                        milb_priors_df=milb_priors_df,
                    )
                    if matrix:
                        bp_html = _render_bullpen_matrix_html(matrix, starters)
                        if bp_html:
                            st.markdown(bp_html, unsafe_allow_html=True)

    # Close page wrapper
    st.markdown('</div>', unsafe_allow_html=True)
