"""Game Prep Report -- MLB-style pregame preparation for coaching staff."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, PITCH_TYPE_COLORS, PITCH_DISPLAY as PITCH_DISPLAY_MAP
from services.data_loader import (
    load_todays_games, load_todays_lineups,
    load_pitcher_arsenal, load_pitcher_arsenal_by_stand,
    load_pitcher_putaway,
    load_hitter_vulnerability, load_hitter_strength,
    load_projections, load_pitcher_archetypes, load_hitter_archetypes,
    load_roster,
    backfill_missing_lineups,
    fetch_live_schedule,
    load_prospect_comps_batters, load_prospect_comps_pitchers, load_milb_priors,
    load_pitcher_game_logs,
    load_pitcher_advanced_stats,
    load_pitcher_location_grid,
    load_game_props,
    load_pitcher_platoon_bb,
)
from components.scouting import (
    build_attack_plan, compute_matchup_xwoba_edge, PITCH_DISPLAY,
    build_comp_proxy_data, build_pitcher_comp_arsenal,
    assess_walk_strategy,
    get_pitcher_recent_form,
    get_pitcher_platoon_bb,
)
from components.headshot import headshot_html
from components.team_logo import team_logo_html
from lib.zone_charts import plot_pitcher_location_compact, plot_pitcher_location_split
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
# Arsenal mix bar + trend badge helpers
# ---------------------------------------------------------------------------

def _render_arsenal_bar(arsenal_df: pd.DataFrame, pitcher_id: int) -> str:
    """Render a horizontal stacked bar showing pitch type usage percentages."""
    p_ars = arsenal_df[arsenal_df["pitcher_id"] == pitcher_id]
    if p_ars.empty:
        return ""

    # Sort by usage descending
    usage_col = "usage_pct" if "usage_pct" in p_ars.columns else "usage"
    p_ars = p_ars.sort_values(usage_col, ascending=False)

    segments = ""
    key_items = ""
    for _, row in p_ars.iterrows():
        pt = row["pitch_type"]
        usage = row.get(usage_col, 0)
        if usage < 0.01:
            continue
        color = PITCH_TYPE_COLORS.get(pt, SLATE)
        pt_label = PITCH_DISPLAY_MAP.get(pt, pt)
        velo = row.get("avg_velo") if "avg_velo" in row.index else row.get("velo")
        velo_str = f" {velo:.0f}" if pd.notna(velo) and velo > 0 else ""
        width_pct = usage * 100

        # Segment in the bar
        show_label = pt if usage >= 0.10 else ""
        segments += (
            f'<div class="seg" style="width:{width_pct:.1f}%;background:{color}">'
            f'{show_label}</div>'
        )

        # Key item below
        key_items += (
            f'<span class="item">'
            f'<span class="dot" style="background:{color}"></span>'
            f'{esc(pt_label)} '
            f'<span class="v">{usage*100:.0f}%</span>'
            f'<span style="opacity:0.7">{velo_str}</span>'
            f'</span>'
        )

    if not segments:
        return ""

    return (
        '<div style="margin-top:0.5rem">'
        '<div style="color:var(--tdd-slate);font-size:0.55rem;letter-spacing:1px;'
        'text-transform:uppercase;margin-bottom:4px">Arsenal</div>'
        f'<div class="gp-arsenal">{segments}</div>'
        f'<div class="gp-arsenal-key">{key_items}</div>'
        '</div>'
    )


def _get_trend_badge(form: dict | None) -> str:
    """Return a trend badge (Hot/Cold/Steady) based on recent form."""
    if not form:
        return ""
    era = form.get("era", 4.00)
    k9 = form.get("k_per_9", 8.0)
    # Hot: low ERA + decent K rate; Cold: high ERA; Steady: middle
    if era < 3.00 and k9 >= 8.0:
        return '<span class="gp-trend-tag up">&#9650; Hot</span>'
    elif era > 5.00:
        return '<span class="gp-trend-tag down">&#9660; Cold</span>'
    else:
        return '<span class="gp-trend-tag flat">&#9644; Steady</span>'


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
    """Find the optimal batting order using MLB-style slot assignment.

    Assigns batters to lineup slots based on role-appropriate profiles:
      - Slot 1 (Leadoff): OBP-oriented — high BB%, low K%, good wOBA
      - Slot 2 (Best hitter): highest projected wOBA + matchup quality
      - Slots 3-4 (Power): best HR rate + production combo
      - Slots 5-9: remaining batters sorted by matchup xwOBA descending

    Parameters
    ----------
    batters : list of dicts with keys including:
        batter_id, batter_name, team_abbr, matchup_xwoba, current_order,
        headshot_id, batter_label, projections (optional dict with
        projected_k_rate, projected_bb_rate, projected_woba, projected_hr_rate)

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

    pool = list(batters[:n])

    # -- Extract projection metrics with league-average fallbacks -----------
    lg_k = 0.220
    lg_bb = 0.085
    lg_woba = 0.315
    lg_hr = 0.030

    def _proj(b: dict, key: str, fallback: float) -> float:
        proj = b.get("projections") or {}
        val = proj.get(key)
        return val if val is not None else fallback

    k_rates = np.array([_proj(b, "projected_k_rate", lg_k) for b in pool])
    bb_rates = np.array([_proj(b, "projected_bb_rate", lg_bb) for b in pool])
    wobas = np.array([_proj(b, "projected_woba", lg_woba) for b in pool])
    hr_rates = np.array([_proj(b, "projected_hr_rate", lg_hr) for b in pool])
    matchups = np.array([b["matchup_xwoba"] for b in pool])

    # -- Z-score each metric (guard against zero-variance) ------------------
    def _zscore(arr: np.ndarray) -> np.ndarray:
        s = arr.std()
        if s < 1e-9:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / s

    k_z = _zscore(k_rates)
    bb_z = _zscore(bb_rates)
    woba_z = _zscore(wobas)
    hr_z = _zscore(hr_rates)
    match_z = _zscore(matchups)

    # -- Slot 1 (Leadoff): OBP + contact - power (table-setter, not slugger) -
    leadoff_scores = (
        0.35 * bb_z + 0.25 * woba_z + 0.25 * (-k_z) + 0.10 * match_z
        - 0.30 * hr_z  # penalize power hitters — they belong in 3-4 hole
    )
    slot1_idx = int(np.argmax(leadoff_scores))

    optimal: list[dict] = []
    optimal.append(pool[slot1_idx])
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[slot1_idx] = False

    # -- Slot 2 (Best hitter): wOBA-dominant + matchup + contact ------------
    slot2_scores = 0.5 * woba_z + 0.3 * match_z + 0.2 * (1 - k_z)
    slot2_scores[~remaining_mask] = -np.inf
    slot2_idx = int(np.argmax(slot2_scores))
    optimal.append(pool[slot2_idx])
    remaining_mask[slot2_idx] = False

    # -- Slots 3-4 (Power): HR rate + production + matchup ------------------
    power_scores = 0.4 * hr_z + 0.3 * woba_z + 0.3 * match_z
    for _ in range(min(2, int(remaining_mask.sum()))):
        power_scores[~remaining_mask] = -np.inf
        idx = int(np.argmax(power_scores))
        optimal.append(pool[idx])
        remaining_mask[idx] = False

    # -- Slots 5-9: remaining batters by matchup xwOBA descending -----------
    rest = [
        (matchups[i], pool[i])
        for i in range(n)
        if remaining_mask[i]
    ]
    rest.sort(key=lambda x: x[0], reverse=True)
    optimal.extend(b for _, b in rest)

    # Assign slot numbers
    for i, b in enumerate(optimal):
        b["optimal_slot"] = i + 1

    optimal_score = sum(
        b["matchup_xwoba"] * pa_w[i] for i, b in enumerate(optimal)
    )

    return optimal, current_score, optimal_score


def _render_lineup_optimization_html(
    batters: list[dict],
    bench: list[dict],
    pitcher_name: str,
    opp_abbr: str,
    is_official: bool = False,
) -> str:
    """Render the batting order card using CSS classes for responsive layout."""
    if len(batters) < 2:
        return ""

    if is_official:
        optimal = sorted(batters, key=lambda b: b["current_order"])
    else:
        optimal, _current_score, _optimal_score = _optimize_lineup(batters)

    n = min(len(batters), 9)

    # Column group headers + column headers (hidden on mobile via CSS)
    rows = (
        '<div class="gp-lu-groups">'
        '<span style="min-width:2rem"></span>'
        '<span style="padding:0 0.5rem;min-width:36px"></span>'
        '<span style="width:14rem;min-width:10rem"></span>'
        '<span class="grp-proj">BAYESIAN PROJECTIONS</span>'
        '<span class="grp-sim">GAME SIMULATION</span>'
        '<span style="flex:1"></span>'
        '<span style="flex:1.6"></span>'
        '<span class="grp-xw">xwOBA</span>'
        '</div>'
        '<div class="gp-lu-hdr" style="border-bottom:1px solid var(--tdd-dark-border);padding:4px 0;margin-bottom:3px">'
        '<span style="min-width:2rem"></span>'
        '<span style="padding:0 0.5rem;min-width:36px"></span>'
        '<span style="width:14rem;min-width:10rem"></span>'
        '<span class="sc proj-bg">K%</span>'
        '<span class="sc proj-bg">BB%</span>'
        '<span class="sc proj-bg">wOBA</span>'
        '<span class="sc sim-bg">H</span>'
        '<span class="sc sim-bg">K</span>'
        '<span class="sc sim-bg">TB</span>'
        '<span class="sc">Edge</span>'
        '<span style="flex:1.6;text-align:left;padding:0 0.3rem">Approach</span>'
        '<span class="sc">Matchup</span>'
        '</div>'
    )

    def _render_lu_row(b: dict, idx: int, is_bench: bool = False) -> str:
        xw = b["matchup_xwoba"]
        color = SAGE if xw > 0.325 else (EMBER if xw < 0.305 else SLATE)
        pos = b.get("position", "")
        label = b.get("batter_label", "")
        meta_parts = [p for p in [pos, label] if p]
        meta_html = f' <span style="color:var(--tdd-slate);font-size:0.78rem">{esc(", ".join(meta_parts))}</span>' if meta_parts else ""

        # Platoon badge
        platoon_html = ""
        plan = b.get("plan")
        if plan and plan.get("platoon") == "favorable":
            platoon_html = '<span class="platoon-badge fav" style="font-size:0.65rem;border:1px solid;border-radius:2px;padding:0 4px;margin-left:0.3rem">+</span>'
        elif plan and plan.get("platoon") == "unfavorable":
            platoon_html = '<span class="platoon-badge unfav" style="font-size:0.65rem;border:1px solid;border-radius:2px;padding:0 4px;margin-left:0.3rem">&minus;</span>'

        # Optimization flag
        opt_flag_html = ""
        if is_official and not is_bench:
            opt_rank = b.get("optimal_rank", idx + 1)
            slot = idx + 1
            if opt_rank <= 9 and slot <= 9 and abs(opt_rank - slot) >= 3:
                oflag_color = SAGE if opt_rank < slot else EMBER
                opt_flag_html = (
                    f'<span style="color:{oflag_color};font-size:0.55rem;'
                    f'border:1px solid {oflag_color};border-radius:2px;padding:0 4px;'
                    f'margin-left:0.3rem" title="Our model ranks #{opt_rank}">#{opt_rank}</span>'
                )

        # Edge label
        edge = b.get("edge")
        if edge and edge.get("advantage") == "hitter":
            edge_html = f'<span style="color:{SAGE}">Hitter</span>'
        elif edge and edge.get("advantage") == "pitcher":
            edge_html = f'<span style="color:{EMBER}">Pitcher</span>'
        else:
            edge_html = f'<span style="color:{SLATE}">Even</span>'

        # Approach summary (desktop only)
        approach_html = ""
        if plan and not is_bench:
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
                approach_html = " &middot; ".join(parts)

        # Bayesian projections
        bproj = b.get("projections", {})
        _ci_s = "font-size:0.62rem;opacity:0.55"

        def _proj_cell(val, lo, hi, fmt_fn, color_fn):
            if not val:
                return '<span class="sc">--</span>'
            c = color_fn(val)
            v_str = fmt_fn(val)
            ci = ""
            if lo and hi:
                ci = f'<br><span class="ci" style="color:{c}">[{fmt_fn(lo)}-{fmt_fn(hi)}]</span>'
            return f'<span class="sc"><span style="color:{c};font-weight:600;font-size:0.92rem">{v_str}</span>{ci}</span>'

        k_pct = bproj.get("projected_k_rate")
        bb_pct = bproj.get("projected_bb_rate")
        woba = bproj.get("projected_woba")

        k_cell = _proj_cell(
            k_pct, bproj.get("projected_k_rate_2_5"), bproj.get("projected_k_rate_97_5"),
            lambda v: f"{v*100:.0f}",
            lambda v: SAGE if v < 0.20 else (EMBER if v > 0.28 else SLATE),
        )
        bb_cell = _proj_cell(
            bb_pct, bproj.get("projected_bb_rate_2_5"), bproj.get("projected_bb_rate_97_5"),
            lambda v: f"{v*100:.0f}",
            lambda v: SAGE if v > 0.10 else (EMBER if v < 0.06 else SLATE),
        )
        woba_cell = _proj_cell(
            woba, bproj.get("projected_woba_2_5"), bproj.get("projected_woba_97_5"),
            lambda v: f".{int(v*1000):03d}",
            lambda v: SAGE if v > 0.340 else (EMBER if v < 0.300 else SLATE),
        )

        # Game sim cells
        sim = b.get("sim_stats", {})
        sim_cells = ""
        for skey, good_t, bad_t, high_good in [("H", 1.0, 0.8, True), ("K", 0.8, 1.0, False), ("TB", 1.5, 1.0, True)]:
            sv = sim.get(skey)
            if sv and sv.get("expected") is not None:
                exp = sv["expected"]
                p1 = sv.get("p_1plus")
                sc = (SAGE if exp >= good_t else (EMBER if exp < bad_t else SLATE)) if high_good else (SAGE if exp < good_t else (EMBER if exp >= bad_t else SLATE))
                p1_str = f'<br><span class="p1" style="color:{sc}">{p1*100:.0f}% 1+</span>' if p1 is not None else ""
                sim_cells += f'<span class="sc"><span style="color:{sc};font-weight:600;font-size:0.92rem">{exp:.1f}</span>{p1_str}</span>'
            else:
                sim_cells += '<span class="sc">--</span>'

        order_str = str(idx + 1) if not is_bench else ""
        row_cls = "gp-lu-row bench" if is_bench else "gp-lu-row"
        hs_size = 30 if is_bench else 36

        return (
            f'<div class="{row_cls}">'
            f'<span class="order">{order_str}</span>'
            f'<span class="hs">{headshot_html(b["batter_id"], size=hs_size)}</span>'
            f'<span class="who">{esc(b["batter_name"])}{meta_html}{platoon_html}{opt_flag_html}</span>'
            f'{k_cell}{bb_cell}{woba_cell}'
            f'{sim_cells}'
            f'<span class="edge-col">{edge_html}</span>'
            f'<span class="approach-col">{approach_html}</span>'
            f'<span class="xw-col" style="color:{color}">.{int(xw*1000):03d}</span>'
            '</div>'
        )

    for i, b in enumerate(optimal[:n]):
        rows += _render_lu_row(b, i)

    # Bench
    bench_html = ""
    if bench:
        bench_rows = ""
        for b in bench:
            bench_rows += _render_lu_row(b, 0, is_bench=True)
        bench_html = (
            '<div class="gp-bench-divider">'
            '<span>Bench</span><span class="line"></span>'
            '</div>'
            f'{bench_rows}'
        )

    title = "Posted Lineup" if is_official else "Recommended Batting Order"
    subtitle = (
        f'{esc(opp_abbr)} vs {esc(pitcher_name)} '
        f'{"(official)" if is_official else "-- ranked by matchup xwOBA"}. '
        f'Matchup xwOBA = batted ball quality vs this pitcher (lg avg .315).'
    )

    return (
        '<div class="gp-card">'
        f'<div class="gp-card-eyebrow">{title}</div>'
        f'<div class="gp-card-sub">{subtitle}</div>'
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


def _render_bullpen_row(rp: dict, batters: list[dict], is_depth: bool = False) -> str:
    """Render one reliever row in the bullpen matrix."""
    LG = 0.315
    hand = rp.get("pitch_hand", "")
    hand_str = f'{"L" if hand == "L" else "R"}HP' if hand else ""
    role = rp.get("role", "")
    appear = rp.get("appear_pct", 0)
    depth_cls = " depth" if is_depth else ""

    cells = (
        f'<td class="rp-col">'
        f'<span class="nm">{esc(rp["pitcher_name"])}</span>'
        f'<span class="rp-meta">'
        f'<span>{hand_str}</span>'
    )
    if role:
        cells += f'<span class="role">{esc(role)}</span>'
    if appear > 0:
        cells += f'<span>{appear*100:.0f}%</span>'
    cells += '</span></td>'

    for b in batters:
        bid = b["batter_id"]
        xw = rp["matchups"].get(bid)
        if xw is None:
            cells += '<td>--</td>'
        else:
            if xw <= LG - 0.030:
                cell_cls = "cell-good-deep"
            elif xw <= LG - 0.010:
                cell_cls = "cell-good-light"
            elif xw >= LG + 0.030:
                cell_cls = "cell-bad-deep"
            elif xw >= LG + 0.010:
                cell_cls = "cell-bad-light"
            else:
                cell_cls = ""

            cells += (
                f'<td class="{cell_cls}" style="font-family:var(--tdd-font-body)">'
                f'.{int(xw*1000):03d}</td>'
            )

    avg = rp["avg_xwoba"]
    avg_cls = ""
    if avg < LG - 0.010:
        avg_cls = "cell-good-light"
    elif avg > LG + 0.010:
        avg_cls = "cell-bad-light"
    cells += (
        f'<td class="avg {avg_cls}" style="font-family:var(--tdd-font-body)">'
        f'.{int(avg*1000):03d}</td>'
    )

    return f'<tr class="gp-bp-row{depth_cls}">{cells}</tr>'


def _render_bullpen_matrix_html(
    matrix: list[dict],
    batters: list[dict],
) -> str:
    """Render bullpen matchup heatmap with probable/unlikely split."""
    if not matrix or not batters:
        return ""

    # Header row with sticky first column
    header = '<th class="rp-col">Reliever</th>'
    for i, b in enumerate(batters):
        parts = b["batter_name"].split()
        short = parts[-1] if len(parts) > 1 else parts[0]
        if len(short) > 6:
            short = short[:6] + "."
        header += f'<th title="{esc_attr(b["batter_name"])}">{i+1}.{esc(short)}</th>'
    header += '<th style="color:var(--tdd-gold)">AVG</th>'

    # Split probable vs unlikely
    probable = [rp for rp in matrix if rp.get("appear_pct", 0) >= _APPEAR_THRESHOLD]
    unlikely = [rp for rp in matrix if rp.get("appear_pct", 0) < _APPEAR_THRESHOLD]

    rows = ""
    for rp in probable:
        rows += _render_bullpen_row(rp, batters)

    if unlikely:
        n_cols = len(batters) + 2
        rows += (
            f'<tr class="gp-bp-divider"><td colspan="{n_cols}">Less likely</td></tr>'
        )
        for rp in unlikely:
            rows += _render_bullpen_row(rp, batters, is_depth=True)

    probable_label = f"{len(probable)} probable" if probable else ""
    unlikely_label = f", {len(unlikely)} depth" if unlikely else ""

    return (
        '<div class="gp-card" style="padding:0.8rem">'
        '<div class="gp-card-eyebrow">Bullpen Matchup Matrix</div>'
        f'<div class="gp-card-sub">'
        f'Matchup xwOBA per reliever vs each batter ({probable_label}{unlikely_label}). '
        f'Green = pitcher advantage, red = hitter advantage.</div>'
        '<div class="gp-bp-wrap">'
        '<table class="gp-bp-table">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table>'
        '</div>'
        '<div class="gp-bp-legend">'
        '<span><span class="sw" style="background:rgba(107,163,142,0.25)"></span>Strong pitcher edge</span>'
        '<span><span class="sw" style="background:rgba(107,163,142,0.12)"></span>Slight</span>'
        '<span><span class="sw" style="background:rgba(212,86,42,0.12)"></span>Slight hitter</span>'
        '<span><span class="sw" style="background:rgba(212,86,42,0.25)"></span>Strong hitter</span>'
        '</div>'
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
            '<div class="gp-pinch-row">'
            f'{headshot_html(o["bench_id"], size=24)}'
            f'<div class="who">'
            f'<span style="color:var(--tdd-cream);font-weight:600">{esc(o["bench_name"])}</span>'
            f' <span style="color:var(--tdd-slate);font-size:0.55rem">{esc(o["bench_pos"])}</span>'
            f'<span style="color:{b_color};font-family:var(--tdd-font-body);'
            f'font-size:0.7rem;margin-left:0.4rem">.{int(b_xw*1000):03d}</span>'
            '</div>'
            '<div style="color:var(--tdd-slate);font-size:0.6rem;padding:0 4px">for</div>'
            f'{headshot_html(o["starter_id"], size=24)}'
            f'<div class="who">'
            f'<span style="color:var(--tdd-cream);opacity:0.7">{esc(o["starter_name"])}</span>'
            f' <span style="color:var(--tdd-slate);font-size:0.55rem">#{o["starter_order"]}</span>'
            f'<span style="color:{s_color};font-family:var(--tdd-font-body);'
            f'font-size:0.7rem;margin-left:0.4rem">.{int(s_xw*1000):03d}</span>'
            '</div>'
            f'<span class="gain">+{gain_pts:.0f} pts</span>'
            '</div>'
        )

    return (
        '<div class="gp-card">'
        '<div class="gp-card-eyebrow">Pinch-Hit Opportunities</div>'
        f'<div class="gp-card-sub">'
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
        '<div class="gp-putaway">'
        '<div class="head">'
        f'<span class="lbl">2-STRIKE PUTAWAY vs {hand_label}</span>'
        '</div>'
        f'<div class="body">'
        f'{usage_pct*100:.0f}% {pt_name.lower()}{whiff_str}'
        f'{secondary}'
        f'</div>'
        f'<div class="loc">'
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
        f'<div class="gp-pitch" style="border-left:3px solid {style["color"]};'
        f'background:{style["bg"]}">'
        '<div class="gp-pitch-top">'
        '<div class="left">'
        f'<span class="pt-name">{esc(plan["pitch_name"])}</span>'
        f'<span class="usage">{usage_label}</span>'
        f'<span class="velo">{velo_str} mph</span>'
        '</div>'
        '<div class="right">'
        f'<span class="app-label" style="color:{style["color"]}">{style["label"]}</span>'
        f'<span class="xw" style="color:{style["color"]}">.{int(xw*1000):03d}</span>'
        '</div>'
        '</div>'
        f'<div class="note">{esc(plan["recommendation"])}</div>'
        '<div class="micro">'
        f'<span>Whiff {whiff*100:.0f}%</span>'
        f'<span>xwOBA con .{int(plan["matchup_xwoba_contact"]*1000):03d}</span>'
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
    projections: dict | None = None,
    sim_stats: dict | None = None,
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

    # Bayesian projection stats row for the attack card header
    proj_row_html = ""
    if projections:
        _proj_parts: list[str] = []
        _pk = projections.get("projected_k_rate")
        _pk_lo = projections.get("projected_k_rate_2_5")
        _pk_hi = projections.get("projected_k_rate_97_5")
        _pbb = projections.get("projected_bb_rate")
        _pbb_lo = projections.get("projected_bb_rate_2_5")
        _pbb_hi = projections.get("projected_bb_rate_97_5")
        _pw = projections.get("projected_woba")
        _pw_lo = projections.get("projected_woba_2_5")
        _pw_hi = projections.get("projected_woba_97_5")

        if _pk:
            _kc = SAGE if _pk < 0.20 else (EMBER if _pk > 0.28 else SLATE)
            _kci = f' <span style="opacity:0.5">[{_pk_lo*100:.0f}-{_pk_hi*100:.0f}]</span>' if _pk_lo and _pk_hi else ""
            _proj_parts.append(f'<span style="color:{_kc}">K% {_pk*100:.0f}{_kci}</span>')
        if _pbb:
            _bbc = SAGE if _pbb > 0.10 else (EMBER if _pbb < 0.06 else SLATE)
            _bbci = f' <span style="opacity:0.5">[{_pbb_lo*100:.0f}-{_pbb_hi*100:.0f}]</span>' if _pbb_lo and _pbb_hi else ""
            _proj_parts.append(f'<span style="color:{_bbc}">BB% {_pbb*100:.0f}{_bbci}</span>')
        if _pw:
            _wc = SAGE if _pw > 0.340 else (EMBER if _pw < 0.300 else SLATE)
            _wci = f' <span style="opacity:0.5">[.{int(_pw_lo*1000):03d}-.{int(_pw_hi*1000):03d}]</span>' if _pw_lo and _pw_hi else ""
            _proj_parts.append(f'<span style="color:{_wc}">wOBA .{int(_pw*1000):03d}{_wci}</span>')

        if _proj_parts:
            proj_row_html = (
                f'<div style="margin-top:4px;font-size:0.72rem;'
                f'font-family:var(--tdd-font-mono)">'
                f'{" &middot; ".join(_proj_parts)}'
                f'</div>'
            )

    # Game sim row (matchup-adjusted Monte Carlo projections)
    sim_row_html = ""
    if sim_stats:
        _sim_parts: list[str] = []
        for _sk, _sl, _good, _bad, _high_good in [
            ("H", "H", 1.0, 0.8, True),
            ("K", "K", 0.8, 1.0, False),
            ("BB", "BB", 0.5, 0.3, True),
            ("TB", "TB", 1.5, 1.0, True),
        ]:
            _sv = sim_stats.get(_sk)
            if _sv and _sv.get("expected") is not None:
                _exp = _sv["expected"]
                _p1 = _sv.get("p_1plus")
                if _high_good:
                    _sc = SAGE if _exp >= _good else (EMBER if _exp < _bad else SLATE)
                else:
                    _sc = SAGE if _exp < _good else (EMBER if _exp >= _bad else SLATE)
                _p1_str = f' ({_p1*100:.0f}% 1+)' if _p1 is not None else ""
                _sim_parts.append(f'<span style="color:{_sc}">{_exp:.1f} {_sl}{_p1_str}</span>')
        if _sim_parts:
            sim_row_html = (
                f'<div style="margin-top:3px;font-size:0.72rem;'
                f'font-family:var(--tdd-font-mono)">'
                f'<span style="color:{GOLD};font-size:0.6rem;font-weight:700;'
                f'letter-spacing:0.5px;margin-right:0.4rem">GAME SIM</span>'
                f'{" &middot; ".join(_sim_parts)}'
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
                '<div class="gp-pitch-sep"></div>'
                '<div class="gp-pitch-sep-label">SECONDARY</div>'
            )
        pitch_rows += _render_pitch_plan_row(p, batter_label=_bl)

    # Card class: gold accent for comp-based cards
    card_cls = "gp-attack comp-border" if comp_info else "gp-attack"

    # Body content: pitch plans + putaway
    body_content = f'{pitch_rows}{putaway_html}'

    if zone_chart_b64:
        body_html = (
            '<div class="gp-attack-body-split">'
            f'<div class="plans">{body_content}</div>'
            f'<div class="gp-zone-area">'
            f'<div class="caption">Hunt pitch location</div>'
            f'<img src="{zone_chart_b64}" />'
            f'</div>'
            '</div>'
        )
    else:
        body_html = body_content

    return (
        f'<div class="{card_cls}">'
        '<div class="gp-attack-head">'
        f'<span class="order">{batting_order}</span>'
        f'{headshot_html(batter_id, size=48)}'
        '<div class="who">'
        f'<div class="nm">'
        f'<span class="name">{esc(batter_name)}</span>'
        f'<span class="team">{esc(team_abbr)}</span>'
        f'{hand_html}'
        f'{platoon_html}'
        f'{milb_badge_html}'
        '</div>'
        f'<div class="summary">{esc(plan["summary"])}</div>'
        f'{comp_subtitle_html}'
        f'{proj_row_html}'
        f'{sim_row_html}'
        '</div>'
        f'<div class="edge-stack">'
        f'<div class="xw" style="color:{edge_color}">.{int(xw*1000):03d}</div>'
        f'<div class="lab" style="color:{edge_color}">{edge_label}</div>'
        '</div>'
        '</div>'
        f'{body_html}'
        '</div>'
    )


def _render_pitcher_overview(
    pitcher_name: str,
    pitch_hand: str | None,
    form: dict | None,
    walk_strat: dict | None,
    zone_b64: str | None,
    arsenal_html: str = "",
) -> str:
    """Render the opposing starter overview card with form + zone chart."""
    hand_label = f"{'LHP' if pitch_hand == 'L' else 'RHP'}" if pitch_hand else ""
    trend_html = _get_trend_badge(form)

    # Form stat grid (inspired by mobile design)
    form_html = ""
    if form:
        era_color = SAGE if form["era"] < 3.50 else (EMBER if form["era"] > 4.50 else SLATE)
        form_html = (
            '<div class="gp-pov-form">'
            f'<div class="stat"><div class="v" style="color:{era_color}">{form["era"]:.2f}</div>'
            f'<div class="k">ERA L{form["n_starts"]}</div></div>'
            f'<div class="stat"><div class="v">{form["k_per_9"]}</div>'
            f'<div class="k">K/9</div></div>'
            f'<div class="stat"><div class="v">{form["bb_per_9"]}</div>'
            f'<div class="k">BB/9</div></div>'
            f'<div class="stat"><div class="v">{form["avg_ip"]}</div>'
            f'<div class="k">IP/Start</div></div>'
            f'<div class="stat"><div class="v">{int(form["avg_pitches"])}</div>'
            f'<div class="k">Pitch Avg</div></div>'
            '</div>'
        )

    # Walk tendency callout
    walk_html = ""
    if walk_strat:
        walk_html = (
            f'<div class="gp-pov-warn">'
            f'&#9888; {esc(walk_strat["note"])}'
            f'</div>'
        )

    # Zone chart image
    zone_html = ""
    if zone_b64:
        zone_html = (
            f'<div style="margin-top:0.8rem">'
            f'<div style="color:var(--tdd-slate);font-size:0.78rem;margin-bottom:0.3rem">'
            f'Pitch location density -- where he throws each pitch type (darker = more frequent)</div>'
            f'<img src="{zone_b64}" style="width:100%;border-radius:4px" />'
            f'</div>'
        )

    return (
        f'<div class="gp-card">'
        f'<div class="gp-card-eyebrow">Opposing Starter</div>'
        f'<div class="gp-pov-head">'
        f'<div style="flex:1;min-width:0">'
        f'<span class="gp-pov-name">{esc(pitcher_name)}</span>'
        f'<div class="gp-pov-hand">'
        f'<span>{hand_label}</span>'
        f'{trend_html}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'{form_html}'
        f'{arsenal_html}'
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

    # Today's date (ET)
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    today = et_now.date()

    # Load today's schedule
    schedule = load_todays_games()
    if not schedule.empty and "game_date" in schedule.columns:
        schedule = schedule[schedule["game_date"] == today.isoformat()].copy()
    if schedule.empty:
        # Fallback to live API
        schedule = fetch_live_schedule(today.isoformat())

    if schedule.empty:
        st.markdown(
            f'<div class="pl-empty">No games scheduled for {today.strftime("%b %d")}.</div></div>',
            unsafe_allow_html=True,
        )
        return

    if "game_datetime_utc" in schedule.columns:
        schedule = schedule.sort_values("game_datetime_utc", na_position="last")

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

    sel_idx = st.selectbox(
        "Select Game", range(len(game_labels)),
        format_func=lambda i: game_labels[i],
        key="gp_game_sel", label_visibility="collapsed",
    )
    gpk = game_pks[sel_idx]
    game = schedule[schedule["game_pk"] == gpk].iloc[0]

    # Load lineups -- build from roster for all position players
    lineups = load_todays_lineups()
    lineups = backfill_missing_lineups(schedule, lineups)

    # Filter out IL / non-active players and mislabeled pitchers from lineups
    if not lineups.empty:
        _roster_check = load_roster()
        if not _roster_check.empty:
            active_ids = set(
                _roster_check[_roster_check["roster_status"] == "active"]["player_id"].astype(int)
            )
            lineups = lineups[lineups["batter_id"].astype(int).isin(active_ids)]
        # Remove pitchers mislabeled as position players
        from services.data_loader import load_reliever_rankings
        _rr = load_reliever_rankings()
        if not _rr.empty:
            pitcher_ids = set(_rr["pitcher_id"].astype(int))
            lineups = lineups[~lineups["batter_id"].astype(int).isin(pitcher_ids)]

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

    # Game sim results (matchup-adjusted Monte Carlo projections)
    game_props_df = load_game_props()
    platoon_bb_df = load_pitcher_platoon_bb()

    # Observed 2026 wOBA fallback for players without vulnerability data
    _obs_woba_fallback: dict[int, float] = {}
    try:
        _trad = pd.read_parquet(
            str(Path(__file__).resolve().parents[1] / "data" / "dashboard" / "hitter_traditional.parquet")
        )
        if not _trad.empty and "woba" in _trad.columns and "pa" in _trad.columns:
            for _, _r in _trad[_trad["pa"] >= 10].iterrows():
                _obs_woba_fallback[int(_r["batter_id"])] = float(_r["woba"])
    except Exception:
        pass

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
        platoon_bb = get_pitcher_platoon_bb(pid, platoon_bb_df)
        walk_strat = assess_walk_strategy(
            pid, proj_df, adv_df, form=form, platoon_bb=platoon_bb,
        )

        # Zone heatmap for pitcher overview
        zone_overview_b64 = None
        if not location_df.empty:
            p_loc = location_df[location_df["pitcher_id"] == pid]
            if not p_loc.empty:
                try:
                    fig = plot_pitcher_location_split(
                        p_loc, pitcher_name=pitcher_name,
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

            # Pitcher overview with form + zone chart + arsenal bar
            arsenal_bar_html = _render_arsenal_bar(arsenal_df, pid)
            overview_html = _render_pitcher_overview(
                pitcher_name, p_hand, form, walk_strat, zone_overview_b64,
                arsenal_html=arsenal_bar_html,
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
                    '<div class="gp-milb-banner">'
                    '<span class="label">MiLB COMP ARSENAL</span>'
                    f'<span class="comps">Based on: {", ".join(comp_parts)}</span>'
                    + (f'<div class="rates">{" &nbsp; ".join(rate_parts)}</div>' if rate_parts else '')
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

                # Fallback: use observed 2026 wOBA as matchup xwOBA
                if edge is None and bid in _obs_woba_fallback:
                    obs_woba = _obs_woba_fallback[bid]
                    adv_label = (
                        "hitter" if obs_woba > 0.325
                        else ("pitcher" if obs_woba < 0.305 else "even")
                    )
                    edge = {
                        "matchup_xwoba": obs_woba,
                        "league_xwoba": 0.315,
                        "edge": obs_woba - 0.315,
                        "advantage": adv_label,
                        "platoon_edge": None,
                    }

                # Look up Bayesian projections for this batter
                b_proj: dict = {}
                if not hitter_proj_df.empty:
                    hp_row = hitter_proj_df[hitter_proj_df["batter_id"] == bid]
                    if not hp_row.empty:
                        for col in (
                            "projected_k_rate", "projected_bb_rate", "projected_woba",
                            "projected_k_rate_2_5", "projected_k_rate_97_5",
                            "projected_bb_rate_2_5", "projected_bb_rate_97_5",
                            "projected_woba_2_5", "projected_woba_97_5",
                        ):
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

            # --- Enrich with game sim distributions ---
            if not game_props_df.empty:
                game_bp = game_props_df[
                    (game_props_df["game_pk"] == gpk)
                    & (game_props_df["player_type"] == "batter")
                ]
                for b in batter_data:
                    bid = b["batter_id"]
                    bp = game_bp[game_bp["player_id"] == bid]
                    if bp.empty:
                        continue
                    sim_stats: dict[str, dict] = {}
                    for _, row in bp.iterrows():
                        stat = row["stat"]
                        if stat in ("H", "K", "BB", "TB"):
                            p1 = row.get("p_over_0.5")
                            sim_stats[stat] = {
                                "expected": round(row["expected"], 2),
                                "std": round(row["std"], 2),
                                "p_1plus": round(p1, 3) if pd.notna(p1) else None,
                                "p_2plus": round(row.get("p_over_1.5", 0), 3) if pd.notna(row.get("p_over_1.5")) else None,
                                "p_0": round(1.0 - p1, 3) if pd.notna(p1) else None,
                            }
                    if sim_stats:
                        b["sim_stats"] = sim_stats

            # --- Build position lookup ---
            roster_df = load_roster()
            pos_lookup: dict[int, str] = {}
            if not roster_df.empty:
                for _, r in roster_df.iterrows():
                    pos_lookup[int(r["player_id"])] = r.get("primary_position", "")

            for b in batter_data:
                b["position"] = pos_lookup.get(b["batter_id"], "")

            # --- Detect official vs roster-based lineup ---
            lu_sources = opp_lu["lineup_source"].unique() if "lineup_source" in opp_lu.columns else []
            has_official = "api" in lu_sources

            if has_official:
                # Official lineup posted -- show manager's order with our overlay
                # Starters = batters with batting_order 1-9 from official lineup
                official_ids = set(
                    opp_lu[
                        (opp_lu["lineup_source"] == "api")
                        & (opp_lu["batting_order"].between(1, 9))
                    ]["batter_id"].astype(int)
                )
                starters = [b for b in batter_data if b["batter_id"] in official_ids]
                # Preserve the manager's filed order
                official_order = {
                    int(r["batter_id"]): int(r["batting_order"])
                    for _, r in opp_lu[opp_lu["lineup_source"] == "api"].iterrows()
                    if r["batting_order"] <= 9
                }
                for b in starters:
                    b["current_order"] = official_order.get(b["batter_id"], 99)
                starters.sort(key=lambda b: b["current_order"])

                bench_pool = [b for b in batter_data if b["batter_id"] not in official_ids]

                # Compute our optimal rank for each batter (for overlay)
                all_ranked = sorted(
                    batter_data,
                    key=lambda x: x["matchup_xwoba"], reverse=True,
                )
                for rank, b in enumerate(all_ranked, 1):
                    b["optimal_rank"] = rank
            else:
                # No official lineup -- rank all by matchup xwOBA
                all_ranked = sorted(
                    batter_data,
                    key=lambda x: x["matchup_xwoba"], reverse=True,
                )
                starters = all_ranked[:9]
                bench_pool = all_ranked[9:]
                for i, b in enumerate(starters):
                    b["current_order"] = i + 1
                    b["optimal_rank"] = i + 1
                for i, b in enumerate(bench_pool, 10):
                    b["optimal_rank"] = i

            bench_data: list[dict] = []
            for b in bench_pool:
                bench_data.append({
                    "batter_id": b["batter_id"],
                    "batter_name": b["batter_name"],
                    "matchup_xwoba": b["matchup_xwoba"],
                    "position": pos_lookup.get(b["batter_id"], ""),
                    "optimal_rank": b.get("optimal_rank", 99),
                    "projections": b.get("projections", {}),
                    "sim_stats": b.get("sim_stats", {}),
                    "plan": b.get("plan"),
                    "edge": b.get("edge"),
                    "batter_label": b.get("batter_label", ""),
                })
            bench_data.sort(key=lambda x: x["matchup_xwoba"], reverse=True)

            # --- Lineup card ---
            scoreable = starters
            if len(scoreable) >= 4:
                opt_html = _render_lineup_optimization_html(
                    scoreable, bench_data, pitcher_name, opp_abbr,
                    is_official=has_official,
                )
                if opt_html:
                    st.markdown(opt_html, unsafe_allow_html=True)

            # --- Pinch-Hit Opportunities (only with official lineups) ---
            if has_official and bench_data and scoreable:
                ph_opps = _find_pinch_hit_opportunities(scoreable, bench_data)
                if ph_opps:
                    ph_html = _render_pinch_hit_html(ph_opps, pitcher_name)
                    if ph_html:
                        st.markdown(ph_html, unsafe_allow_html=True)

            # --- Attack plan cards (starters only, carousel on mobile) ---
            all_attack_cards: list[str] = []
            for i, b in enumerate(starters):
                bid = b["batter_id"]
                order = b["current_order"]
                if b["plan"] is None:
                    all_attack_cards.append(
                        f'<div class="gp-attack" style="display:flex;align-items:center;gap:0.6rem;padding:0.8rem 1rem">'
                        f'<span class="gp-attack-head order" style="font-size:1.1rem;min-width:1.4rem">{order}</span>'
                        f'{headshot_html(bid, size=40)}'
                        f'<span style="color:var(--tdd-cream);font-family:var(--tdd-font-heading);'
                        f'font-weight:700;font-size:0.95rem">{esc(b["batter_name"])}</span>'
                        f'<span style="color:var(--tdd-slate);font-size:0.7rem">No matchup data</span>'
                        f'</div>'
                    )
                else:
                    # Zone chart for top 6 batters
                    zone_b64 = None
                    if i < 6 and not location_df.empty and b.get("batter_hand"):
                        p_loc = location_df[location_df["pitcher_id"] == pid]
                        if not p_loc.empty:
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
                    all_attack_cards.append(
                        _render_batter_attack_card(
                            bid, b["batter_name"], b["team_abbr"],
                            order, b["plan"], b["edge"],
                            putaway_html=pa_html,
                            comp_info=b.get("comp_info"),
                            zone_chart_b64=zone_b64,
                            projections=b.get("projections"),
                            sim_stats=b.get("sim_stats"),
                        )
                    )

            if all_attack_cards:
                n_cards = len(all_attack_cards)
                carousel_id = f"gp-car-{side}"
                # Dots (mobile only)
                dots = "".join(
                    f'<span class="gp-dot{" active" if i == 0 else ""}"></span>'
                    for i in range(n_cards)
                )
                nav_html = (
                    '<div class="gp-carousel-nav">'
                    f'<span class="gp-carousel-counter">1 / {n_cards}</span>'
                    f'<button class="gp-carousel-btn prev" data-car="{carousel_id}" aria-label="Previous">&#8249;</button>'
                    f'<button class="gp-carousel-btn next" data-car="{carousel_id}" aria-label="Next">&#8250;</button>'
                    '</div>'
                )
                carousel_html = (
                    '<div class="gp-carousel-wrap">'
                    f'{nav_html}'
                    f'<div id="{carousel_id}" class="gp-carousel">'
                    + "".join(all_attack_cards)
                    + '</div>'
                    f'<div class="gp-carousel-dots">{dots}</div>'
                    '</div>'
                )
                st.markdown(carousel_html, unsafe_allow_html=True)

                # Inject JS controller via st.components.v1.html (invisible iframe)
                import streamlit.components.v1 as stc
                stc.html(
                    '<script>'
                    'function gpInit() {'
                    '  var doc = window.parent.document;'
                    f'  var c = doc.getElementById("{carousel_id}");'
                    '  if (!c) return;'
                    '  var wrap = c.closest(".gp-carousel-wrap");'
                    '  if (!wrap) return;'
                    '  var cards = c.querySelectorAll(".gp-attack");'
                    '  var dots = wrap.querySelectorAll(".gp-dot");'
                    '  var counter = wrap.querySelector(".gp-carousel-counter");'
                    '  var n = cards.length;'
                    '  if (!n) return;'
                    # Update dots + counter on scroll
                    '  function update() {'
                    '    var w = cards[0].getBoundingClientRect().width + 8;'
                    '    var idx = Math.round(c.scrollLeft / w);'
                    '    idx = Math.max(0, Math.min(idx, n - 1));'
                    '    dots.forEach(function(d, i) { d.classList.toggle("active", i === idx); });'
                    '    if (counter) counter.textContent = (idx + 1) + " / " + n;'
                    '  }'
                    '  c.addEventListener("scroll", update, {passive: true});'
                    # Prev/next buttons
                    '  var prevBtn = wrap.querySelector(".gp-carousel-btn.prev");'
                    '  var nextBtn = wrap.querySelector(".gp-carousel-btn.next");'
                    '  if (prevBtn) prevBtn.addEventListener("click", function() {'
                    '    var w = cards[0].getBoundingClientRect().width + 8;'
                    '    c.scrollBy({left: -w, behavior: "smooth"});'
                    '  });'
                    '  if (nextBtn) nextBtn.addEventListener("click", function() {'
                    '    var w = cards[0].getBoundingClientRect().width + 8;'
                    '    c.scrollBy({left: w, behavior: "smooth"});'
                    '  });'
                    # Dot click navigation
                    '  dots.forEach(function(d, i) {'
                    '    d.style.cursor = "pointer";'
                    '    d.addEventListener("click", function() {'
                    '      var w = cards[0].getBoundingClientRect().width + 8;'
                    '      c.scrollTo({left: i * w, behavior: "smooth"});'
                    '    });'
                    '  });'
                    '}'
                    'gpInit();'
                    '</script>',
                    height=0,
                )

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
