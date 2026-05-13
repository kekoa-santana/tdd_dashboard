"""Scouting report bullet generators and renderers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, POSITIVE, NEGATIVE,
    PITCH_DISPLAY, PITCH_FAMILY_DISPLAY,
    GRADE_LABELS, HITTER_GRADE_SKILLS, PITCHER_GRADE_SKILLS,
    GOOD_DIRECTION_LABEL,
)
from components.metric_cards import percentile_rank
from utils.formatters import fmt_stat


def generate_scouting_bullets(
    stat_configs: list[tuple[str, str, bool, str]],
    player_row: pd.Series,
    all_df: pd.DataFrame,
    player_type: str,
) -> list[tuple[str, str]]:
    """Generate plain English scouting report bullets.

    Returns list of (color_hex, text) tuples.
    """
    bullets: list[tuple[str, str]] = []

    for label, key, higher_better, _desc in stat_configs:
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        ci_lo_col = f"projected_{key}_2_5"
        ci_hi_col = f"projected_{key}_97_5"

        if obs_col not in player_row.index or pd.isna(player_row.get(obs_col)):
            continue

        observed = player_row[obs_col]
        projected = player_row[proj_col]
        delta_pp = (projected - observed) * 100

        ci_lo = player_row.get(ci_lo_col, projected)
        ci_hi = player_row.get(ci_hi_col, projected)
        ci_width = (ci_hi - ci_lo) * 100

        pctile = percentile_rank(all_df[proj_col], projected, higher_better)

        improving = (delta_pp > 0 and higher_better) or (delta_pp < 0 and not higher_better)

        obs_str = fmt_stat(observed, key)
        proj_str = fmt_stat(projected, key)

        if abs(delta_pp) < 0.5:
            direction_text = (
                f"{label} projected to hold steady at {proj_str}"
            )
            dot_color = SLATE
        elif improving:
            good_label = GOOD_DIRECTION_LABEL.get((key, higher_better), "improve")
            if abs(delta_pp) > 3:
                direction_text = (
                    f"{label} jumps from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- expect him to {good_label}"
                )
            else:
                direction_text = (
                    f"{label} ticks from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- slight improvement"
                )
            dot_color = POSITIVE
        else:
            if abs(delta_pp) > 3:
                direction_text = (
                    f"{label} projected to slide from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- notable regression risk"
                )
            else:
                direction_text = (
                    f"{label} may slip from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- minor adjustment"
                )
            dot_color = NEGATIVE

        if ci_width < 6:
            conf_text = "high confidence"
        elif ci_width < 12:
            conf_text = "moderate confidence"
        else:
            conf_text = "wide range of outcomes"

        if pctile >= 90:
            rank_text = f"elite ({pctile:.0f}th percentile)"
        elif pctile >= 75:
            rank_text = f"above-average ({pctile:.0f}th pctile)"
        elif pctile >= 40:
            rank_text = f"mid-tier ({pctile:.0f}th pctile)"
        else:
            rank_text = f"below-average ({pctile:.0f}th pctile)"

        full_text = f"{direction_text}. {conf_text.capitalize()}, {rank_text}."
        bullets.append((dot_color, full_text))

    return bullets


def compute_matchup_xwoba_edge(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
    *,
    pitcher_hand: str | None = None,
    batter_hand: str | None = None,
) -> dict:
    """Compute odds-ratio-adjusted expected xwOBA for the matchup.

    Uses Tango's odds-ratio method (The Book, Ch. 10):
        matchup_rate = (pitcher_rate * batter_rate) / league_rate

    applied per pitch type, then usage-weighted across the arsenal.

    The result is a full-PA xwOBA that accounts for both contact quality
    AND strikeout probability:
        full_xwoba = P(contact) * xwOBA_on_contact

    Returns dict with:
        matchup_xwoba   -- expected xwOBA for this batter vs this pitcher
        league_xwoba    -- league average xwOBA (~0.315)
        edge            -- matchup_xwoba - league_xwoba (positive = hitter edge)
        advantage       -- "hitter" | "pitcher" | "even"
        platoon_edge    -- "favorable" | "unfavorable" | None
    """
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    LG_XWOBA = LEAGUE_AVG_OVERALL.get("xwoba", 0.315)
    LG_K_RATE = LEAGUE_AVG_OVERALL.get("k_rate", 0.224)
    XWOBA_K = 0.0  # strikeouts have zero wOBA value

    empty = {
        "matchup_xwoba": LG_XWOBA, "league_xwoba": LG_XWOBA,
        "edge": 0.0, "advantage": "even", "platoon_edge": None,
    }

    p_df = arsenal_df.copy()
    p_df = p_df[p_df["pitches"] >= 20]
    if p_df.empty:
        return empty

    v_df = vuln_df.copy()
    v_df = v_df[v_df["pitches"] >= 15] if "pitches" in v_df.columns else v_df
    if not v_df.empty:
        v_df = v_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first",
        )
    s_df = str_df.copy() if not str_df.empty else pd.DataFrame()
    if not s_df.empty and "pitches" in s_df.columns:
        s_df = s_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first",
        )

    total_usage = 0.0
    weighted_xwoba = 0.0

    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        usage = row.get("usage_pct", 0)
        if usage <= 0:
            continue
        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_xwoba_con = lg.get("xwoba_contact", 0.320)

        # --- Pitcher rates ---
        p_whiff = row.get("whiff_rate", lg_whiff)
        if pd.isna(p_whiff):
            p_whiff = lg_whiff
        p_xwoba_con = row.get("xwoba_against", lg_xwoba_con)
        if pd.isna(p_xwoba_con):
            p_xwoba_con = lg_xwoba_con

        # --- Batter rates ---
        h_row = v_df[v_df["pitch_type"] == pt]
        s_row = s_df[s_df["pitch_type"] == pt] if not s_df.empty else pd.DataFrame()

        # Batter whiff rate vs this pitch type
        b_whiff = lg_whiff  # default: league average
        b_whiff_n = 0
        if len(h_row) > 0 and "swings" in h_row.columns and "whiffs" in h_row.columns:
            sw = h_row["swings"].iloc[0]
            wh = h_row["whiffs"].iloc[0]
            if pd.notna(sw) and sw >= 10:
                b_whiff = wh / sw
                b_whiff_n = int(sw)

        # Batter xwOBA on contact vs this pitch type
        b_xwoba_con = lg_xwoba_con  # default: league average
        b_xwoba_n = 0
        if len(s_row) > 0 and "xwoba_contact" in s_row.columns:
            val = s_row["xwoba_contact"].iloc[0]
            n = s_row["bip"].iloc[0] if "bip" in s_row.columns else 0
            if pd.notna(val) and pd.notna(n) and n >= 10:
                b_xwoba_con = val
                b_xwoba_n = int(n)
        elif len(h_row) > 0 and "xwoba_contact" in h_row.columns:
            val = h_row["xwoba_contact"].iloc[0]
            n = h_row["bip"].iloc[0] if "bip" in h_row.columns else 0
            if pd.notna(val) and pd.notna(n) and n >= 10:
                b_xwoba_con = val
                b_xwoba_n = int(n)

        # Regress toward league average for small samples
        # ~50 swings for whiff stability, ~30 BIP for xwOBA stability
        whiff_reg = min(b_whiff_n / 50.0, 1.0) if b_whiff_n > 0 else 0.0
        b_whiff = whiff_reg * b_whiff + (1 - whiff_reg) * lg_whiff

        xwoba_reg = min(b_xwoba_n / 30.0, 1.0) if b_xwoba_n > 0 else 0.0
        b_xwoba_con = xwoba_reg * b_xwoba_con + (1 - xwoba_reg) * lg_xwoba_con

        # --- Odds-ratio method for whiff rate ---
        # Clamp to avoid division by zero
        _clamp = lambda x: max(0.01, min(0.99, x))
        p_w = _clamp(p_whiff)
        b_w = _clamp(b_whiff)
        lg_w = _clamp(lg_whiff)

        matchup_whiff = (p_w * b_w / lg_w) / (
            p_w * b_w / lg_w + (1 - p_w) * (1 - b_w) / (1 - lg_w)
        )

        # --- Odds-ratio method for xwOBA on contact ---
        # Scale xwOBA to [0,1] range for odds-ratio (divide by max ~0.7)
        _scale = 0.7
        p_x = _clamp(p_xwoba_con / _scale)
        b_x = _clamp(b_xwoba_con / _scale)
        lg_x = _clamp(lg_xwoba_con / _scale)

        matchup_xwoba_con = (p_x * b_x / lg_x) / (
            p_x * b_x / lg_x + (1 - p_x) * (1 - b_x) / (1 - lg_x)
        ) * _scale

        # --- Full-PA xwOBA = P(contact) * xwOBA_on_contact ---
        # (strikeouts contribute 0 to wOBA)
        p_contact = 1.0 - matchup_whiff
        matchup_xwoba_pt = p_contact * matchup_xwoba_con

        weighted_xwoba += usage * matchup_xwoba_pt
        total_usage += usage

    if total_usage <= 0:
        return empty

    matchup_xwoba = weighted_xwoba / total_usage
    edge = matchup_xwoba - LG_XWOBA

    # Threshold: ~0.010 xwOBA is roughly 1 run per 60 PA
    if edge > 0.010:
        advantage = "hitter"
    elif edge < -0.010:
        advantage = "pitcher"
    else:
        advantage = "even"

    # Platoon edge
    platoon_edge = None
    if pitcher_hand and batter_hand:
        ph = pitcher_hand.upper()[0] if pitcher_hand else None
        bh = batter_hand.upper()[0] if batter_hand else None
        if ph and bh:
            if ph != bh:  # opposite hand = batter advantage
                platoon_edge = "favorable"
            elif bh != "S":  # same hand = pitcher advantage (unless switch)
                platoon_edge = "unfavorable"

    return {
        "matchup_xwoba": matchup_xwoba,
        "league_xwoba": LG_XWOBA,
        "edge": edge,
        "advantage": advantage,
        "platoon_edge": platoon_edge,
    }


def build_matchup_scouting_bullets(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
    pitcher_name: str,
    hitter_name: str,
    *,
    pitcher_hand: str | None = None,
    batter_hand: str | None = None,
) -> list[tuple[str, str]]:
    """Generate descriptive per-pitch scouting bullets for the matchup.

    Bullets describe observable facts (whiff rates, contact quality, chase
    tendencies) per pitch type.  They do NOT declare overall advantage --
    that comes from ``score_matchup_advantage()`` in ``lib/matchup.py``.

    Color coding:
      EMBER -- detail favors the pitcher (high whiff, high chase, weak contact)
      SAGE  -- detail favors the hitter  (low whiff, hard contact, low chase)
      GOLD  -- neutral / notable
    """
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    p_df = arsenal_df.copy()
    p_df = p_df[p_df["pitches"] >= 20]
    if p_df.empty:
        return []

    v_df = vuln_df.copy()
    v_df = v_df[v_df["pitches"] >= 15] if "pitches" in v_df.columns else v_df
    if not v_df.empty:
        v_df = v_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )
    s_df = str_df.copy() if not str_df.empty else pd.DataFrame()
    if not s_df.empty and "pitches" in s_df.columns:
        s_df = s_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )

    # Collect per-pitch observations: (pitch_name, usage, pitcher_facts, hitter_facts)
    pitch_notes: list[tuple[str, float, list[tuple[str, str]], list[tuple[str, str]]]] = []

    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_xwoba = lg.get("xwoba_contact", 0.320)
        lg_chase = lg.get("chase_rate", 0.30)

        p_whiff = row.get("whiff_rate", np.nan)
        usage = row.get("usage_pct", 0)

        h_row = v_df[v_df["pitch_type"] == pt]
        h_whiff = np.nan
        h_swings = 0
        if len(h_row) > 0 and "swings" in h_row.columns and "whiffs" in h_row.columns:
            sw = h_row["swings"].iloc[0]
            wh = h_row["whiffs"].iloc[0]
            if pd.notna(sw) and sw > 0:
                h_whiff = wh / sw
                h_swings = int(sw)

        s_row = s_df[s_df["pitch_type"] == pt] if not s_df.empty else pd.DataFrame()
        h_xwoba = np.nan
        h_bip = 0
        if len(s_row) > 0 and "xwoba_contact" in s_row.columns:
            h_xwoba = s_row["xwoba_contact"].iloc[0]
            h_bip = int(s_row["bip"].iloc[0]) if "bip" in s_row.columns and pd.notna(s_row["bip"].iloc[0]) else 0
        elif len(h_row) > 0 and "xwoba_contact" in h_row.columns:
            h_xwoba = h_row["xwoba_contact"].iloc[0]
            h_bip = int(h_row["bip"].iloc[0]) if "bip" in h_row.columns and pd.notna(h_row["bip"].iloc[0]) else 0

        h_chase = np.nan
        h_oz = 0
        if len(h_row) > 0 and "chase_swings" in h_row.columns and "out_of_zone_pitches" in h_row.columns:
            cs = h_row["chase_swings"].iloc[0]
            oz = h_row["out_of_zone_pitches"].iloc[0]
            if pd.notna(oz) and oz > 0:
                h_chase = cs / oz
                h_oz = int(oz)

        h_hh = np.nan
        h_hh_bip = 0
        if len(s_row) > 0 and "hard_hit_rate" in s_row.columns:
            h_hh = s_row["hard_hit_rate"].iloc[0]
            h_hh_bip = int(s_row["bip"].iloc[0]) if "bip" in s_row.columns and pd.notna(s_row["bip"].iloc[0]) else 0

        # Sample size annotation -- shown when sample is modest
        def _n_tag(n: int, unit: str = "pitches") -> str:
            if n >= 100:
                return ""
            return (
                f' <span style="opacity:0.5; font-size:0.6rem;">'
                f'(only {n} {unit})</span>'
            )

        # Minimum BIP to trust contact quality stats (matches edge regression)
        MIN_BIP = 30

        # Collect notable facts (color, text) -- only include if meaningfully
        # above or below league average
        pitcher_facts: list[tuple[str, str]] = []
        hitter_facts: list[tuple[str, str]] = []

        if pd.notna(p_whiff):
            if p_whiff > lg_whiff * 1.15:
                pitcher_facts.append((EMBER, f"{p_whiff*100:.0f}% whiff rate"))
            elif p_whiff < lg_whiff * 0.80:
                pitcher_facts.append((SAGE, f"low {p_whiff*100:.0f}% whiff rate"))

        if pd.notna(h_whiff) and h_swings >= 10:
            if h_whiff > lg_whiff * 1.20:
                hitter_facts.append((EMBER, f"hitter whiffs {h_whiff*100:.0f}%{_n_tag(h_swings, 'swings')}"))
            elif h_whiff < lg_whiff * 0.75:
                hitter_facts.append((SAGE, f"hitter rarely whiffs ({h_whiff*100:.0f}%){_n_tag(h_swings, 'swings')}"))

        if pd.notna(h_chase) and h_oz >= 10:
            if h_chase > lg_chase * 1.20:
                hitter_facts.append((EMBER, f"chases {h_chase*100:.0f}%{_n_tag(h_oz, 'pitches')}"))
            elif h_chase < lg_chase * 0.75:
                hitter_facts.append((SAGE, f"disciplined ({h_chase*100:.0f}% chase){_n_tag(h_oz, 'pitches')}"))

        if pd.notna(h_xwoba) and h_bip >= MIN_BIP:
            if h_xwoba >= 0.380:
                hitter_facts.append((SAGE, f"does damage on contact (.{int(h_xwoba*1000):03d} xwOBA){_n_tag(h_bip, 'BIP')}"))
            elif h_xwoba <= 0.270:
                hitter_facts.append((EMBER, f"weak contact (.{int(h_xwoba*1000):03d} xwOBA){_n_tag(h_bip, 'BIP')}"))

        if pd.notna(h_hh) and h_hh_bip >= MIN_BIP:
            if h_hh > 0.42:
                hitter_facts.append((SAGE, f"{h_hh*100:.0f}% hard hit{_n_tag(h_hh_bip, 'BIP')}"))
            elif h_hh < 0.25:
                hitter_facts.append((EMBER, f"soft contact ({h_hh*100:.0f}% hard hit){_n_tag(h_hh_bip, 'BIP')}"))

        all_facts = pitcher_facts + hitter_facts
        if all_facts:
            pitch_notes.append((pt_name, usage, pitcher_facts, hitter_facts))

    # Sort by usage (most-used pitches first -- most relevant)
    pitch_notes.sort(key=lambda x: x[1], reverse=True)

    bullets: list[tuple[str, str]] = []
    for pt_name, usage, pitcher_facts, hitter_facts in pitch_notes[:3]:
        all_facts = pitcher_facts + hitter_facts
        if not all_facts:
            continue
        # Pick the dominant color -- whichever side has more facts
        n_sage = sum(1 for c, _ in all_facts if c == SAGE)
        n_ember = sum(1 for c, _ in all_facts if c == EMBER)
        color = SAGE if n_sage >= n_ember else EMBER

        detail = ", ".join(t for _, t in all_facts)
        text = f"<b>{pt_name}</b> ({usage*100:.0f}% usage): {detail}"
        bullets.append((color, text))

    if not bullets:
        bullets.append((GOLD, "No strong pitch-level edges in this matchup"))

    # Platoon bullet (inserted at the top when relevant)
    if pitcher_hand and batter_hand:
        ph = pitcher_hand.upper()[0] if pitcher_hand else None
        bh = batter_hand.upper()[0] if batter_hand else None
        if ph and bh and bh != "S":
            hand_labels = {"L": "left", "R": "right"}
            ph_label = hand_labels.get(ph, ph)
            bh_label = hand_labels.get(bh, bh)
            if ph != bh:
                bullets.insert(0, (
                    SAGE,
                    f"<b>Platoon advantage</b>: {bh_label}-handed batter vs "
                    f"{ph_label}-handed pitcher",
                ))
            else:
                bullets.insert(0, (
                    EMBER,
                    f"<b>Platoon disadvantage</b>: {bh_label}-handed batter vs "
                    f"{ph_label}-handed pitcher",
                ))

    return bullets


# ---------------------------------------------------------------------------
# Pitcher Attack Plan (per-batter approach recommendations)
# ---------------------------------------------------------------------------

def build_attack_plan(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
    *,
    pitcher_hand: str | None = None,
    batter_hand: str | None = None,
) -> dict:
    """Generate a prescriptive attack plan for a batter vs a pitcher.

    Returns a dict with:
        summary        -- one-line approach summary (what to do)
        pitch_plans    -- list of per-pitch dicts with approach recommendations
        platoon        -- "favorable" | "unfavorable" | None
        matchup_xwoba  -- overall matchup xwOBA
        hunt_pitch     -- pitch type to sit on (best xwOBA for hitter)
        avoid_pitch    -- pitch type to lay off (worst xwOBA for hitter)
    """
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    LG_XWOBA = LEAGUE_AVG_OVERALL.get("xwoba", 0.315)

    empty = {
        "summary": "", "pitch_plans": [], "platoon": None,
        "matchup_xwoba": LG_XWOBA, "hunt_pitch": None, "avoid_pitch": None,
    }

    p_df = arsenal_df.copy()
    p_df = p_df[p_df["pitches"] >= 20]
    if p_df.empty:
        return empty

    v_df = vuln_df.copy()
    v_df = v_df[v_df["pitches"] >= 15] if "pitches" in v_df.columns else v_df
    if not v_df.empty:
        v_df = v_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first",
        )
    s_df = str_df.copy() if not str_df.empty else pd.DataFrame()
    if not s_df.empty and "pitches" in s_df.columns:
        s_df = s_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first",
        )

    # Platoon
    platoon = None
    if pitcher_hand and batter_hand:
        ph = pitcher_hand.upper()[0] if pitcher_hand else None
        bh = batter_hand.upper()[0] if batter_hand else None
        if ph and bh and bh != "S":
            platoon = "favorable" if ph != bh else "unfavorable"

    # Analyze each pitch type
    pitch_plans: list[dict] = []

    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        usage = row.get("usage_pct", 0)
        if usage <= 0:
            continue

        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_xwoba_con = lg.get("xwoba_contact", 0.320)
        lg_chase = lg.get("chase_rate", 0.30)

        p_whiff = row.get("whiff_rate", lg_whiff)
        if pd.isna(p_whiff):
            p_whiff = lg_whiff
        p_velo = row.get("avg_velo", np.nan)
        p_xwoba = row.get("xwoba_against", lg_xwoba_con)
        if pd.isna(p_xwoba):
            p_xwoba = lg_xwoba_con

        # Hitter data
        h_row = v_df[v_df["pitch_type"] == pt]
        s_row = s_df[s_df["pitch_type"] == pt] if not s_df.empty else pd.DataFrame()

        h_whiff = lg_whiff
        h_swings = 0
        if len(h_row) > 0 and "swings" in h_row.columns and "whiffs" in h_row.columns:
            sw = h_row["swings"].iloc[0]
            wh = h_row["whiffs"].iloc[0]
            if pd.notna(sw) and sw >= 10:
                h_whiff = wh / sw
                h_swings = int(sw)

        h_chase = lg_chase
        h_oz = 0
        if len(h_row) > 0 and "chase_swings" in h_row.columns and "out_of_zone_pitches" in h_row.columns:
            cs = h_row["chase_swings"].iloc[0]
            oz = h_row["out_of_zone_pitches"].iloc[0]
            if pd.notna(oz) and oz >= 10:
                h_chase = cs / oz
                h_oz = int(oz)

        h_xwoba_con = lg_xwoba_con
        h_bip = 0
        if len(s_row) > 0 and "xwoba_contact" in s_row.columns:
            val = s_row["xwoba_contact"].iloc[0]
            n = s_row["bip"].iloc[0] if "bip" in s_row.columns else 0
            if pd.notna(val) and pd.notna(n) and n >= 10:
                h_xwoba_con = val
                h_bip = int(n)
        elif len(h_row) > 0 and "xwoba_contact" in h_row.columns:
            val = h_row["xwoba_contact"].iloc[0]
            n = h_row["bip"].iloc[0] if "bip" in h_row.columns else 0
            if pd.notna(val) and pd.notna(n) and n >= 10:
                h_xwoba_con = val
                h_bip = int(n)

        h_hh = np.nan
        if len(s_row) > 0 and "hard_hit_rate" in s_row.columns:
            h_hh = s_row["hard_hit_rate"].iloc[0]

        # --- Classify this pitch for the batter ---
        # Odds-ratio matchup xwOBA for this pitch type
        _clamp = lambda x: max(0.01, min(0.99, x))
        p_w = _clamp(p_whiff)
        b_w = _clamp(h_whiff)
        lg_w = _clamp(lg_whiff)
        matchup_whiff = (p_w * b_w / lg_w) / (
            p_w * b_w / lg_w + (1 - p_w) * (1 - b_w) / (1 - lg_w)
        )

        _scale = 0.7
        p_x = _clamp(p_xwoba / _scale)
        b_x = _clamp(h_xwoba_con / _scale)
        lg_x = _clamp(lg_xwoba_con / _scale)
        matchup_xwoba_con = (p_x * b_x / lg_x) / (
            p_x * b_x / lg_x + (1 - p_x) * (1 - b_x) / (1 - lg_x)
        ) * _scale

        p_contact = 1.0 - matchup_whiff
        pitch_xwoba = p_contact * matchup_xwoba_con

        # Weak contact flag: batter makes contact but it's poor quality
        weak_contact = (matchup_whiff < 0.22 and matchup_xwoba_con < 0.270)

        # Determine approach tag
        if pitch_xwoba >= LG_XWOBA + 0.040:
            approach = "hunt"       # batter crushes this
        elif pitch_xwoba >= LG_XWOBA + 0.015:
            approach = "aggressive" # batter has edge
        elif pitch_xwoba <= LG_XWOBA - 0.040:
            approach = "avoid"      # pitcher's weapon
        elif pitch_xwoba <= LG_XWOBA - 0.015:
            approach = "defensive"  # pitcher has edge
        else:
            approach = "neutral"

        # Generate approach recommendation text
        rec_parts: list[str] = []
        if approach == "hunt":
            rec_parts.append("Sit on this pitch")
            if h_xwoba_con >= 0.400 and h_bip >= 20:
                rec_parts.append(f"does damage on contact (.{int(h_xwoba_con*1000):03d})")
            if h_whiff < lg_whiff * 0.75 and h_swings >= 20:
                rec_parts.append(f"rarely whiffs ({h_whiff*100:.0f}%)")
        elif approach == "aggressive":
            rec_parts.append("Look to drive")
            if h_xwoba_con >= 0.380 and h_bip >= 20:
                rec_parts.append(f"solid contact quality (.{int(h_xwoba_con*1000):03d})")
        elif approach == "avoid":
            if matchup_whiff >= 0.35:
                rec_parts.append("Take unless 2-strike")
            else:
                rec_parts.append("Lay off out of zone")
            if h_whiff > lg_whiff * 1.20 and h_swings >= 20:
                rec_parts.append(f"high whiff rate ({h_whiff*100:.0f}%)")
            if h_chase > lg_chase * 1.20 and h_oz >= 20:
                rec_parts.append(f"chases ({h_chase*100:.0f}%)")
        elif approach == "defensive":
            rec_parts.append("Be selective")
            if h_chase > lg_chase * 1.10 and h_oz >= 20:
                rec_parts.append(f"watch chase tendency ({h_chase*100:.0f}%)")
        else:
            rec_parts.append("Neutral matchup")

        recommendation = ". ".join(rec_parts) if rec_parts else ""

        # Weak contact annotations
        if weak_contact:
            if approach == "neutral":
                recommendation = (
                    f"Contact likely but weak ({matchup_xwoba_con:.3f} xwOBA). "
                    f"Lay off in hitter's counts."
                )
            elif approach in ("defensive", "avoid"):
                recommendation += ". Weak contact out likely"
            elif approach in ("hunt", "aggressive"):
                recommendation += (
                    f". Contact quality is weak ({matchup_xwoba_con:.3f}), "
                    f"look to drive the count"
                )

        pitch_plans.append({
            "pitch_type": pt,
            "pitch_name": pt_name,
            "usage": usage,
            "approach": approach,
            "recommendation": recommendation,
            "matchup_xwoba": pitch_xwoba,
            "matchup_whiff": matchup_whiff,
            "matchup_xwoba_contact": matchup_xwoba_con,
            "weak_contact": weak_contact,
            "pitcher_whiff": p_whiff,
            "hitter_whiff": h_whiff,
            "hitter_chase": h_chase,
            "hitter_xwoba_contact": h_xwoba_con,
            "hitter_bip": h_bip,
            "velo": p_velo if pd.notna(p_velo) else None,
        })

    if not pitch_plans:
        return empty

    # Sort by usage
    pitch_plans.sort(key=lambda x: x["usage"], reverse=True)

    # --- Usage tiers ---
    # Primary (>=10%): pitches the batter should game-plan around
    # Secondary (5-10%): worth knowing, shown below threshold line
    # Rare (<5%): mentioned but not actionable
    PRIMARY_THRESH = 0.10
    SECONDARY_THRESH = 0.05

    for p in pitch_plans:
        if p["usage"] >= PRIMARY_THRESH:
            p["tier"] = "primary"
        elif p["usage"] >= SECONDARY_THRESH:
            p["tier"] = "secondary"
        else:
            p["tier"] = "rare"

    primary = [p for p in pitch_plans if p["tier"] == "primary"]

    # --- Limit hunt pitches ---
    # A batter can't "sit on" every pitch. Among primary pitches,
    # allow at most 2 "hunt" tags. The rest get demoted to "aggressive".
    # Also: only the best xwOBA primary pitch gets "hunt" if there are 3+
    # primary pitches tagged hunt/aggressive.
    hunt_primary = sorted(
        [p for p in primary if p["approach"] == "hunt"],
        key=lambda x: x["matchup_xwoba"], reverse=True,
    )
    if len(hunt_primary) > 2:
        for p in hunt_primary[2:]:
            p["approach"] = "aggressive"
            # Re-generate recommendation
            parts = ["Look to drive"]
            if p["hitter_xwoba_contact"] >= 0.380 and p["hitter_bip"] >= 20:
                parts.append(f"solid contact quality (.{int(p['hitter_xwoba_contact']*1000):03d})")
            p["recommendation"] = ". ".join(parts)

    # Secondary/rare pitches that are "hunt" get demoted to "aggressive"
    for p in pitch_plans:
        if p["tier"] != "primary" and p["approach"] == "hunt":
            p["approach"] = "aggressive"
            parts = ["Look to drive if it's there"]
            if p["hitter_xwoba_contact"] >= 0.380 and p["hitter_bip"] >= 20:
                parts.append(f"contact quality (.{int(p['hitter_xwoba_contact']*1000):03d})")
            p["recommendation"] = ". ".join(parts)

    # --- Overall matchup xwOBA ---
    total_usage = sum(p["usage"] for p in pitch_plans)
    matchup_xwoba = (
        sum(p["usage"] * p["matchup_xwoba"] for p in pitch_plans) / total_usage
        if total_usage > 0 else LG_XWOBA
    )

    # Identify best and worst
    hunt_pitch = max(pitch_plans, key=lambda x: x["matchup_xwoba"])
    avoid_pitch = min(pitch_plans, key=lambda x: x["matchup_xwoba"])

    # --- Build game plan summary ---
    # Priority: 1) what to hunt (max 2), 2) what to drive, 3) what to avoid
    hunt_picks = [p for p in primary if p["approach"] == "hunt"]
    agg_picks = [p for p in primary if p["approach"] == "aggressive"]
    avoid_picks = [p for p in primary if p["approach"] in ("avoid", "defensive")]
    neutral_picks = [p for p in primary if p["approach"] == "neutral"]

    summary_parts: list[str] = []

    # Check if everything is hittable (all primary are hunt/aggressive)
    all_hittable = len(avoid_picks) == 0 and len(neutral_picks) == 0
    all_tough = len(hunt_picks) == 0 and len(agg_picks) == 0

    if all_hittable and len(primary) >= 3:
        # Special case: batter has edge on entire arsenal
        best = max(primary, key=lambda x: x["matchup_xwoba"])
        summary_parts.append(
            f"Batter has edge across the arsenal. "
            f"Hunt the {best['pitch_name'].lower()}, be aggressive on everything"
        )
    elif all_tough:
        # Special case: pitcher dominates
        least_bad = max(primary, key=lambda x: x["matchup_xwoba"])
        summary_parts.append(
            f"Tough matchup. Be selective, best chance is the "
            f"{least_bad['pitch_name'].lower()}"
        )
    else:
        # Normal case: mix of edges
        if hunt_picks:
            names = " and ".join(p["pitch_name"].lower() for p in hunt_picks[:2])
            summary_parts.append(f"Hunt the {names}")
        if agg_picks and not hunt_picks:
            names = " and ".join(p["pitch_name"].lower() for p in agg_picks[:2])
            summary_parts.append(f"Look to drive the {names}")
        elif agg_picks and hunt_picks:
            names = " and ".join(p["pitch_name"].lower() for p in agg_picks[:1])
            summary_parts.append(f"drive the {names} if it's there")

        if avoid_picks:
            names = " and ".join(p["pitch_name"].lower() for p in avoid_picks[:2])
            high_whiff = any(p["matchup_whiff"] >= 0.35 for p in avoid_picks)
            if high_whiff:
                summary_parts.append(f"take the {names} unless two strikes")
            else:
                summary_parts.append(f"lay off the {names}")

    if not summary_parts:
        summary_parts.append("No strong pitch-type edges")

    summary = ", ".join(summary_parts) + "."
    summary = summary[0].upper() + summary[1:]

    # Batter hand label for UI context
    batter_label = None
    if batter_hand:
        bh = batter_hand.upper()[0] if batter_hand else None
        if bh == "L":
            batter_label = "LHB"
        elif bh == "R":
            batter_label = "RHB"
        elif bh == "S":
            batter_label = "Switch"

    return {
        "summary": summary,
        "pitch_plans": pitch_plans,
        "platoon": platoon,
        "matchup_xwoba": matchup_xwoba,
        "hunt_pitch": hunt_pitch["pitch_type"] if hunt_pitch else None,
        "avoid_pitch": avoid_pitch["pitch_type"] if avoid_pitch else None,
        "batter_label": batter_label,
        "primary_threshold": PRIMARY_THRESH,
        "secondary_threshold": SECONDARY_THRESH,
    }


# ---------------------------------------------------------------------------
# Comp-based proxy data for MiLB callups
# ---------------------------------------------------------------------------


def build_comp_proxy_data(
    batter_id: int,
    comps_df: pd.DataFrame,
    all_vuln_df: pd.DataFrame,
    all_str_df: pd.DataFrame,
    milb_priors_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """Build synthetic vuln/str DataFrames from prospect MLB comps.

    For a MiLB callup with no MLB pitch-type data, uses their top
    prospect comps' career pitch-type profiles as a proxy.  Rates are
    similarity-weighted across comps, with synthetic sample sizes that
    cause ~30% regression toward league average in the matchup engine.

    Parameters
    ----------
    batter_id : int
        The prospect's player_id.
    comps_df : pd.DataFrame
        Full ``prospect_comps_batters.parquet`` (all prospects).
    all_vuln_df, all_str_df : pd.DataFrame
        Full career vulnerability / strength parquets (all MLB batters).
    milb_priors_df : pd.DataFrame | None
        ``milb_priors.parquet`` for rate projections (optional).

    Returns
    -------
    (vuln_df, str_df, comp_info) — synthetic data shaped like the real
    parquets, plus a dict of comp metadata.  Returns (empty, empty, None)
    when no usable comps exist.
    """
    if comps_df.empty or "player_id" not in comps_df.columns:
        return pd.DataFrame(), pd.DataFrame(), None

    player_comps = comps_df[comps_df["player_id"] == batter_id].sort_values(
        "similarity_score", ascending=False,
    ).head(3)

    if player_comps.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    comp_ids = player_comps["comp_player_id"].tolist()
    sim_weights = dict(zip(
        player_comps["comp_player_id"],
        player_comps["similarity_score"],
    ))

    # ---- Vulnerability data from comps ----
    if all_vuln_df.empty or "batter_id" not in all_vuln_df.columns:
        return pd.DataFrame(), pd.DataFrame(), None
    cv = all_vuln_df[all_vuln_df["batter_id"].isin(comp_ids)].copy()
    if cv.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    cv["_sim"] = cv["batter_id"].map(sim_weights)

    # Synthetic sample sizes: moderate counts → ~30% regression to lg avg
    # whiff stability ~50 swings → 35 gives 0.70 weight
    # xwOBA stability ~30 BIP  → 20 gives 0.67 weight
    SYNTH_PITCHES = 60
    SYNTH_SWINGS = 35
    SYNTH_OZ = 25
    SYNTH_BIP = 20

    vuln_rows: list[dict] = []
    for pt, g in cv.groupby("pitch_type"):
        w = g["_sim"].values
        w_sum = w.sum()
        if w_sum <= 0:
            continue

        # Similarity-weighted count totals (then convert to rates)
        sw = g["swings"].fillna(0).values
        wh = g["whiffs"].fillna(0).values
        oz = g["out_of_zone_pitches"].fillna(0).values
        cs = g["chase_swings"].fillna(0).values
        pt_count = g["pitches"].fillna(0).values
        csw_count = g["csw"].fillna(0).values
        bip_count = g["bip"].fillna(0).values
        hh_count = g["hard_hits"].fillna(0).values

        sw_w = (w * sw).sum() / w_sum
        wh_w = (w * wh).sum() / w_sum
        oz_w = (w * oz).sum() / w_sum
        cs_w = (w * cs).sum() / w_sum
        pt_w = (w * pt_count).sum() / w_sum
        csw_w = (w * csw_count).sum() / w_sum
        bip_w = (w * bip_count).sum() / w_sum
        hh_w = (w * hh_count).sum() / w_sum

        whiff_rate = wh_w / sw_w if sw_w > 10 else np.nan
        chase_rate = cs_w / oz_w if oz_w > 10 else np.nan
        csw_pct = csw_w / pt_w if pt_w > 10 else np.nan
        hh_rate = hh_w / bip_w if bip_w > 10 else 0.0

        family = (
            g["pitch_family"].iloc[0]
            if "pitch_family" in g.columns and pd.notna(g["pitch_family"].iloc[0])
            else None
        )

        vuln_rows.append({
            "batter_id": batter_id,
            "batter_stand": "R",  # placeholder — overridden later if known
            "pitch_type": pt,
            "pitch_family": family,
            "pitches": SYNTH_PITCHES,
            "swings": SYNTH_SWINGS,
            "whiffs": int(round(SYNTH_SWINGS * whiff_rate)) if pd.notna(whiff_rate) else 0,
            "out_of_zone_pitches": SYNTH_OZ,
            "chase_swings": int(round(SYNTH_OZ * chase_rate)) if pd.notna(chase_rate) else 0,
            "called_strikes": 0,
            "csw": int(round(SYNTH_PITCHES * csw_pct)) if pd.notna(csw_pct) else 0,
            "bip": SYNTH_BIP,
            "hard_hits": int(round(SYNTH_BIP * hh_rate)),
            "barrels_proxy": 0,
            "whiff_rate": whiff_rate if pd.notna(whiff_rate) else None,
            "chase_rate": chase_rate if pd.notna(chase_rate) else None,
            "csw_pct": csw_pct if pd.notna(csw_pct) else None,
            "xwoba_contact": np.nan,  # filled from str data below
        })

    synth_vuln = pd.DataFrame(vuln_rows) if vuln_rows else pd.DataFrame()

    # ---- Strength / contact quality data from comps ----
    cs_df = all_str_df[all_str_df["batter_id"].isin(comp_ids)].copy()
    str_rows: list[dict] = []
    if not cs_df.empty:
        cs_df["_sim"] = cs_df["batter_id"].map(sim_weights)
        for pt, g in cs_df.groupby("pitch_type"):
            w = g["_sim"].values
            w_sum = w.sum()
            if w_sum <= 0:
                continue

            bip_arr = g["bip"].fillna(0).values
            hh_arr = g["hard_hits"].fillna(0).values
            bar_arr = g["barrels_proxy"].fillna(0).values

            bip_w = (w * bip_arr).sum() / w_sum
            hh_w = (w * hh_arr).sum() / w_sum
            bar_w = (w * bar_arr).sum() / w_sum

            hh_rate = hh_w / bip_w if bip_w > 10 else np.nan
            bar_rate = bar_w / bip_w if bip_w > 10 else np.nan

            # Similarity × BIP–weighted xwOBA on contact
            xwoba_vals = g["xwoba_contact"].values
            valid = pd.notna(xwoba_vals) & (bip_arr > 0)
            if valid.any():
                xw_num = (w[valid] * bip_arr[valid] * xwoba_vals[valid]).sum()
                xw_den = (w[valid] * bip_arr[valid]).sum()
                xwoba_con = xw_num / xw_den if xw_den > 0 else np.nan
            else:
                xwoba_con = np.nan

            family = (
                g["pitch_family"].iloc[0]
                if "pitch_family" in g.columns and pd.notna(g["pitch_family"].iloc[0])
                else None
            )

            str_rows.append({
                "batter_id": batter_id,
                "batter_stand": "R",
                "pitch_type": pt,
                "pitch_family": family,
                "bip": SYNTH_BIP,
                "barrels_proxy": int(round(SYNTH_BIP * bar_rate)) if pd.notna(bar_rate) else 0,
                "hard_hits": int(round(SYNTH_BIP * hh_rate)) if pd.notna(hh_rate) else 0,
                "barrel_rate_contact": bar_rate,
                "hard_hit_rate": hh_rate,
                "xwoba_contact": xwoba_con,
            })

    synth_str = pd.DataFrame(str_rows) if str_rows else pd.DataFrame()

    # Merge xwoba_contact from str into vuln (matchup engine checks both)
    if not synth_str.empty and not synth_vuln.empty:
        xwoba_map = synth_str.set_index("pitch_type")["xwoba_contact"].to_dict()
        synth_vuln["xwoba_contact"] = synth_vuln["pitch_type"].map(xwoba_map)

    # ---- MiLB rate projections (optional enrichment) ----
    milb_rates: dict[str, dict] = {}
    if milb_priors_df is not None and not milb_priors_df.empty:
        player_priors = milb_priors_df[
            (milb_priors_df["player_id"] == batter_id)
            & (milb_priors_df["player_type"] == "batter")
        ]
        for _, row in player_priors.iterrows():
            milb_rates[row["stat"]] = {
                "mean": row["prior_rate_mean"],
                "p10": row["prior_rate_p10"],
                "p90": row["prior_rate_p90"],
            }

    comp_info = {
        "comp_names": player_comps["comp_name"].tolist(),
        "comp_similarities": player_comps["similarity_score"].tolist(),
        "milb_rates": milb_rates,
    }

    return synth_vuln, synth_str, comp_info


def build_pitcher_comp_arsenal(
    pitcher_id: int,
    comps_df: pd.DataFrame,
    all_arsenal_df: pd.DataFrame,
    milb_priors_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """Build a synthetic pitcher arsenal from prospect pitcher comps.

    When a MiLB callup starts with no MLB Statcast data, uses their top
    prospect comps' arsenal profiles (similarity-weighted) so the matchup
    engine has something to work with.

    Returns
    -------
    (arsenal_df, comp_info) — synthetic arsenal shaped like
    ``pitcher_arsenal.parquet``, plus comp metadata.
    Returns (empty, None) when no usable comps exist.
    """
    if comps_df.empty or "player_id" not in comps_df.columns:
        return pd.DataFrame(), None

    player_comps = comps_df[comps_df["player_id"] == pitcher_id].sort_values(
        "similarity_score", ascending=False,
    ).head(5)

    if player_comps.empty:
        return pd.DataFrame(), None

    comp_ids = player_comps["comp_player_id"].tolist()
    sim_weights = dict(zip(
        player_comps["comp_player_id"],
        player_comps["similarity_score"],
    ))

    if all_arsenal_df.empty or "pitcher_id" not in all_arsenal_df.columns:
        return pd.DataFrame(), None

    ca = all_arsenal_df[all_arsenal_df["pitcher_id"].isin(comp_ids)].copy()
    if ca.empty:
        return pd.DataFrame(), None

    ca["_sim"] = ca["pitcher_id"].map(sim_weights)

    # Infer pitch hand from comps (majority vote)
    pitch_hand = None
    if "pitch_hand" in ca.columns:
        hands = ca.drop_duplicates("pitcher_id")[["pitcher_id", "pitch_hand", "_sim"]]
        if not hands.empty:
            pitch_hand = hands.sort_values("_sim", ascending=False)["pitch_hand"].iloc[0]

    # For each pitch type present in any comp, build similarity-weighted row
    arsenal_rows: list[dict] = []
    for pt, g in ca.groupby("pitch_type"):
        w = g["_sim"].values
        w_sum = w.sum()
        if w_sum <= 0:
            continue

        def _wavg(col: str, default: float = 0.0) -> float:
            vals = g[col].fillna(default).values if col in g.columns else np.full(len(g), default)
            return float((w * vals).sum() / w_sum)

        usage = _wavg("usage_pct")
        if usage < 0.02:  # skip negligible pitch types
            continue

        whiff_rate = _wavg("whiff_rate", 0.25)
        csw_pct = _wavg("csw_pct", 0.28)
        xwoba = _wavg("xwoba_against", 0.315)
        velo = _wavg("avg_velo", 90.0)
        pfx_x = _wavg("avg_pfx_x", 0.0)
        pfx_z = _wavg("avg_pfx_z", 0.0)
        hh_rate = _wavg("hard_hit_rate_against", 0.35)
        barrel_rate = _wavg("barrel_rate_against", 0.06)

        # Synthetic counts (moderate)
        SYNTH_PITCHES = 200
        total_p = int(round(SYNTH_PITCHES * usage))
        if total_p < 5:
            continue

        swing_rate = 0.45  # approximate
        synth_swings = int(round(total_p * swing_rate))
        synth_bip = int(round(total_p * 0.20))

        family = (
            g["pitch_family"].iloc[0]
            if "pitch_family" in g.columns and pd.notna(g["pitch_family"].iloc[0])
            else None
        )

        arsenal_rows.append({
            "pitcher_id": pitcher_id,
            "pitch_hand": pitch_hand,
            "pitch_type": pt,
            "pitches": total_p,
            "total_pitches": SYNTH_PITCHES,
            "usage_pct": usage,
            "swings": synth_swings,
            "whiffs": int(round(synth_swings * whiff_rate)),
            "called_strikes": 0,
            "csw": int(round(total_p * csw_pct)),
            "bip": synth_bip,
            "barrels_proxy": int(round(synth_bip * barrel_rate)),
            "hard_hits": int(round(synth_bip * hh_rate)),
            "xwoba_against": xwoba,
            "avg_velo": velo,
            "avg_pfx_x": pfx_x,
            "avg_pfx_z": pfx_z,
            "whiff_rate": whiff_rate,
            "csw_pct": csw_pct,
            "barrel_rate_against": barrel_rate,
            "hard_hit_rate_against": hh_rate,
            "pitch_family": family,
            "season": 2026,
        })

    synth_arsenal = pd.DataFrame(arsenal_rows) if arsenal_rows else pd.DataFrame()

    if synth_arsenal.empty:
        return pd.DataFrame(), None

    # Renormalize usage to sum to 1.0
    total_usage = synth_arsenal["usage_pct"].sum()
    if total_usage > 0:
        synth_arsenal["usage_pct"] = synth_arsenal["usage_pct"] / total_usage

    # MiLB rate projections
    milb_rates: dict[str, dict] = {}
    if milb_priors_df is not None and not milb_priors_df.empty:
        player_priors = milb_priors_df[
            (milb_priors_df["player_id"] == pitcher_id)
            & (milb_priors_df["player_type"] == "pitcher")
        ]
        for _, row in player_priors.iterrows():
            milb_rates[row["stat"]] = {
                "mean": row["prior_rate_mean"],
                "p10": row["prior_rate_p10"],
                "p90": row["prior_rate_p90"],
            }

    # Comp names (only those with arsenal data)
    comps_with_data = [
        cid for cid in comp_ids
        if not all_arsenal_df[all_arsenal_df["pitcher_id"] == cid].empty
    ]
    comp_info = {
        "comp_names": [
            player_comps[player_comps["comp_player_id"] == cid]["comp_name"].iloc[0]
            for cid in comps_with_data
        ],
        "comp_similarities": [
            sim_weights[cid] for cid in comps_with_data
        ],
        "milb_rates": milb_rates,
    }

    return synth_arsenal, comp_info


# ---------------------------------------------------------------------------
# Scouting Card (casual-friendly narrative report)
# ---------------------------------------------------------------------------

def _grade_label(grade: float) -> str:
    """Map a 20-80 scouting grade to a plain-english label."""
    rounded = int(round(grade / 5) * 5)
    return GRADE_LABELS.get(rounded, "average")


def _family_name(family: str) -> str:
    return PITCH_FAMILY_DISPLAY.get(family, family)


def generate_scouting_card(
    player_id: int,
    player_type: str,
    player_row: pd.Series,
    trad_row: pd.Series | None,
    counting_row: pd.Series | None,
    stat_configs: list[tuple[str, str, bool, str]],
    all_df: pd.DataFrame,
    *,
    archetype_df: pd.DataFrame | None = None,
    grade_df: pd.DataFrame | None = None,
    str_df: pd.DataFrame | None = None,
    vuln_df: pd.DataFrame | None = None,
    arsenal_df: pd.DataFrame | None = None,
    adv_df: pd.DataFrame | None = None,
    breakout_df: pd.DataFrame | None = None,
) -> dict:
    """Build a casual-friendly scouting card from available data.

    Returns dict with keys: one_liner, strengths, weaknesses, outlook,
    development.
    """
    is_hitter = player_type in ("Hitter", "Two-Way")
    id_col = "batter_id" if is_hitter else "pitcher_id"

    card: dict = {
        "one_liner": "",
        "strengths": [],
        "weaknesses": [],
        "outlook": [],
        "development": None,
    }

    # ── ONE-LINER ────────────────────────────────────────────────────
    desc = ""
    if archetype_df is not None and not archetype_df.empty:
        arch_match = archetype_df[archetype_df[id_col] == player_id]
        if not arch_match.empty:
            desc = str(arch_match.iloc[0].get("archetype_desc", ""))

    grade_row = None
    if grade_df is not None and not grade_df.empty:
        gm = grade_df[grade_df["player_id"] == player_id]
        if not gm.empty:
            grade_row = gm.iloc[0]

    if desc:
        qualifiers: list[str] = []
        skills = HITTER_GRADE_SKILLS if is_hitter else PITCHER_GRADE_SKILLS
        if grade_row is not None:
            for g_col, g_name in skills:
                g_val = grade_row.get(g_col)
                if pd.notna(g_val) and g_val >= 65:
                    qualifiers.append(f"{_grade_label(g_val)} {g_name}")
        if qualifiers:
            card["one_liner"] = f"{desc}, with {' and '.join(qualifiers[:2])}."
        else:
            card["one_liner"] = f"{desc}."
    elif grade_row is not None:
        # No archetype -- build from grades alone
        skills = HITTER_GRADE_SKILLS if is_hitter else PITCHER_GRADE_SKILLS
        parts = []
        for g_col, g_name in skills:
            g_val = grade_row.get(g_col)
            if pd.notna(g_val) and g_val >= 55:
                parts.append(f"{_grade_label(g_val)} {g_name}")
        if parts:
            card["one_liner"] = f"Player profile features {', '.join(parts)}."

    # ── STRENGTHS ────────────────────────────────────────────────────
    strengths: list[str] = []

    if is_hitter:
        # Pitch-type strengths from career data
        if str_df is not None and not str_df.empty:
            ps = str_df[str_df["batter_id"] == player_id]
            if not ps.empty:
                # Aggregate by pitch_family for reliable signals
                fam_agg = (
                    ps[ps["bip"] >= 10]
                    .groupby("pitch_family")
                    .apply(
                        lambda g: pd.Series({
                            "xwoba": np.average(g["xwoba_contact"], weights=g["bip"]),
                            "bip": g["bip"].sum(),
                        }),
                        include_groups=False,
                    )
                )
                for fam, row in fam_agg[fam_agg["bip"] >= 20].iterrows():
                    if row["xwoba"] >= 0.400:
                        strengths.append(
                            f"Punishes {_family_name(fam)} pitching -- does real damage on contact"
                        )
                    elif row["xwoba"] >= 0.360:
                        strengths.append(
                            f"Handles {_family_name(fam)} pitching well"
                        )

                # Specific standout pitch types (deduplicate by pitch_type)
                standouts = (
                    ps[ps["bip"] >= 15]
                    .sort_values("xwoba_contact", ascending=False)
                    .drop_duplicates(subset=["pitch_type"], keep="first")
                )
                standouts = standouts[standouts["xwoba_contact"] >= 0.450]
                _seen_pts = set()
                for _, sr in standouts.head(2).iterrows():
                    pt_name = PITCH_DISPLAY.get(sr["pitch_type"], sr["pitch_type"])
                    if pt_name not in _seen_pts:
                        strengths.append(f"Especially dangerous against {pt_name}s")
                        _seen_pts.add(pt_name)

        # Grade-based strengths
        if grade_row is not None:
            for g_col, g_name in HITTER_GRADE_SKILLS:
                g_val = grade_row.get(g_col)
                if pd.notna(g_val) and g_val >= 60:
                    strengths.append(f"{_grade_label(g_val).capitalize()} {g_name}")

        # Advanced stat strengths
        if adv_df is not None and not adv_df.empty:
            adv_p = adv_df[adv_df["batter_id"] == player_id]
            if not adv_p.empty:
                ar = adv_p.iloc[0]
                pop = adv_df[adv_df["pa"] >= 50]
                if pd.notna(ar.get("chase_rate")) and len(pop) >= 20:
                    chase_pct = percentile_rank(pop["chase_rate"], ar["chase_rate"], False)
                    if chase_pct >= 75:
                        strengths.append("Excellent plate discipline -- rarely chases pitches outside the zone")
                if pd.notna(ar.get("avg_exit_velo")) and len(pop) >= 20:
                    ev_pct = percentile_rank(pop["avg_exit_velo"], ar["avg_exit_velo"], True)
                    if ev_pct >= 80:
                        strengths.append("Hits the ball extremely hard when he makes contact")

    else:
        # Pitcher strengths from arsenal
        if arsenal_df is not None and not arsenal_df.empty:
            pa = arsenal_df[arsenal_df["pitcher_id"] == player_id]
            pa = pa[pa["pitches"] >= 50] if not pa.empty else pa
            if not pa.empty:
                for _, row in pa.sort_values("whiff_rate", ascending=False).head(2).iterrows():
                    pt_name = PITCH_DISPLAY.get(row["pitch_type"], row["pitch_type"])
                    velo = row.get("avg_velo")
                    velo_tag = f" ({velo:.0f} mph)" if pd.notna(velo) and velo >= 85 else ""
                    if row["whiff_rate"] >= 0.30:
                        strengths.append(
                            f"Devastating {pt_name}{velo_tag} -- generates whiffs at an elite rate"
                        )
                    elif row["whiff_rate"] >= 0.24:
                        strengths.append(
                            f"Effective {pt_name}{velo_tag} with above-average swing-and-miss"
                        )

                # Primary pitch callout if high usage and effective
                primary = pa.sort_values("usage_pct", ascending=False).iloc[0]
                if primary["usage_pct"] >= 0.30:
                    p_name = PITCH_DISPLAY.get(primary["pitch_type"], primary["pitch_type"])
                    p_velo = primary.get("avg_velo")
                    if pd.notna(p_velo) and p_velo >= 94:
                        strengths.append(f"Brings heat -- primary {p_name} averages {p_velo:.0f} mph")

        if grade_row is not None:
            for g_col, g_name in PITCHER_GRADE_SKILLS:
                g_val = grade_row.get(g_col)
                if pd.notna(g_val) and g_val >= 60:
                    strengths.append(f"{_grade_label(g_val).capitalize()} {g_name}")

        if adv_df is not None and not adv_df.empty:
            adv_p = adv_df[adv_df["pitcher_id"] == player_id]
            if not adv_p.empty:
                ar = adv_p.iloc[0]
                pop = adv_df[adv_df["batters_faced"] >= 50]
                if pd.notna(ar.get("chase_pct")) and len(pop) >= 20:
                    chase_pct = percentile_rank(pop["chase_pct"], ar["chase_pct"], True)
                    if chase_pct >= 75:
                        strengths.append("Gets hitters to expand the zone consistently")
                if pd.notna(ar.get("gb_pct")) and len(pop) >= 20:
                    gb_pct = percentile_rank(pop["gb_pct"], ar["gb_pct"], True)
                    if gb_pct >= 75:
                        strengths.append("Elite ground ball rate -- keeps the ball on the ground")

    card["strengths"] = strengths[:5]

    # ── WEAKNESSES ───────────────────────────────────────────────────
    weaknesses: list[str] = []

    if is_hitter:
        if vuln_df is not None and not vuln_df.empty:
            pv = vuln_df[vuln_df["batter_id"] == player_id]
            pv = pv[pv["pitches"] >= 50] if not pv.empty else pv
            if not pv.empty:
                fam_vuln = (
                    pv.groupby("pitch_family")
                    .apply(
                        lambda g: pd.Series({
                            "whiff": np.average(g["whiff_rate"], weights=g["pitches"]),
                            "chase": np.average(g["chase_rate"], weights=g["pitches"]),
                            "pitches": g["pitches"].sum(),
                        }),
                        include_groups=False,
                    )
                )
                for fam, row in fam_vuln.iterrows():
                    if row["pitches"] < 50:
                        continue
                    if row["whiff"] >= 0.30:
                        weaknesses.append(
                            f"{_family_name(fam).capitalize()} pitching gives him trouble -- high swing-and-miss rate"
                        )
                    elif row["chase"] >= 0.35:
                        weaknesses.append(
                            f"Tends to chase {_family_name(fam)} pitches outside the zone"
                        )

                # Specific pitch-type callouts
                worst_pts = (
                    pv[pv["pitches"] >= 40]
                    .sort_values("whiff_rate", ascending=False)
                    .drop_duplicates(subset=["pitch_type"], keep="first")
                )
                worst_pts = worst_pts[worst_pts["whiff_rate"] >= 0.32].head(2)
                for _, wr in worst_pts.iterrows():
                    pt_name = PITCH_DISPLAY.get(wr["pitch_type"], wr["pitch_type"])
                    weaknesses.append(
                        f"Struggles most against {pt_name}s"
                    )

        if grade_row is not None:
            for g_col, g_name in HITTER_GRADE_SKILLS:
                g_val = grade_row.get(g_col)
                if pd.notna(g_val) and g_val <= 40:
                    weaknesses.append(f"Limited {g_name}")

        if adv_df is not None and not adv_df.empty:
            adv_p = adv_df[adv_df["batter_id"] == player_id]
            if not adv_p.empty:
                ar = adv_p.iloc[0]
                pop = adv_df[adv_df["pa"] >= 50]
                if pd.notna(ar.get("whiff_rate")) and len(pop) >= 20:
                    whiff_pct = percentile_rank(pop["whiff_rate"], ar["whiff_rate"], False)
                    if whiff_pct <= 25:
                        weaknesses.append("Swing-and-miss is a concern -- whiff rate is high")

    else:
        if arsenal_df is not None and not arsenal_df.empty:
            pa = arsenal_df[arsenal_df["pitcher_id"] == player_id]
            pa = pa[pa["pitches"] >= 50] if not pa.empty else pa
            if not pa.empty:
                # Pitches that get hit hard
                for _, row in pa.sort_values("xwoba_against", ascending=False).head(2).iterrows():
                    if pd.notna(row.get("xwoba_against")) and row["xwoba_against"] >= 0.370:
                        pt_name = PITCH_DISPLAY.get(row["pitch_type"], row["pitch_type"])
                        weaknesses.append(f"{pt_name} gets hit hard -- hitters do damage against it")

                # Low-whiff pitches that get heavy usage (hittable)
                hittable = pa[(pa["usage_pct"] >= 0.15) & (pa["whiff_rate"] < 0.18)]
                for _, row in hittable.head(1).iterrows():
                    pt_name = PITCH_DISPLAY.get(row["pitch_type"], row["pitch_type"])
                    weaknesses.append(
                        f"{pt_name} doesn't miss bats -- hitters put it in play consistently"
                    )

        if grade_row is not None:
            for g_col, g_name in PITCHER_GRADE_SKILLS:
                g_val = grade_row.get(g_col)
                if pd.notna(g_val) and g_val <= 45:
                    weaknesses.append(f"{g_name.capitalize()} can be inconsistent")

        if adv_df is not None and not adv_df.empty:
            adv_p = adv_df[adv_df["pitcher_id"] == player_id]
            if not adv_p.empty:
                ar = adv_p.iloc[0]
                pop = adv_df[adv_df["batters_faced"] >= 50]
                if pd.notna(ar.get("hard_hit_pct_against")) and len(pop) >= 20:
                    hh_pct = percentile_rank(pop["hard_hit_pct_against"], ar["hard_hit_pct_against"], False)
                    if hh_pct <= 25:
                        weaknesses.append("Gives up hard contact at an above-average rate")

    card["weaknesses"] = weaknesses[:5]

    # Cross-reference: if a specific pitch type appears in both strengths and
    # weaknesses, keep the weakness (more actionable for casual fans)
    if card["strengths"] and card["weaknesses"]:
        if is_hitter:
            _weak_families = set()
            for w in card["weaknesses"]:
                wl = w.lower()
                for fam in ("fastball", "breaking ball", "offspeed"):
                    if fam in wl:
                        _weak_families.add(fam)
            if _weak_families:
                card["strengths"] = [
                    s for s in card["strengths"]
                    if not any(fam in s.lower() for fam in _weak_families)
                ]
        else:
            # For pitchers: remove strength bullets for pitches that also show
            # as weaknesses (e.g., 4-Seam has high whiff but also gets hit hard)
            _weak_pitches = set()
            for w in card["weaknesses"]:
                for pt_name in PITCH_DISPLAY.values():
                    if pt_name in w:
                        _weak_pitches.add(pt_name)
            if _weak_pitches:
                card["strengths"] = [
                    s for s in card["strengths"]
                    if not any(pt in s for pt in _weak_pitches)
                ]

    # ── OUTLOOK ──────────────────────────────────────────────────────
    outlook: list[str] = []

    # EOS pace from counting sim
    if counting_row is not None:
        pace_parts = []
        if is_hitter:
            for label, prefix in [("HR", "total_hr"), ("R", "total_r"), ("RBI", "total_rbi")]:
                mean_col = f"{prefix}_mean"
                if mean_col in counting_row.index and pd.notna(counting_row.get(mean_col)):
                    pace_parts.append(f"{int(round(counting_row[mean_col]))} {label}")
        else:
            for label, prefix in [("K", "total_k"), ("IP", "projected_ip")]:
                mean_col = f"{prefix}_mean"
                if mean_col in counting_row.index and pd.notna(counting_row.get(mean_col)):
                    pace_parts.append(f"{int(round(counting_row[mean_col]))} {label}")
        if pace_parts:
            outlook.append(f"Projected for {', '.join(pace_parts)} by season end.")

    # Bayesian rate direction
    for label, key, higher_better, _desc in stat_configs:
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        if obs_col not in player_row.index or pd.isna(player_row.get(obs_col)):
            continue
        observed = player_row[obs_col]
        projected = player_row[proj_col]
        delta_pp = (projected - observed) * 100

        ci_lo = player_row.get(f"projected_{key}_2_5", projected)
        ci_hi = player_row.get(f"projected_{key}_97_5", projected)
        ci_width = (ci_hi - ci_lo) * 100

        if ci_width < 6:
            conf = "Model confidence is high."
        elif ci_width < 12:
            conf = "Moderate confidence in this projection."
        else:
            conf = "Wide range of outcomes possible."

        obs_s = fmt_stat(observed, key)
        proj_s = fmt_stat(projected, key)

        if abs(delta_pp) < 0.5:
            outlook.append(f"{label} projected to hold steady at {proj_s}. {conf}")
        elif abs(delta_pp) > 3:
            direction = "up" if delta_pp > 0 else "down"
            outlook.append(f"{label} moving {direction} from {obs_s} to {proj_s}. {conf}")
        else:
            direction = "ticking up" if delta_pp > 0 else "trending down slightly"
            outlook.append(f"{label} {direction} from {obs_s} to {proj_s}. {conf}")

    card["outlook"] = outlook

    # ── DEVELOPMENT ──────────────────────────────────────────────────
    if breakout_df is not None and not breakout_df.empty:
        bo = breakout_df[breakout_df["batter_id"] == player_id] if is_hitter else pd.DataFrame()
        if not bo.empty:
            bo_row = bo.iloc[0]
            if pd.notna(bo_row.get("breakout_tier")) and bo_row["breakout_tier"]:
                narrative = bo_row.get("breakout_narrative", "")
                if narrative:
                    card["development"] = str(narrative)

    return card


def render_scouting_card(card: dict) -> None:
    """Render the scouting card as styled HTML in Streamlit."""
    parts: list[str] = []

    # One-liner
    if card.get("one_liner"):
        parts.append(
            f'<div style="font-size:1rem; font-weight:600; color:{CREAM}; '
            f'margin-bottom:12px; line-height:1.4;">{card["one_liner"]}</div>'
        )

    # Strengths / Weaknesses side by side
    has_strengths = bool(card.get("strengths"))
    has_weaknesses = bool(card.get("weaknesses"))

    if has_strengths or has_weaknesses:
        cols_html = '<div style="display:flex; gap:24px; margin-bottom:12px;">'

        if has_strengths:
            s_bullets = "".join(
                f'<li style="margin-bottom:4px; color:{CREAM};">{b}</li>'
                for b in card["strengths"]
            )
            cols_html += (
                f'<div style="flex:1;">'
                f'<div style="color:{SAGE}; font-weight:600; font-size:0.7rem; '
                f'letter-spacing:1px; margin-bottom:4px;">STRENGTHS</div>'
                f'<ul style="margin:0; padding-left:1.2rem; font-size:0.85rem;">{s_bullets}</ul>'
                f'</div>'
            )

        if has_weaknesses:
            w_bullets = "".join(
                f'<li style="margin-bottom:4px; color:{CREAM};">{b}</li>'
                for b in card["weaknesses"]
            )
            cols_html += (
                f'<div style="flex:1;">'
                f'<div style="color:{EMBER}; font-weight:600; font-size:0.7rem; '
                f'letter-spacing:1px; margin-bottom:4px;">WEAKNESSES</div>'
                f'<ul style="margin:0; padding-left:1.2rem; font-size:0.85rem;">{w_bullets}</ul>'
                f'</div>'
            )

        cols_html += '</div>'
        parts.append(cols_html)

    # Outlook
    if card.get("outlook"):
        o_text = " ".join(card["outlook"])
        parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<div style="color:{GOLD}; font-weight:600; font-size:0.7rem; '
            f'letter-spacing:1px; margin-bottom:4px;">SEASON OUTLOOK</div>'
            f'<div style="font-size:0.85rem; color:{CREAM}; line-height:1.5;">{o_text}</div>'
            f'</div>'
        )

    # Development watch
    if card.get("development"):
        parts.append(
            f'<div>'
            f'<div style="color:{GOLD}; font-weight:600; font-size:0.7rem; '
            f'letter-spacing:1px; margin-bottom:4px;">DEVELOPMENT WATCH</div>'
            f'<div style="font-size:0.85rem; color:{CREAM}; line-height:1.5; '
            f'font-style:italic;">{card["development"]}</div>'
            f'</div>'
        )

    if parts:
        st.markdown(
            f'<div style="margin:0.5rem 0 1rem; padding:12px 16px; '
            f'border-left:3px solid {GOLD}; background:rgba(200,169,110,0.04);">'
            f'<div style="color:{GOLD}; font-weight:600; font-size:0.75rem; '
            f'letter-spacing:1px; margin-bottom:8px;">SCOUTING REPORT</div>'
            + "".join(parts)
            + '</div>',
            unsafe_allow_html=True,
        )


def render_scouting_html(report) -> None:
    """Render a PitcherReport's scouting bullets as styled HTML."""
    sections = [
        ("ADVANTAGES", EMBER, report.advantages),
        ("STRUGGLES", SAGE, report.struggles),
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


# ---------------------------------------------------------------------------
# Walk Strategy Assessment
# ---------------------------------------------------------------------------


def assess_walk_strategy(
    pitcher_id: int,
    proj_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    form: dict | None = None,
    platoon_bb: dict[str, float] | None = None,
) -> dict | None:
    """Assess walk tendency against a pitcher.

    Uses a two-tier threshold with platoon-aware observations:
      - Tier 1 (strong): season BB% >= 12% OR recent BB/9 >= 4.0
      - Tier 2 (moderate): season BB% >= 9.5% OR recent BB/9 >= 3.5
    Platoon splits override when one hand is significantly more walk-prone.

    Parameters
    ----------
    pitcher_id : int
        The opposing pitcher's ID.
    proj_df : pd.DataFrame
        Pitcher projections with ``projected_bb_rate`` column.
    adv_df : pd.DataFrame
        Pitcher advanced stats with ``zone_pct``, ``bb_pct`` columns.
    form : dict | None
        Recent form from ``get_pitcher_recent_form()`` — used for ``bb_per_9``.
    platoon_bb : dict | None
        BB rates by batter hand, e.g. ``{"L": 0.213, "R": 0.108}``.

    Returns
    -------
    dict | None
        Walk strategy recommendation, or None if not noteworthy.
    """
    if proj_df.empty and adv_df.empty:
        return None

    bb_rate_obs = None
    zone_pct = None

    # Observed / current-season BB rate + zone%
    if not adv_df.empty:
        row = adv_df[adv_df["pitcher_id"] == pitcher_id]
        if not row.empty:
            if "zone_pct" in adv_df.columns:
                val = row["zone_pct"].iloc[0]
                if pd.notna(val):
                    zone_pct = float(val)
            if "bb_pct" in adv_df.columns:
                val = row["bb_pct"].iloc[0]
                if pd.notna(val):
                    bb_rate_obs = float(val)

    # Recent form BB/9 (last 5 starts)
    recent_bb9 = form.get("bb_per_9") if form else None

    # --- Platoon-aware assessment ---
    if platoon_bb:
        lhb_bb = platoon_bb.get("L", 0)
        rhb_bb = platoon_bb.get("R", 0)

        # Strong platoon split: one hand >= 12%, other < 9.5%
        lhb_high = lhb_bb >= 0.12
        rhb_high = rhb_bb >= 0.12
        lhb_moderate = lhb_bb >= 0.095
        rhb_moderate = rhb_bb >= 0.095

        lhb_pa = platoon_bb.get("L_pa")
        rhb_pa = platoon_bb.get("R_pa")
        lhb_n = f" ({lhb_pa} PA)" if lhb_pa else ""
        rhb_n = f" ({rhb_pa} PA)" if rhb_pa else ""

        parts = []
        if lhb_high or lhb_moderate:
            verb = "Patient approach" if lhb_high else "Extended ABs rewarded"
            parts.append(
                f"LHB: {verb} -- walks {lhb_bb*100:.0f}% of lefties{lhb_n}"
            )
        if rhb_high or rhb_moderate:
            verb = "Patient approach" if rhb_high else "Extended ABs rewarded"
            parts.append(
                f"RHB: {verb} -- walks {rhb_bb*100:.0f}% of righties{rhb_n}"
            )

        # Add contrast when one side is low
        if (lhb_high or lhb_moderate) and not rhb_moderate:
            parts.append(f"RHB: swing away -- only walks {rhb_bb*100:.0f}% of righties{rhb_n}")
        elif (rhb_high or rhb_moderate) and not lhb_moderate:
            parts.append(f"LHB: swing away -- only walks {lhb_bb*100:.0f}% of lefties{lhb_n}")

        if parts:
            return {
                "bb_rate": bb_rate_obs,
                "zone_pct": zone_pct,
                "platoon_bb": platoon_bb,
                "note": " | ".join(parts),
            }

    # --- Fallback: overall assessment (no platoon data) ---
    bb_rate = bb_rate_obs

    # Also check recent form
    recent_high = recent_bb9 is not None and recent_bb9 >= 4.0
    recent_moderate = recent_bb9 is not None and recent_bb9 >= 3.5

    season_high = bb_rate is not None and bb_rate >= 0.12
    season_moderate = bb_rate is not None and bb_rate >= 0.095

    if not (season_high or season_moderate or recent_high or recent_moderate):
        return None

    # Build note
    if season_high or recent_high:
        prefix = "Patient approach"
    else:
        prefix = "Extended ABs rewarded"

    note_parts = []
    if bb_rate is not None and (season_high or season_moderate):
        note_parts.append(f"walks {bb_rate*100:.1f}% of batters")
    if recent_high or recent_moderate:
        note_parts.append(f"{recent_bb9:.1f} BB/9 in last 5 starts")
    if zone_pct is not None and zone_pct < 0.42:
        note_parts.append(f"low zone rate ({zone_pct*100:.0f}%)")

    note = f"{prefix} — {', '.join(note_parts)}."

    return {
        "bb_rate": bb_rate,
        "zone_pct": zone_pct,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Pitcher BB by Batter Hand
# ---------------------------------------------------------------------------


def get_pitcher_platoon_bb(
    pitcher_id: int,
    platoon_bb_df: pd.DataFrame | None = None,
) -> dict[str, float] | None:
    """Look up pitcher's BB rate split by batter hand.

    Reads from precomputed ``pitcher_platoon_bb.parquet``.

    Returns
    -------
    dict | None
        e.g. ``{"L": 0.213, "R": 0.108}`` or None if insufficient data.
    """
    if platoon_bb_df is None or platoon_bb_df.empty:
        return None

    row = platoon_bb_df[platoon_bb_df["pitcher_id"] == pitcher_id]
    if row.empty:
        return None

    result: dict[str, float] = {}
    lhb = row["bb_rate_vs_lhb"].iloc[0]
    rhb = row["bb_rate_vs_rhb"].iloc[0]
    if pd.notna(lhb):
        result["L"] = float(lhb)
    if pd.notna(rhb):
        result["R"] = float(rhb)

    # Include PA counts for sample size context
    lhb_pa = row["bf_vs_lhb"].iloc[0] if "bf_vs_lhb" in row.columns else None
    rhb_pa = row["bf_vs_rhb"].iloc[0] if "bf_vs_rhb" in row.columns else None
    if pd.notna(lhb_pa):
        result["L_pa"] = int(lhb_pa)
    if pd.notna(rhb_pa):
        result["R_pa"] = int(rhb_pa)

    return result if result else None


# ---------------------------------------------------------------------------
# Pitcher Recent Form Helper
# ---------------------------------------------------------------------------


def get_pitcher_recent_form(
    pitcher_id: int,
    game_logs_df: pd.DataFrame,
    n_starts: int = 5,
) -> dict | None:
    """Summarize a pitcher's recent starting performance.

    Parameters
    ----------
    pitcher_id : int
        The pitcher to look up.
    game_logs_df : pd.DataFrame
        Full pitcher game logs with columns: pitcher_id, is_starter,
        game_pk, innings_pitched, strike_outs, walks, earned_runs,
        number_of_pitches.
    n_starts : int
        Number of recent starts to summarize.

    Returns
    -------
    dict | None
        Summary stats, or None if fewer than 2 starts found.
    """
    if game_logs_df.empty:
        return None

    starts = game_logs_df[
        (game_logs_df["pitcher_id"] == pitcher_id)
        & (game_logs_df["is_starter"] == True)  # noqa: E712
    ].sort_values("game_pk", ascending=False).head(n_starts)

    if len(starts) < 2:
        return None

    total_ip = starts["innings_pitched"].sum()
    if total_ip <= 0:
        return None

    n = len(starts)
    avg_ip = total_ip / n
    k_per_9 = starts["strike_outs"].sum() * 9 / total_ip
    bb_per_9 = starts["walks"].sum() * 9 / total_ip
    era = starts["earned_runs"].sum() * 9 / total_ip
    avg_pitches = starts["number_of_pitches"].mean()

    return {
        "n_starts": n,
        "avg_ip": round(avg_ip, 1),
        "k_per_9": round(k_per_9, 1),
        "bb_per_9": round(bb_per_9, 1),
        "era": round(era, 2),
        "avg_pitches": round(avg_pitches, 0),
    }
