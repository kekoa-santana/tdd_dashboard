"""Scouting report bullet generators and renderers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, POSITIVE, NEGATIVE,
    PITCH_DISPLAY,
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
      SAGE  -- detail favors the pitcher (high whiff, high chase, weak contact)
      EMBER -- detail favors the hitter  (low whiff, hard contact, low chase)
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
        if len(h_row) > 0 and "swings" in h_row.columns and "whiffs" in h_row.columns:
            sw = h_row["swings"].iloc[0]
            wh = h_row["whiffs"].iloc[0]
            h_whiff = wh / sw if pd.notna(sw) and sw > 0 else np.nan

        s_row = s_df[s_df["pitch_type"] == pt] if not s_df.empty else pd.DataFrame()
        h_xwoba = np.nan
        if len(s_row) > 0 and "xwoba_contact" in s_row.columns:
            h_xwoba = s_row["xwoba_contact"].iloc[0]
        elif len(h_row) > 0 and "xwoba_contact" in h_row.columns:
            h_xwoba = h_row["xwoba_contact"].iloc[0]

        h_chase = np.nan
        if len(h_row) > 0 and "chase_swings" in h_row.columns and "out_of_zone_pitches" in h_row.columns:
            cs = h_row["chase_swings"].iloc[0]
            oz = h_row["out_of_zone_pitches"].iloc[0]
            h_chase = cs / oz if pd.notna(oz) and oz > 0 else np.nan

        h_hh = np.nan
        if len(s_row) > 0 and "hard_hit_rate" in s_row.columns:
            h_hh = s_row["hard_hit_rate"].iloc[0]

        # Collect notable facts (color, text) -- only include if meaningfully
        # above or below league average
        pitcher_facts: list[tuple[str, str]] = []
        hitter_facts: list[tuple[str, str]] = []

        if pd.notna(p_whiff):
            if p_whiff > lg_whiff * 1.15:
                pitcher_facts.append((SAGE, f"{p_whiff*100:.0f}% whiff rate"))
            elif p_whiff < lg_whiff * 0.80:
                pitcher_facts.append((EMBER, f"low {p_whiff*100:.0f}% whiff rate"))

        if pd.notna(h_whiff):
            if h_whiff > lg_whiff * 1.20:
                hitter_facts.append((SAGE, f"hitter whiffs {h_whiff*100:.0f}%"))
            elif h_whiff < lg_whiff * 0.75:
                hitter_facts.append((EMBER, f"hitter rarely whiffs ({h_whiff*100:.0f}%)"))

        if pd.notna(h_chase):
            if h_chase > lg_chase * 1.20:
                hitter_facts.append((SAGE, f"chases {h_chase*100:.0f}%"))
            elif h_chase < lg_chase * 0.75:
                hitter_facts.append((EMBER, f"disciplined ({h_chase*100:.0f}% chase)"))

        if pd.notna(h_xwoba):
            if h_xwoba >= 0.380:
                hitter_facts.append((EMBER, f"does damage on contact (.{int(h_xwoba*1000):03d} xwOBA)"))
            elif h_xwoba <= 0.270:
                hitter_facts.append((SAGE, f"weak contact (.{int(h_xwoba*1000):03d} xwOBA)"))

        if pd.notna(h_hh):
            if h_hh > 0.42:
                hitter_facts.append((EMBER, f"{h_hh*100:.0f}% hard hit"))
            elif h_hh < 0.25:
                hitter_facts.append((SAGE, f"soft contact ({h_hh*100:.0f}% hard hit)"))

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


def render_scouting_html(report) -> None:
    """Render a PitcherReport's scouting bullets as styled HTML."""
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
