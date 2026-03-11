"""Scouting report bullet generators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    GOLD, EMBER, SAGE, SLATE, POSITIVE, NEGATIVE,
    PITCH_DISPLAY,
    STAT_NAMES, GOOD_DIRECTION_LABEL,
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

        stat_name = STAT_NAMES.get(key, label)
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


def build_matchup_scouting_bullets(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
    pitcher_name: str,
    hitter_name: str,
) -> list[tuple[str, str]]:
    """Generate per-pitch scouting bullets for the matchup."""
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

    bullets: list[tuple[str, str]] = []
    pitch_edges: list[tuple[str, float, str, float]] = []

    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)

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

        lg_xwoba = lg.get("xwoba_contact", 0.320)
        lg_hh = lg.get("hard_hit_rate", 0.33)
        lg_chase = lg.get("chase_rate", 0.30)

        h_chase = np.nan
        if len(h_row) > 0 and "chase_swings" in h_row.columns and "out_of_zone_pitches" in h_row.columns:
            cs = h_row["chase_swings"].iloc[0]
            oz = h_row["out_of_zone_pitches"].iloc[0]
            h_chase = cs / oz if pd.notna(oz) and oz > 0 else np.nan

        h_hh = np.nan
        if len(s_row) > 0 and "hard_hit_rate" in s_row.columns:
            h_hh = s_row["hard_hit_rate"].iloc[0]

        edge = 0.0
        detail_parts = []
        if pd.notna(p_whiff):
            edge += (p_whiff - lg_whiff) * 2.0
            if p_whiff > lg_whiff * 1.1:
                detail_parts.append(f"pitcher's whiff rate {p_whiff*100:.0f}%")
        if pd.notna(h_whiff):
            edge += (h_whiff - lg_whiff) * 1.5
            if h_whiff > lg_whiff * 1.15:
                detail_parts.append(f"hitter whiffs {h_whiff*100:.0f}% of the time")
        if pd.notna(h_chase):
            edge += (h_chase - lg_chase) * 1.0
        if pd.notna(h_xwoba):
            edge -= (h_xwoba - lg_xwoba) * 4.0
            if h_xwoba >= 0.400:
                detail_parts.append(f"but hitter does damage on contact (.{int(h_xwoba*1000):03d} xwOBA)")
            elif h_xwoba <= 0.280:
                detail_parts.append(f"weak contact (.{int(h_xwoba*1000):03d} xwOBA)")
        if pd.notna(h_hh):
            edge -= (h_hh - lg_hh) * 2.0

        detail = ", ".join(detail_parts) if detail_parts else ""
        pitch_edges.append((pt_name, edge, detail, usage))

    pitch_edges.sort(key=lambda x: abs(x[1]), reverse=True)

    for pt_name, edge, detail, usage in pitch_edges[:3]:
        if abs(edge) < 0.02:
            continue
        if edge > 0:
            color = SAGE
            direction = "Pitcher advantage"
        else:
            color = EMBER
            direction = "Hitter advantage"

        text = f"<b>{pt_name}</b> ({usage*100:.0f}% usage) — {direction}"
        if detail:
            text += f": {detail}"
        bullets.append((color, text))

    if not bullets:
        bullets.append((GOLD, "No strong pitch-level edges in this matchup"))

    return bullets
