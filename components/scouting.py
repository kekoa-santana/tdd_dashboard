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
