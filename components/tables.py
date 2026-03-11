"""HTML table builders for pitch profiles and matchups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
    PITCH_DISPLAY, PITCH_ORDER, PITCH_TYPE_TO_FAMILY, PITCH_FAMILY_COLORS,
)
from utils.formatters import spark_color_rate, spark_color_xwoba, spark_html


def combine_platoon_vuln(vuln_df: pd.DataFrame) -> pd.DataFrame:
    """Combine L/R platoon splits into a single row per pitch type."""
    sum_cols = [
        "pitches", "swings", "whiffs", "out_of_zone_pitches",
        "chase_swings", "called_strikes", "csw", "bip",
        "hard_hits", "barrels_proxy",
    ]
    agg = {c: "sum" for c in sum_cols if c in vuln_df.columns}
    combined = vuln_df.groupby("pitch_type").agg(agg).reset_index()
    combined["whiff_rate"] = combined["whiffs"] / combined["swings"].replace(0, np.nan)
    combined["chase_rate"] = combined["chase_swings"] / combined["out_of_zone_pitches"].replace(0, np.nan)
    combined["csw_pct"] = combined["csw"] / combined["pitches"].replace(0, np.nan)
    if "pitch_family" in vuln_df.columns:
        fam_map = vuln_df.drop_duplicates("pitch_type").set_index("pitch_type")["pitch_family"].to_dict()
        combined["pitch_family"] = combined["pitch_type"].map(fam_map)
    if "xwoba_contact" in vuln_df.columns:
        xw = vuln_df[vuln_df["xwoba_contact"].notna() & (vuln_df["bip"] > 0)]
        if not xw.empty:
            xw_agg = xw.groupby("pitch_type").apply(
                lambda g: (g["xwoba_contact"] * g["bip"]).sum() / g["bip"].sum()
                if g["bip"].sum() > 0 else np.nan,
                include_groups=False,
            ).reset_index(name="xwoba_contact")
            combined = combined.merge(xw_agg, on="pitch_type", how="left")
        else:
            combined["xwoba_contact"] = np.nan
    combined["batter_id"] = vuln_df["batter_id"].iloc[0]
    return combined


def build_hitter_profile_table(vuln_df: pd.DataFrame) -> str:
    """Build HTML stat table for a hitter's pitch-type profile."""
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    df = vuln_df.copy()
    df = df[df["pitches"] >= 15]
    if df.empty:
        return ""

    df["called_str_rate"] = df["called_strikes"] / df["pitches"].replace(0, np.nan)
    df["whiff_rate_raw"] = df["whiffs"] / df["swings"].replace(0, np.nan)
    df["chase_rate_raw"] = df["chase_swings"] / df["out_of_zone_pitches"].replace(0, np.nan)

    df["_order"] = df["pitch_type"].map({pt: i for i, pt in enumerate(PITCH_ORDER)}).fillna(99)
    df = df.sort_values("_order")

    max_whiff = max(df["whiff_rate_raw"].dropna().max(), 0.01)
    max_cstr = max(df["called_str_rate"].dropna().max(), 0.01)
    max_chase = max(df["chase_rate_raw"].dropna().max(), 0.01)
    max_xwoba = max(df["xwoba_contact"].dropna().max(), 0.01) if "xwoba_contact" in df.columns else 0.5

    rows_html = ""
    for _, row in df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        family = PITCH_TYPE_TO_FAMILY.get(pt, "offspeed")
        family_color = PITCH_FAMILY_COLORS.get(family, SLATE)
        n_pitches = int(row["pitches"])

        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_chase = lg.get("chase_rate", 0.30)

        swings = row.get("swings", 0) or 0
        alpha = min(1.0, 0.45 + 0.55 * (min(swings, 80) / 80))

        whiff = row.get("whiff_rate_raw", np.nan)
        if pd.notna(whiff):
            w_color = spark_color_rate(whiff, lg_whiff, higher_is_worse=True)
            whiff_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(whiff, max_whiff, w_color, alpha)}'
                f'<span class="spark-val" style="color:{w_color};">{whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            whiff_cell = f'<span style="color:{SLATE};">--</span>'

        cstr = row.get("called_str_rate", np.nan)
        if pd.notna(cstr):
            c_color = spark_color_rate(cstr, 0.14, higher_is_worse=True)
            cstr_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(cstr, max_cstr, c_color, alpha)}'
                f'<span class="spark-val" style="color:{c_color};">{cstr*100:.0f}%</span>'
                f'</div>'
            )
        else:
            cstr_cell = f'<span style="color:{SLATE};">--</span>'

        chase = row.get("chase_rate_raw", np.nan)
        if pd.notna(chase):
            ch_color = spark_color_rate(chase, lg_chase, higher_is_worse=True)
            chase_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(chase, max_chase, ch_color, alpha)}'
                f'<span class="spark-val" style="color:{ch_color};">{chase*100:.0f}%</span>'
                f'</div>'
            )
        else:
            chase_cell = f'<span style="color:{SLATE};">--</span>'

        xwoba = row.get("xwoba_contact", np.nan)
        if pd.notna(xwoba) and xwoba > 0 and xwoba < 2.0:
            x_color = spark_color_xwoba(xwoba, for_pitcher=False)
            xwoba_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(xwoba, max_xwoba, x_color, alpha)}'
                f'<span class="spark-val" style="color:{x_color};">.{int(xwoba*1000):03d}</span>'
                f'</div>'
            )
        else:
            xwoba_cell = f'<span style="color:{SLATE};">--</span>'

        rows_html += (
            f'<tr>'
            f'<td><span class="pt-name" style="color:{family_color};">{pt_name}</span></td>'
            f'<td>{whiff_cell}</td>'
            f'<td>{cstr_cell}</td>'
            f'<td>{chase_cell}</td>'
            f'<td>{xwoba_cell}</td>'
            f'<td class="pt-n">{n_pitches:,}</td>'
            f'</tr>'
        )

    return (
        f'<table class="pitch-table">'
        f'<thead><tr>'
        f'<th>Pitch</th><th>Whiff%</th><th>CStr%</th>'
        f'<th>Chase%</th><th>xwOBA</th><th>Pitches</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def build_pitcher_profile_table(arsenal_df: pd.DataFrame) -> str:
    """Build HTML stat table for a pitcher's arsenal profile."""
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    df = arsenal_df.copy()
    df = df[df["pitches"] >= 20]
    if df.empty:
        return ""

    if "csw_pct" not in df.columns:
        if "swings" in df.columns and "whiffs" in df.columns:
            df["csw_pct"] = (df["whiffs"] + df.get("called_strikes", 0)) / df["pitches"].replace(0, np.nan)
        else:
            df["csw_pct"] = np.nan

    df["_order"] = df["pitch_type"].map({pt: i for i, pt in enumerate(PITCH_ORDER)}).fillna(99)
    df = df.sort_values("_order")

    max_whiff = max(df["whiff_rate"].dropna().max(), 0.01)
    max_csw = max(df["csw_pct"].dropna().max(), 0.01) if "csw_pct" in df.columns else 0.40
    max_xwoba = 0.500

    rows_html = ""
    for _, row in df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        family = PITCH_TYPE_TO_FAMILY.get(pt, "offspeed")
        family_color = PITCH_FAMILY_COLORS.get(family, SLATE)

        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_csw = lg.get("csw_pct", 0.29)

        usage = row.get("usage_pct", 0)
        velo = row.get("avg_velo", np.nan)

        whiff = row.get("whiff_rate", np.nan)
        if pd.notna(whiff):
            w_color = spark_color_rate(whiff, lg_whiff, higher_is_worse=False)
            whiff_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(whiff, max_whiff, w_color)}'
                f'<span class="spark-val" style="color:{w_color};">{whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            whiff_cell = f'<span style="color:{SLATE};">--</span>'

        csw = row.get("csw_pct", np.nan)
        if pd.notna(csw):
            csw_color = spark_color_rate(csw, lg_csw, higher_is_worse=False)
            csw_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(csw, max_csw, csw_color)}'
                f'<span class="spark-val" style="color:{csw_color};">{csw*100:.0f}%</span>'
                f'</div>'
            )
        else:
            csw_cell = f'<span style="color:{SLATE};">--</span>'

        xwoba = row.get("xwoba_against", np.nan)
        if pd.notna(xwoba):
            x_color = spark_color_xwoba(xwoba, for_pitcher=True)
            xwoba_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(xwoba, max_xwoba, x_color)}'
                f'<span class="spark-val" style="color:{x_color};">.{int(xwoba*1000):03d}</span>'
                f'</div>'
            )
        else:
            xwoba_cell = f'<span style="color:{SLATE};">--</span>'

        velo_str = f'{velo:.1f}' if pd.notna(velo) else '--'
        usage_str = f'{usage*100:.0f}%'

        rows_html += (
            f'<tr>'
            f'<td><span class="pt-name" style="color:{family_color};">{pt_name}</span></td>'
            f'<td>{whiff_cell}</td>'
            f'<td>{csw_cell}</td>'
            f'<td>{xwoba_cell}</td>'
            f'<td style="color:{CREAM}; font-size:0.82rem; font-weight:600;">{velo_str}</td>'
            f'<td style="color:{CREAM}; font-size:0.82rem; font-weight:600;">{usage_str}</td>'
            f'</tr>'
        )

    return (
        f'<table class="pitch-table">'
        f'<thead><tr>'
        f'<th>Pitch</th><th>Whiff%</th><th>CSW%</th>'
        f'<th>xwOBA Ag</th><th>Velo</th><th>Usage</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def build_matchup_table(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
) -> str:
    """Build combined pitcher-vs-hitter matchup table with sparkbars."""
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    p_df = arsenal_df.copy()
    p_df = p_df[p_df["pitches"] >= 20]
    if p_df.empty:
        return ""

    v_df = vuln_df.copy()
    v_df = v_df[v_df["pitches"] >= 15]
    if not v_df.empty:
        v_df = v_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )
    s_df = str_df.copy() if not str_df.empty else pd.DataFrame()
    if not s_df.empty and "pitches" in s_df.columns:
        s_df = s_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )

    p_df["_order"] = p_df["pitch_type"].map(
        {pt: i for i, pt in enumerate(PITCH_ORDER)}
    ).fillna(99)
    p_df = p_df.sort_values("_order")

    max_p_whiff = max(p_df["whiff_rate"].dropna().max(), 0.01)
    h_whiffs = []
    h_chases = []
    h_xwobas = []
    h_hard_hit_rates = []
    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        h_row = v_df[v_df["pitch_type"] == pt]
        if len(h_row) > 0:
            sw = h_row["swings"].iloc[0] if "swings" in h_row.columns else 0
            wh = h_row["whiffs"].iloc[0] if "whiffs" in h_row.columns else 0
            whiff_r = wh / sw if pd.notna(sw) and sw > 0 else np.nan
            h_whiffs.append(whiff_r)
            if "chase_swings" in h_row.columns and "out_of_zone_pitches" in h_row.columns:
                cs = h_row["chase_swings"].iloc[0]
                oz = h_row["out_of_zone_pitches"].iloc[0]
                h_chases.append(cs / oz if pd.notna(oz) and oz > 0 else np.nan)
            else:
                h_chases.append(np.nan)
        else:
            h_whiffs.append(np.nan)
            h_chases.append(np.nan)
        s_row = s_df[s_df["pitch_type"] == pt] if not s_df.empty else pd.DataFrame()
        if len(s_row) > 0 and "xwoba_contact" in s_row.columns:
            h_xwobas.append(s_row["xwoba_contact"].iloc[0])
        elif len(h_row) > 0 and "xwoba_contact" in h_row.columns:
            h_xwobas.append(h_row["xwoba_contact"].iloc[0])
        else:
            h_xwobas.append(np.nan)
        if len(s_row) > 0 and "hard_hit_rate" in s_row.columns:
            h_hard_hit_rates.append(s_row["hard_hit_rate"].iloc[0])
        else:
            h_hard_hit_rates.append(np.nan)

    max_h_whiff = max(pd.Series(h_whiffs).dropna().max(), 0.01) if any(pd.notna(v) for v in h_whiffs) else 0.40
    max_h_chase = max(pd.Series(h_chases).dropna().max(), 0.01) if any(pd.notna(v) for v in h_chases) else 0.40
    max_h_xwoba = max(pd.Series(h_xwobas).dropna().max(), 0.01) if any(pd.notna(v) for v in h_xwobas) else 0.50

    rows_html = ""
    for idx, (_, row) in enumerate(p_df.iterrows()):
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        family = PITCH_TYPE_TO_FAMILY.get(pt, "offspeed")
        family_color = PITCH_FAMILY_COLORS.get(family, SLATE)

        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_chase = lg.get("chase_rate", 0.30)

        usage = row.get("usage_pct", 0)
        usage_str = f'{usage * 100:.0f}%'

        p_whiff = row.get("whiff_rate", np.nan)
        if pd.notna(p_whiff):
            pw_color = spark_color_rate(p_whiff, lg_whiff, higher_is_worse=False)
            pw_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(p_whiff, max_p_whiff, pw_color)}'
                f'<span class="spark-val" style="color:{pw_color};">{p_whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            pw_cell = f'<span style="color:{SLATE};">--</span>'

        h_whiff = h_whiffs[idx]
        if pd.notna(h_whiff):
            hw_color = spark_color_rate(h_whiff, lg_whiff, higher_is_worse=True)
            hw_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(h_whiff, max_h_whiff, hw_color)}'
                f'<span class="spark-val" style="color:{hw_color};">{h_whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            hw_cell = f'<span style="color:{SLATE};">--</span>'

        h_chase = h_chases[idx]
        if pd.notna(h_chase):
            hc_color = spark_color_rate(h_chase, lg_chase, higher_is_worse=True)
            hc_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(h_chase, max_h_chase, hc_color)}'
                f'<span class="spark-val" style="color:{hc_color};">{h_chase*100:.0f}%</span>'
                f'</div>'
            )
        else:
            hc_cell = f'<span style="color:{SLATE};">--</span>'

        h_xwoba = h_xwobas[idx]
        if pd.notna(h_xwoba) and 0 < h_xwoba < 2.0:
            hx_color = spark_color_xwoba(h_xwoba, for_pitcher=False)
            hx_cell = (
                f'<div class="spark-cell">'
                f'{spark_html(h_xwoba, max_h_xwoba, hx_color)}'
                f'<span class="spark-val" style="color:{hx_color};">.{int(h_xwoba*1000):03d}</span>'
                f'</div>'
            )
        else:
            hx_cell = f'<span style="color:{SLATE};">--</span>'

        lg_xwoba = lg.get("xwoba_contact", 0.320)
        lg_hh = lg.get("hard_hit_rate", 0.33)
        h_hh = h_hard_hit_rates[idx]

        edge_score = 0.0
        if pd.notna(p_whiff):
            edge_score += (p_whiff - lg_whiff) * 2.0
        if pd.notna(h_whiff):
            edge_score += (h_whiff - lg_whiff) * 1.5
        if pd.notna(h_chase):
            edge_score += (h_chase - lg_chase) * 1.0
        if pd.notna(h_xwoba):
            edge_score -= (h_xwoba - lg_xwoba) * 4.0
        if pd.notna(h_hh):
            edge_score -= (h_hh - lg_hh) * 2.0

        if edge_score > 0.10:
            edge_color = SAGE
        elif edge_score < -0.10:
            edge_color = EMBER
        else:
            edge_color = SLATE
        edge_cell = f'<span class="edge-dot" style="background:{edge_color};"></span>'

        rows_html += (
            f'<tr>'
            f'<td><span class="pt-name" style="color:{family_color};">{pt_name}</span></td>'
            f'<td style="color:{SLATE}; font-size:0.82rem;">{usage_str}</td>'
            f'<td>{pw_cell}</td>'
            f'<td>{hw_cell}</td>'
            f'<td>{hc_cell}</td>'
            f'<td>{hx_cell}</td>'
            f'<td style="text-align:center;">{edge_cell}</td>'
            f'</tr>'
        )

    return (
        f'<table class="matchup-table">'
        f'<thead><tr>'
        f'<th>Pitch</th><th>Usage</th><th>P Whiff%</th>'
        f'<th>H Whiff%</th><th>H Chase%</th><th>H xwOBA</th><th>Edge</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )
