"""Projection card component for counting stat projections with confidence tiers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER
from services.data_loader import load_counting_sim


# -----------------------------------------------------------------------
# Confidence tier colour mapping
# -----------------------------------------------------------------------
# HIGH  -> SAGE (brand green/teal)
# MEDIUM -> GOLD/SLATE blend
# LOW   -> muted slate
_TIER_COLORS: dict[str, str] = {
    "HIGH": SAGE,
    "MEDIUM": GOLD,
    "LOW": SLATE,
}
_TIER_BG: dict[str, str] = {
    "HIGH": "rgba(107,163,142,0.12)",
    "MEDIUM": "rgba(200,169,110,0.10)",
    "LOW": "rgba(123,143,166,0.08)",
}

# -----------------------------------------------------------------------
# Stat display configuration
# -----------------------------------------------------------------------
# (display_label, column_prefix, format_fn, higher_is_better)
_HITTER_STATS: list[tuple[str, str, str, bool]] = [
    ("K", "total_k", "int", False),
    ("BB", "total_bb", "int", True),
    ("HR", "total_hr", "int", True),
    ("R", "total_r", "int", True),
    ("RBI", "total_rbi", "int", True),
    ("SB", "total_sb", "int", True),
    ("H", "total_h", "int", True),
]

_PITCHER_STATS: list[tuple[str, str, str, bool]] = [
    ("K", "total_k", "int", True),
    ("BB", "total_bb", "int", False),
    ("IP", "projected_ip", "dec1", True),
    ("ERA", "projected_era", "dec2", False),
    ("WHIP", "projected_whip", "dec2", False),
    ("SV", "total_sv", "int", True),
    ("HLD", "total_hld", "int", True),
]

# Primary headline metric per player type
_PRIMARY_METRIC = {
    "hitter": ("wRC+", "projected_wrc_plus", "int"),
    "pitcher": ("FIP-ERA", "projected_fip_era", "dec2"),
}


def _fmt(val: float | None, fmt: str) -> str:
    """Format a stat value according to its format type."""
    if val is None or pd.isna(val):
        return "--"
    if fmt == "int":
        return f"{val:.0f}"
    if fmt == "dec1":
        return f"{val:.1f}"
    if fmt == "dec2":
        return f"{val:.2f}"
    return str(val)


def _stat_has_data(row: pd.Series, prefix: str) -> bool:
    """Check whether a stat column exists and has a non-null mean."""
    col = f"{prefix}_mean"
    return col in row.index and pd.notna(row.get(col))


def _per_stat_cv(row: pd.Series, prefix: str) -> float:
    """Return CV for a single stat, or 1.0 if unavailable."""
    m = row.get(f"{prefix}_mean")
    s = row.get(f"{prefix}_sd")
    if m is None or s is None or pd.isna(m) or pd.isna(s) or abs(m) < 0.01:
        return 1.0
    return float(s) / max(float(abs(m)), 0.01)


def _stat_tier(cv: float) -> str:
    """Map a single-stat CV to a tier label for display filtering."""
    if cv < 0.25:
        return "HIGH"
    if cv < 0.45:
        return "MEDIUM"
    return "LOW"


# -----------------------------------------------------------------------
# Public component
# -----------------------------------------------------------------------

def render_projection_card(
    player_id: int,
    player_type: str = "hitter",
) -> None:
    """Render a Streamlit projection card for one player.

    Displays the player's projected counting stats colour-coded by
    confidence tier, with the primary metric (wRC+ for hitters,
    FIP-ERA for pitchers) shown prominently.

    Parameters
    ----------
    player_id : int
        ``batter_id`` or ``pitcher_id``.
    player_type : str
        ``"hitter"`` or ``"pitcher"``.
    """
    sim_df = load_counting_sim(player_type)
    if sim_df.empty:
        st.info("No projection data available.")
        return

    id_col = "batter_id" if player_type == "hitter" else "pitcher_id"
    name_col = "batter_name" if player_type == "hitter" else "pitcher_name"

    player_rows = sim_df[sim_df[id_col] == player_id]
    if player_rows.empty:
        st.info("Player not found in projection data.")
        return

    row = player_rows.iloc[0]
    player_name = row.get(name_col, f"ID {player_id}")
    tier = row.get("confidence_tier", "MEDIUM")
    score = row.get("confidence_score", 0.5)

    tier_color = _TIER_COLORS.get(tier, SLATE)
    tier_bg = _TIER_BG.get(tier, "transparent")

    # ------------------------------------------------------------------
    # Primary headline metric
    # ------------------------------------------------------------------
    pm_label, pm_prefix, pm_fmt = _PRIMARY_METRIC[player_type]
    pm_val = row.get(f"{pm_prefix}_mean")
    pm_str = _fmt(pm_val, pm_fmt)

    # Role badge for pitchers
    role_badge = ""
    if player_type == "pitcher" and "role" in row.index:
        role = row["role"]
        if pd.notna(role):
            role_badge = (
                f'<span style="background:{DARK_BORDER}; color:{CREAM}; '
                f'padding:2px 8px; border-radius:4px; font-size:0.75rem; '
                f'margin-left:8px;">{role}</span>'
            )

    # ------------------------------------------------------------------
    # Build stat table rows (only HIGH or MEDIUM confidence stats)
    # ------------------------------------------------------------------
    stat_defs = _HITTER_STATS if player_type == "hitter" else _PITCHER_STATS
    stat_rows_html = ""
    low_stats: list[str] = []

    for label, prefix, fmt, _hib in stat_defs:
        if not _stat_has_data(row, prefix):
            continue

        cv = _per_stat_cv(row, prefix)
        s_tier = _stat_tier(cv)

        mean_val = row.get(f"{prefix}_mean")
        p10_val = row.get(f"{prefix}_p10")
        p90_val = row.get(f"{prefix}_p90")

        mean_str = _fmt(mean_val, fmt)
        range_str = ""
        if p10_val is not None and p90_val is not None and pd.notna(p10_val) and pd.notna(p90_val):
            range_str = f"{_fmt(p10_val, fmt)}-{_fmt(p90_val, fmt)}"

        if s_tier == "LOW":
            low_stats.append(label)
            continue

        s_color = _TIER_COLORS.get(s_tier, SLATE)
        stat_rows_html += (
            f'<tr>'
            f'<td style="padding:4px 10px; color:{CREAM}; font-size:0.85rem;">{label}</td>'
            f'<td style="padding:4px 10px; color:{s_color}; font-weight:600; '
            f'font-size:0.95rem; text-align:right;">{mean_str}</td>'
            f'<td style="padding:4px 10px; color:{SLATE}; font-size:0.78rem; '
            f'text-align:right;">{range_str}</td>'
            f'</tr>'
        )

    # Low-confidence disclaimer
    low_html = ""
    if low_stats:
        low_list = ", ".join(low_stats)
        low_html = (
            f'<div style="margin-top:8px; padding:6px 10px; '
            f'background:rgba(123,143,166,0.08); border-radius:4px; '
            f'font-size:0.78rem; color:{SLATE};">'
            f'&#9888; High uncertainty: {low_list}'
            f'</div>'
        )

    # ------------------------------------------------------------------
    # Confidence badge
    # ------------------------------------------------------------------
    conf_badge = (
        f'<span style="background:{tier_bg}; color:{tier_color}; '
        f'padding:2px 10px; border-radius:4px; font-size:0.75rem; '
        f'font-weight:600; letter-spacing:0.5px;">'
        f'{tier} ({score:.0%})</span>'
    )

    # ------------------------------------------------------------------
    # Assemble card HTML
    # ------------------------------------------------------------------
    card_html = f"""
    <div style="
        background:{DARK_CARD};
        border:1px solid {DARK_BORDER};
        border-radius:10px;
        padding:18px 20px 14px 20px;
        margin-bottom:12px;
    ">
        <!-- Header: name + confidence badge -->
        <div style="display:flex; align-items:center; justify-content:space-between;
                    margin-bottom:10px;">
            <div>
                <span style="color:{CREAM}; font-size:1.1rem; font-weight:700;">
                    {player_name}
                </span>
                {role_badge}
            </div>
            {conf_badge}
        </div>

        <!-- Primary metric -->
        <div style="text-align:center; margin:12px 0 16px 0;">
            <div style="color:{tier_color}; font-size:2.2rem; font-weight:700;
                        line-height:1;">{pm_str}</div>
            <div style="color:{SLATE}; font-size:0.82rem; margin-top:2px;">
                Projected {pm_label}
            </div>
        </div>

        <!-- Counting stats table -->
        <table style="width:100%; border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:1px solid {DARK_BORDER};">
                    <th style="padding:4px 10px; color:{SLATE}; font-size:0.75rem;
                              text-align:left; font-weight:400;">Stat</th>
                    <th style="padding:4px 10px; color:{SLATE}; font-size:0.75rem;
                              text-align:right; font-weight:400;">Proj.</th>
                    <th style="padding:4px 10px; color:{SLATE}; font-size:0.75rem;
                              text-align:right; font-weight:400;">Range</th>
                </tr>
            </thead>
            <tbody>
                {stat_rows_html}
            </tbody>
        </table>

        {low_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)
