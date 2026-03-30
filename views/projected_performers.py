"""Props Lab -- model picks joined against book lines."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, EMBER, SLATE, CREAM
from services.data_loader import load_projections, load_game_props, load_dk_props, load_pp_props


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STAT_LABELS = {
    "K": "Strikeouts",
    "H": "Hits",
    "HR": "Home Runs",
    "TB": "Total Bases",
    "BB": "Walks",
    "R": "Runs",
    "RBI": "RBIs",
    "HRR": "H+R+RBI",
    "Outs": "Outs",
}


def _build_name_lookup() -> dict[int, str]:
    """Build player_id -> name from projections."""
    lookup: dict[int, str] = {}
    hp = load_projections("hitter")
    if not hp.empty and "batter_name" in hp.columns:
        for _, r in hp.iterrows():
            lookup[int(r["batter_id"])] = r["batter_name"]
    pp = load_projections("pitcher")
    if not pp.empty and "pitcher_name" in pp.columns:
        for _, r in pp.iterrows():
            lookup[int(r["pitcher_id"])] = r["pitcher_name"]
    return lookup


def _lookup_p_over(row: pd.Series, line: float) -> float | None:
    """Look up model P(over) at a specific line from the p_over_* columns."""
    col = f"p_over_{line:.1f}"
    if col in row.index and pd.notna(row.get(col)):
        return float(row[col])
    return None


def _american_to_implied(american: str | int | float | None) -> float | None:
    """Convert American odds to implied probability."""
    if american is None or (isinstance(american, float) and pd.isna(american)):
        return None
    try:
        cleaned = str(american).replace("\u2212", "-").replace("\u2013", "-")
        odds = int(cleaned)
    except (ValueError, TypeError):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _edge_color(edge: float) -> str:
    if edge >= 10:
        return SAGE
    if edge >= 5:
        return GOLD
    return SLATE


def _recommendation(edge: float) -> tuple[str, str]:
    """Return (label, color) based on edge size."""
    if edge >= 15:
        return "Strong", SAGE
    if edge >= 8:
        return "Lean", GOLD
    if edge >= 3:
        return "Slight", SLATE
    return "Pass", EMBER


_PP_TIER_BADGE = {
    "standard": ("Standard", SAGE),
    "demon": ("Reach", EMBER),
    "goblin": ("Floor", GOLD),
}


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def page_projected_performers() -> None:
    """Render the Props Lab page."""
    st.markdown(
        '<div class="section-header">Props Lab</div>',
        unsafe_allow_html=True,
    )

    all_props = load_game_props()
    if all_props.empty:
        st.warning("No game props data found.")
        return

    # Compat: coalesce old and new column names
    if "line_mid" in all_props.columns:
        if "line" not in all_props.columns:
            all_props.rename(
                columns={"line_mid": "line", "p_over_mid": "p_over"},
                inplace=True,
            )
        else:
            all_props["line"] = all_props["line"].fillna(all_props["line_mid"])
            all_props["p_over"] = all_props["p_over"].fillna(
                all_props["p_over_mid"]
            )

    # Upcoming picks only
    props = all_props[
        (all_props["game_status"].isin(["scheduled", "in_progress", ""]))
        | all_props["game_status"].isna()
    ].copy()

    if props.empty:
        st.info("No upcoming games with props data.")
        return

    # Resolve player names
    name_lookup = _build_name_lookup()
    props["player_name"] = props["player_id"].map(
        lambda pid: name_lookup.get(int(pid), str(pid))
    )

    # Load book lines
    dk = load_dk_props()
    pp = load_pp_props()

    # --- DraftKings section ---
    _render_dk_section(props, dk)

    # --- Alternate Lines section ---
    _render_pp_section(props, pp)


def _render_dk_section(props: pd.DataFrame, dk: pd.DataFrame) -> None:
    """Render DraftKings edge picks."""
    st.markdown(
        '<div style="color:var(--tdd-gold); font-size:1.0rem; font-weight:700; '
        'margin:1rem 0 0.5rem; letter-spacing:0.3px;">Market Lines</div>',
        unsafe_allow_html=True,
    )

    if dk.empty:
        st.markdown(
            '<div class="tdd-meta">No market lines available. '
            "Lines may not be posted yet for upcoming games.</div>",
            unsafe_allow_html=True,
        )
        return

    # Join DK lines to model props by (player_id, stat)
    dk_std = dk.copy()
    dk_std["dk_implied"] = dk_std["over_odds"].apply(_american_to_implied)

    merged = props.merge(
        dk_std[["player_id", "stat", "line", "over_odds", "dk_implied"]].rename(
            columns={"line": "dk_line", "over_odds": "dk_odds"}
        ),
        on=["player_id", "stat"],
        how="inner",
    )

    if merged.empty:
        st.markdown(
            '<div class="tdd-meta">No market lines matched to model projections.</div>',
            unsafe_allow_html=True,
        )
        return

    # Look up model P(over) at the DK line
    merged["model_p"] = merged.apply(
        lambda r: _lookup_p_over(r, r["dk_line"]), axis=1,
    )
    # Fall back to default p_over if the DK line matches the model line
    merged["model_p"] = merged["model_p"].fillna(
        merged.apply(
            lambda r: r["p_over"] if r.get("dk_line") == r.get("line") else None,
            axis=1,
        )
    )
    merged = merged[merged["model_p"].notna()].copy()

    # Compute edge
    merged["edge"] = (merged["model_p"] - merged["dk_implied"]) * 100

    # Filter: P(over) >= 63%
    strong = merged[merged["model_p"] >= 0.63].sort_values("edge", ascending=False)

    if strong.empty:
        st.markdown(
            '<div class="tdd-meta">No props above 63% with market lines.</div>',
            unsafe_allow_html=True,
        )
        return

    rows_html = ""
    for _, row in strong.iterrows():
        rows_html += _dk_row(row)
    st.markdown(
        f'<div style="display:flex; flex-direction:column; gap:4px;">'
        f"{rows_html}</div>",
        unsafe_allow_html=True,
    )


def _render_pp_section(props: pd.DataFrame, pp: pd.DataFrame) -> None:
    """Render Alternate Lines picks by tier."""
    st.markdown(
        '<div style="color:var(--tdd-gold); font-size:1.0rem; font-weight:700; '
        'margin:1.5rem 0 0.5rem; letter-spacing:0.3px;">Alternate Lines</div>',
        unsafe_allow_html=True,
    )

    if pp.empty:
        st.markdown(
            '<div class="tdd-meta">No Alternate Lines lines available.</div>',
            unsafe_allow_html=True,
        )
        return

    # Join PP lines to model props by (player_id, stat)
    merged = props.merge(
        pp[["player_id", "stat", "line", "odds_type"]].rename(
            columns={"line": "pp_line"}
        ),
        on=["player_id", "stat"],
        how="inner",
    )

    if merged.empty:
        st.markdown(
            '<div class="tdd-meta">No Alternate Lines lines matched to model projections.</div>',
            unsafe_allow_html=True,
        )
        return

    # Look up model P(over) at each PP line
    merged["model_p"] = merged.apply(
        lambda r: _lookup_p_over(r, r["pp_line"]), axis=1,
    )
    merged["model_p"] = merged["model_p"].fillna(
        merged.apply(
            lambda r: r["p_over"] if r.get("pp_line") == r.get("line") else None,
            axis=1,
        )
    )
    merged = merged[merged["model_p"].notna()].copy()

    # Show each tier
    for tier in ("standard", "demon", "goblin"):
        tier_df = merged[merged["odds_type"] == tier].copy()
        if tier_df.empty:
            continue

        strong = tier_df[tier_df["model_p"] >= 0.63].sort_values(
            "model_p", ascending=False
        )
        if strong.empty:
            continue

        label, color = _PP_TIER_BADGE[tier]
        with st.expander(f"{label} ({len(strong)} picks)", expanded=(tier == "standard")):
            rows_html = ""
            for _, row in strong.iterrows():
                rows_html += _pp_row(row, tier)
            st.markdown(
                f'<div style="display:flex; flex-direction:column; gap:4px;">'
                f"{rows_html}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Row renderers
# ---------------------------------------------------------------------------

def _dk_row(row: pd.Series) -> str:
    """Render a DK prop row with edge."""
    name = row["player_name"]
    stat = row["stat"]
    stat_label = _STAT_LABELS.get(stat, stat)
    team = row["team"]
    opp = row["opponent"]
    ptype = row["player_type"]
    type_badge = "P" if ptype == "pitcher" else "H"

    line = row["dk_line"]
    line_str = f"{line:.0f}" if line == int(line) else f"{line:.1f}"
    model_p = row["model_p"]
    pct = model_p * 100
    edge = float(row["edge"])
    dk_odds = str(row["dk_odds"]) if pd.notna(row.get("dk_odds")) else ""
    dk_implied = row["dk_implied"]
    dk_pct = f"{dk_implied * 100:.0f}%" if pd.notna(dk_implied) else ""

    rec_label, rec_color = _recommendation(edge)
    e_color = _edge_color(edge)

    return (
        f'<div style="display:flex; align-items:center; gap:0.5rem; '
        f'padding:6px 10px; background:var(--tdd-dark-card); '
        f'border:1px solid var(--tdd-dark-border); border-radius:6px; '
        f'flex-wrap:wrap;">'
        # Rec badge
        f'<span style="background:{rec_color}22; color:{rec_color}; '
        f'font-size:0.7rem; font-weight:700; padding:2px 8px; '
        f'border-radius:4px; white-space:nowrap;">{rec_label}</span>'
        # Type badge
        f'<span style="font-size:0.65rem; color:{SLATE}; '
        f'border:1px solid var(--tdd-dark-border); border-radius:3px; '
        f'padding:0 0.25rem;">{type_badge}</span>'
        # Name + matchup
        f'<span style="color:{CREAM}; font-size:0.85rem; font-weight:600; '
        f'min-width:0; flex:1;">{name}'
        f'<span style="color:{SLATE}; font-size:0.75rem; margin-left:0.4rem;">'
        f'{team} vs {opp}</span></span>'
        # Stat + line
        f'<span style="color:{CREAM}; font-size:0.8rem; white-space:nowrap;">'
        f'{stat_label} O {line_str}</span>'
        # Model P(over)
        f'<span style="color:{SAGE}; font-size:0.8rem; font-weight:600; '
        f'white-space:nowrap;">{pct:.0f}%</span>'
        # Odds
        f'<span style="color:{SLATE}; font-size:0.75rem; white-space:nowrap;">'
        f'{dk_odds} ({dk_pct})</span>'
        # Edge
        f'<span style="color:{e_color}; font-size:0.8rem; font-weight:700; '
        f'white-space:nowrap; min-width:60px; text-align:right;">'
        f'{edge:+.0f}%</span>'
        f'</div>'
    )


def _pp_row(row: pd.Series, tier: str) -> str:
    """Render a Alternate Lines prop row."""
    name = row["player_name"]
    stat = row["stat"]
    stat_label = _STAT_LABELS.get(stat, stat)
    team = row["team"]
    opp = row["opponent"]
    ptype = row["player_type"]
    type_badge = "P" if ptype == "pitcher" else "H"

    line = row["pp_line"]
    line_str = f"{line:.0f}" if line == int(line) else f"{line:.1f}"
    model_p = row["model_p"]
    pct = model_p * 100

    _, tier_color = _PP_TIER_BADGE.get(tier, ("PP", SLATE))
    p_color = SAGE if model_p >= 0.68 else GOLD if model_p >= 0.63 else SLATE

    return (
        f'<div style="display:flex; align-items:center; gap:0.5rem; '
        f'padding:6px 10px; background:var(--tdd-dark-card); '
        f'border:1px solid var(--tdd-dark-border); border-radius:6px; '
        f'flex-wrap:wrap;">'
        # Type badge
        f'<span style="font-size:0.65rem; color:{SLATE}; '
        f'border:1px solid var(--tdd-dark-border); border-radius:3px; '
        f'padding:0 0.25rem;">{type_badge}</span>'
        # Name + matchup
        f'<span style="color:{CREAM}; font-size:0.85rem; font-weight:600; '
        f'min-width:0; flex:1;">{name}'
        f'<span style="color:{SLATE}; font-size:0.75rem; margin-left:0.4rem;">'
        f'{team} vs {opp}</span></span>'
        # Stat + line
        f'<span style="color:{CREAM}; font-size:0.8rem; white-space:nowrap;">'
        f'{stat_label} O {line_str}</span>'
        # Model P(over)
        f'<span style="color:{p_color}; font-size:0.8rem; font-weight:600; '
        f'white-space:nowrap;">{pct:.0f}%</span>'
        f'</div>'
    )
