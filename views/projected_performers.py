"""Props Lab -- model picks with inline filters."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, EMBER, SLATE, CREAM
from services.data_loader import load_projections, load_game_props, load_dk_props, load_pp_props


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAT_OPTIONS = {
    "Hits": "H",
    "Strikeouts": "K",
    "H+R+RBI": "HRR",
    "Total Bases": "TB",
    "Walks": "BB",
    "All": None,
}

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

_CONFIDENCE_BUCKETS = {
    "All": 0.0,
    "Super High (75%+)": 0.75,
    "High (63%+)": 0.63,
    "Medium (55%+)": 0.55,
    "Low (<55%)": -1.0,  # special: below 55%
}

_TIER_OPTIONS = {
    "All": None,
    "Market": "market",
    "Floor": "goblin",
    "Reach": "demon",
}

_TIER_BADGE = {
    "standard": ("Market", SAGE),
    "goblin": ("Floor", GOLD),
    "demon": ("Reach", EMBER),
    "market": ("Market", SAGE),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_name_lookup() -> dict[int, str]:
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
    col = f"p_over_{line:.1f}"
    if col in row.index and pd.notna(row.get(col)):
        return float(row[col])
    return None


def _american_to_implied(american: str | int | float | None) -> float | None:
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


def _p_color(model_p: float) -> str:
    if model_p >= 0.75:
        return SAGE
    if model_p >= 0.63:
        return GOLD
    return SLATE


def _today_et() -> str:
    """Return today's date in ET as ISO string."""
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    return et_now.date().isoformat()


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _build_all_picks(props: pd.DataFrame) -> pd.DataFrame:
    """Join model props against DK and PP lines into a unified DataFrame."""
    dk = load_dk_props()
    pp = load_pp_props()

    picks: list[pd.DataFrame] = []

    # --- DK picks ---
    if not dk.empty:
        dk_std = dk.copy()
        dk_std["dk_implied"] = dk_std["over_odds"].apply(_american_to_implied)

        merged = props.merge(
            dk_std[["player_id", "stat", "line", "over_odds", "dk_implied"]].rename(
                columns={"line": "book_line", "over_odds": "dk_odds"}
            ),
            on=["player_id", "stat"],
            how="inner",
        )
        if not merged.empty:
            merged["model_p"] = merged.apply(
                lambda r: _lookup_p_over(r, r["book_line"]), axis=1,
            )
            merged["model_p"] = merged["model_p"].fillna(
                merged.apply(
                    lambda r: r["p_over"]
                    if r.get("book_line") == r.get("line")
                    else None,
                    axis=1,
                )
            )
            merged["edge"] = merged.apply(
                lambda r: (r["model_p"] - r["dk_implied"]) * 100
                if pd.notna(r.get("model_p")) and pd.notna(r.get("dk_implied"))
                else None,
                axis=1,
            )
            merged["source"] = "market"
            merged["tier"] = "market"
            picks.append(merged)

    # --- PP picks ---
    if not pp.empty:
        merged = props.merge(
            pp[["player_id", "stat", "line", "odds_type"]].rename(
                columns={"line": "book_line"}
            ),
            on=["player_id", "stat"],
            how="inner",
        )
        if not merged.empty:
            merged["model_p"] = merged.apply(
                lambda r: _lookup_p_over(r, r["book_line"]), axis=1,
            )
            merged["model_p"] = merged["model_p"].fillna(
                merged.apply(
                    lambda r: r["p_over"]
                    if r.get("book_line") == r.get("line")
                    else None,
                    axis=1,
                )
            )
            merged["edge"] = None
            merged["dk_odds"] = None
            merged["dk_implied"] = None
            merged["source"] = "alt"
            merged["tier"] = merged["odds_type"]
            picks.append(merged)

    if not picks:
        return pd.DataFrame()

    combined = pd.concat(picks, ignore_index=True)
    combined = combined[combined["model_p"].notna()].copy()
    combined = combined.drop_duplicates(
        subset=["player_id", "stat", "book_line", "tier"],
        keep="first",
    )

    return combined


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

    # Compat
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

    # Today only, scheduled games
    today = _today_et()
    props = all_props[
        (all_props["game_date"] == today)
        & (
            (all_props["game_status"].isin(["scheduled", "in_progress", ""]))
            | all_props["game_status"].isna()
        )
    ].copy()

    if props.empty:
        st.info("No props for today's games.")
        return

    name_lookup = _build_name_lookup()
    props["player_name"] = props["player_id"].map(
        lambda pid: name_lookup.get(int(pid), str(pid))
    )

    all_picks = _build_all_picks(props)
    if all_picks.empty:
        st.info("No book lines matched to model projections.")
        return

    # --- Inline toolbar ---
    col_prop, col_conf, col_tier = st.columns([1, 1, 1])

    with col_prop:
        prop_choice = st.selectbox(
            "Prop", list(_STAT_OPTIONS.keys()),
            index=0, key="lab_prop", label_visibility="collapsed",
        )
    with col_conf:
        conf_choice = st.selectbox(
            "Confidence", list(_CONFIDENCE_BUCKETS.keys()),
            index=0, key="lab_conf", label_visibility="collapsed",
        )
    with col_tier:
        tier_choice = st.selectbox(
            "Line Type", list(_TIER_OPTIONS.keys()),
            index=1, key="lab_tier", label_visibility="collapsed",
        )

    # --- Apply filters ---
    filtered = all_picks.copy()

    # Prop filter
    stat_key = _STAT_OPTIONS[prop_choice]
    if stat_key is not None:
        filtered = filtered[filtered["stat"] == stat_key]

    # Confidence filter
    conf_threshold = _CONFIDENCE_BUCKETS[conf_choice]
    if conf_threshold == -1.0:
        # Low: below 55%
        filtered = filtered[filtered["model_p"] < 0.55]
    elif conf_threshold > 0:
        filtered = filtered[filtered["model_p"] >= conf_threshold]

    # Tier filter -- "Market" matches both DK market and PP standard
    tier_key = _TIER_OPTIONS[tier_choice]
    if tier_key == "market":
        filtered = filtered[filtered["tier"].isin(["market", "standard"])]
    elif tier_key is not None:
        filtered = filtered[filtered["tier"] == tier_key]

    # Sort by confidence
    filtered = filtered.sort_values("model_p", ascending=False)

    # --- Render ---
    if filtered.empty:
        st.markdown(
            '<div class="tdd-meta">No picks match the selected filters.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="tdd-meta" style="margin-bottom:0.5rem;">'
        f'{len(filtered)} picks</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for _, row in filtered.iterrows():
        rows_html += _pick_row(row)
    st.markdown(
        f'<div style="display:flex; flex-direction:column; gap:4px;">'
        f"{rows_html}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Row renderer
# ---------------------------------------------------------------------------

def _pick_row(row: pd.Series) -> str:
    """Render a single prop pick row."""
    name = row["player_name"]
    stat = row["stat"]
    stat_label = _STAT_LABELS.get(stat, stat)
    team = row["team"]
    opp = row["opponent"]
    ptype = row["player_type"]
    type_badge = "P" if ptype == "pitcher" else "H"

    line = row["book_line"]
    line_str = f"{line:.0f}" if line == int(line) else f"{line:.1f}"
    model_p = row["model_p"]
    pct = model_p * 100
    color = _p_color(model_p)

    # Tier badge
    tier = row.get("tier", "standard")
    tier_label, tier_color = _TIER_BADGE.get(tier, ("", SLATE))
    tier_html = (
        f'<span style="background:{tier_color}22; color:{tier_color}; '
        f'font-size:0.65rem; font-weight:700; padding:1px 6px; '
        f'border-radius:3px; white-space:nowrap;">{tier_label}</span>'
    )

    # Edge info (market lines only)
    edge_html = ""
    if pd.notna(row.get("edge")) and tier == "market":
        edge = float(row["edge"])
        dk_implied = row.get("dk_implied")
        dk_pct = f"{dk_implied * 100:.0f}%" if pd.notna(dk_implied) else ""
        e_color = _edge_color(edge)
        edge_html = (
            f'<span style="color:{SLATE}; font-size:0.72rem; white-space:nowrap;">'
            f'Mkt {dk_pct}</span>'
            f'<span style="color:{e_color}; font-size:0.78rem; font-weight:700; '
            f'white-space:nowrap;">{edge:+.0f}%</span>'
        )

    return (
        f'<div style="display:flex; align-items:center; gap:0.5rem; '
        f'padding:6px 10px; background:var(--tdd-dark-card); '
        f'border:1px solid var(--tdd-dark-border); border-radius:6px; '
        f'flex-wrap:wrap;">'
        # Tier badge
        f'{tier_html}'
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
        f'<span style="color:{color}; font-size:0.8rem; font-weight:600; '
        f'white-space:nowrap;">{pct:.0f}%</span>'
        # Edge (market only)
        f'{edge_html}'
        f'</div>'
    )
