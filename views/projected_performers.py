"""Props Lab -- model picks organized by stat leaderboards."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, EMBER, SLATE, CREAM
from services.data_loader import (
    load_projections, load_game_props, load_dk_props, load_pp_props,
    fetch_live_schedule,
)
from components.headshot import headshot_html


# ---------------------------------------------------------------------------
# Constants
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

# Display order for stat leaderboards
_STAT_ORDER = ["K", "H", "HR", "TB", "HRR", "BB", "R", "RBI", "Outs"]

_CONFIDENCE_BUCKETS = {
    "All": 0.0,
    "Super High (75%+)": 0.75,
    "High (63%+)": 0.63,
    "Medium (55%+)": 0.55,
    "Low (<55%)": -1.0,
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
# Leaderboard renderer
# ---------------------------------------------------------------------------

def _render_stat_leaderboard(
    stat: str,
    picks: pd.DataFrame,
) -> None:
    """Render a single stat leaderboard card showing top picks."""
    stat_label = _STAT_LABELS.get(stat, stat)

    if picks.empty:
        return

    rows_html = ""
    for i, (_, row) in enumerate(picks.iterrows(), 1):
        name = row["player_name"]
        pid = int(row["player_id"])
        team = row.get("team", "")
        opp = row.get("opponent", "")
        ptype = row.get("player_type", "")
        type_badge = "P" if ptype == "pitcher" else "H"

        line = row["book_line"]
        line_str = f"{line:.0f}" if line == int(line) else f"{line:.1f}"
        model_p = row["model_p"]
        pct = model_p * 100
        color = _p_color(model_p)

        rank_class = "lb-rank lb-rank-top" if i <= 3 else "lb-rank"
        hs = headshot_html(pid, size=32)

        # Tier badge
        tier = row.get("tier", "standard")
        tier_label, tier_color = _TIER_BADGE.get(tier, ("", SLATE))
        tier_html = (
            f'<span style="background:{tier_color}22; color:{tier_color}; '
            f'font-size:0.6rem; font-weight:700; padding:1px 4px; '
            f'border-radius:3px; white-space:nowrap;">{tier_label}</span>'
        )

        # Edge info (market lines only)
        edge_html = ""
        if pd.notna(row.get("edge")) and tier == "market":
            edge = float(row["edge"])
            e_color = _edge_color(edge)
            edge_html = (
                f'<span style="color:{e_color}; font-size:0.75rem; '
                f'font-weight:700; min-width:2.2rem; text-align:right;">'
                f'{edge:+.0f}%</span>'
            )

        # Matchup
        matchup_html = ""
        if team and opp:
            matchup_html = (
                f'<span class="lb-team" data-team="{team}">{team}'
                f'<span style="color:var(--tdd-slate); font-size:0.7rem;"> v </span>'
                f'{opp}</span>'
            )

        rows_html += (
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{i}.</span>'
            f'<span class="lb-headshot">{hs}</span>'
            f'<span style="font-size:0.6rem; color:{SLATE}; '
            f'border:1px solid var(--tdd-dark-border); border-radius:3px; '
            f'padding:0 0.2rem; margin-right:0.2rem;">{type_badge}</span>'
            f'<span class="lb-name" style="font-size:0.85rem;">{name}</span>'
            f'{matchup_html}'
            f'{tier_html}'
            f'<span style="color:{CREAM}; font-size:0.75rem; '
            f'white-space:nowrap; margin-left:auto;">O {line_str}</span>'
            f'<span class="lb-val" style="color:{color}; min-width:2.5rem;">'
            f'{pct:.0f}%</span>'
            f'{edge_html}'
            f'</div>'
        )

    n_picks = len(picks)
    html = (
        f'<div class="lb-card lb-card-full" style="padding:0 0.75rem;">'
        f'<div class="lb-title-row">'
        f'<span class="lb-title">{stat_label}</span>'
        f'<span class="lb-subtitle">{n_picks} picks</span>'
        f'</div>'
        f'<div style="max-height:320px; overflow-y:auto;">'
        f'{rows_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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

    # Today's props
    today = _today_et()
    today_props = all_props[all_props["game_date"] == today].copy()

    if today_props.empty:
        st.info("No props for today's games.")
        return

    # Fetch live game status from MLB API so toggles reflect
    # real-time state (the props parquet status is stale).
    try:
        live_schedule = fetch_live_schedule(today)
        if (
            not live_schedule.empty
            and "game_pk" in live_schedule.columns
            and "status" in live_schedule.columns
            and "game_pk" in today_props.columns
        ):
            live_status = live_schedule.set_index("game_pk")["status"]
            today_props["game_status"] = (
                today_props["game_pk"].map(live_status).fillna(today_props["game_status"])
            )
    except Exception:
        pass  # fall back to parquet status on API failure

    name_lookup = _build_name_lookup()
    today_props["player_name"] = today_props["player_id"].map(
        lambda pid: name_lookup.get(int(pid), str(pid))
    )

    # --- Inline toolbar ---
    col_game, col_type, col_conf, col_tier = st.columns([1, 1, 1, 1])
    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        include_final = st.toggle("Include final", value=False, key="lab_final")
    with col_t2:
        include_live = st.toggle("Include in-progress", value=True, key="lab_live")

    # Filter by game status using live status values:
    # "Scheduled", "Pre-Game", "In Progress", "Final", etc.
    _status_lower = today_props["game_status"].str.lower().str.strip().fillna("")
    is_scheduled = _status_lower.isin(["scheduled", "pre-game", ""])  | _status_lower.isna()
    is_live = _status_lower == "in progress"
    is_final = _status_lower == "final"

    mask = is_scheduled
    if include_live:
        mask = mask | is_live
    if include_final:
        mask = mask | is_final
    props = today_props[mask].copy()

    if props.empty:
        st.info("No props match the current filters.")
        return

    all_picks = _build_all_picks(props)
    if all_picks.empty:
        st.info("No book lines matched to model projections.")
        return

    # --- Build game dropdown options ---
    game_options: dict[str, int | None] = {"All Games": None}
    if "game_pk" in props.columns:
        game_matchups = (
            props.drop_duplicates("game_pk")[["game_pk", "team", "opponent"]]
            .sort_values("team")
        )
        for _, g in game_matchups.iterrows():
            label = f"{g['team']} vs {g['opponent']}"
            game_options[label] = int(g["game_pk"])

    with col_game:
        game_choice = st.selectbox(
            "Game", list(game_options.keys()),
            index=0, key="lab_game", label_visibility="collapsed",
        )
    with col_type:
        type_choice = st.selectbox(
            "Player Type", ["All", "Pitchers", "Hitters"],
            index=0, key="lab_type", label_visibility="collapsed",
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

    # Game filter
    selected_gpk = game_options[game_choice]
    if selected_gpk is not None and "game_pk" in filtered.columns:
        filtered = filtered[filtered["game_pk"] == selected_gpk]

    # Player type filter
    if type_choice == "Pitchers":
        filtered = filtered[filtered["player_type"] == "pitcher"]
    elif type_choice == "Hitters":
        filtered = filtered[filtered["player_type"] != "pitcher"]

    # Confidence filter
    conf_threshold = _CONFIDENCE_BUCKETS[conf_choice]
    if conf_threshold == -1.0:
        filtered = filtered[filtered["model_p"] < 0.55]
    elif conf_threshold > 0:
        filtered = filtered[filtered["model_p"] >= conf_threshold]

    # Tier filter
    tier_key = _TIER_OPTIONS[tier_choice]
    if tier_key == "market":
        filtered = filtered[filtered["tier"].isin(["market", "standard"])]
    elif tier_key is not None:
        filtered = filtered[filtered["tier"] == tier_key]

    if filtered.empty:
        st.markdown(
            '<div class="tdd-meta">No picks match the selected filters.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="tdd-meta" style="margin-bottom:0.5rem;">'
        f'{len(filtered)} picks across '
        f'{filtered["stat"].nunique()} stats</div>',
        unsafe_allow_html=True,
    )

    # --- Render leaderboards by stat in 3-column grid ---
    # Collect stats that have data, in display order
    active_stats = [
        s for s in _STAT_ORDER
        if s in filtered["stat"].values
    ]

    # Render in rows of 3 with padding between columns
    for row_start in range(0, len(active_stats), 3):
        row_stats = active_stats[row_start:row_start + 3]
        cols = st.columns(len(row_stats), gap="large")
        for col, stat in zip(cols, row_stats):
            with col:
                stat_picks = (
                    filtered[filtered["stat"] == stat]
                    .sort_values("model_p", ascending=False)
                )
                _render_stat_leaderboard(stat, stat_picks)
