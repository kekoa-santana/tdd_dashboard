"""Props Lab -- player-first board with calibration trust, filters, and why drawers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, EMBER, SLATE, CREAM
from utils.alerts import tdd_info, tdd_warn
from services.data_loader import (
    load_projections, load_game_props, load_dk_props, load_pp_props,
    load_todays_games,
    load_stat_tier_thresholds, load_prop_attribution,
    dedupe_pp_against_dk,
)
from components.attribution import build_attribution_panel
from components.headshot import headshot_html
from utils.helpers import format_game_time
from utils.html import esc, esc_attr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAT_LABELS = {
    "K": "Strikeouts", "H": "Hits", "HR": "Home Runs",
    "TB": "Total Bases", "BB": "Walks", "R": "Runs",
    "RBI": "RBIs", "HRR": "H+R+RBI", "Outs": "Outs",
}

_STAT_ORDER = ["H", "TB", "HR", "HRR", "BB", "Outs", "R", "RBI", "K"]

_CONFIDENCE_TIER_ORDER = ["Lock", "Strong", "Lean", "Pass"]

_TIER_COLORS = {
    "Lock": SAGE, "Strong": GOLD, "Lean": SLATE, "Pass": EMBER,
}

# Calibration trust thresholds
_CAL_LABELS = {
    (0.80, 1.0): ("Calibrated", "sage"),
    (0.65, 0.80): ("Reliable", "gold"),
    (0.55, 0.65): ("Watch", "slate"),
    (0.0, 0.55): ("Soft", "ember"),
}

# Team avatar colors
_TEAM_COLORS = {
    "NYY": "#0a2747", "LAD": "#005A9C", "NYM": "#FF5910", "BOS": "#BD3039",
    "SF": "#FD5A1E", "ATL": "#13274F", "PHI": "#E81828", "HOU": "#EB6E1F",
    "SEA": "#0C2C56", "TEX": "#003278", "LAA": "#BA0021", "SD": "#2F241D",
    "TOR": "#134A8E", "BAL": "#DF4601", "DET": "#0C2340", "KC": "#004687",
    "MIL": "#12284B", "STL": "#C41E3A", "CIN": "#C6011F", "CLE": "#00385D",
    "MIN": "#002B5C", "TB": "#092C5C", "CWS": "#27251F", "PIT": "#27251F",
    "ARI": "#A71930", "COL": "#33006F", "MIA": "#00A3E0", "CHC": "#0E3386",
    "WSH": "#AB0003", "OAK": "#003831",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_et() -> str:
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    return et_now.date().isoformat()


def _tomorrow_et() -> str:
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    return (et_now.date() + timedelta(days=1)).isoformat()


def _lookup_p_over(row: pd.Series, line: float) -> float | None:
    col = f"p_over_{line:.1f}"
    if col in row.index and pd.notna(row.get(col)):
        return float(row[col])
    return None


def _american_to_implied(american) -> float | None:
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


def _confidence_tier(
    thresholds: dict, player_type: str, stat: str, model_p: float,
) -> str:
    confidence = max(model_p, 1 - model_p)
    lookup = thresholds.get("thresholds", {}).get(f"{player_type}_{stat}")
    if not lookup:
        return "Pass"
    for tier in ("Lock", "Strong", "Lean"):
        thr = lookup.get(tier)
        if thr is not None and confidence >= thr:
            return tier
    return "Pass"


def _cal_label(trust: float) -> tuple[str, str]:
    """Return (label, color_class) for a trust score."""
    for (lo, hi), (label, cls) in _CAL_LABELS.items():
        if lo <= trust < hi:
            return label, cls
    return "Soft", "ember"


def _edge_color(edge_pp: float) -> str:
    if abs(edge_pp) >= 12:
        return SAGE
    if abs(edge_pp) >= 6:
        return GOLD
    return SLATE


def _avatar_html(name: str, team: str, size: int = 30) -> str:
    initials = "".join(w[0] for w in name.split()[:2])
    bg = _TEAM_COLORS.get(team, "#2A2E3A")
    fs = size * 0.36
    return (
        f'<div class="pl-avatar" style="width:{size}px;height:{size}px;'
        f'background:{bg};font-size:{fs:.0f}px;">{initials}</div>'
    )


def _edge_bar_svg(model_p: float, market_p: float | None, w: int = 88, h: int = 20) -> str:
    """Inline SVG edge bar showing model vs market probability."""
    lo, hi = 0.30, 0.85
    def norm(p: float) -> float:
        return max(0, min(1, (p - lo) / (hi - lo)))

    mx = norm(model_p) * (w - 4) + 2
    cy = h / 2

    edge = (model_p - market_p) * 100 if market_p is not None else None
    ec = _edge_color(edge) if edge is not None else SLATE

    parts = [
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<line x1="2" x2="{w-2}" y1="{cy}" y2="{cy}" stroke="var(--tdd-dark-border)" stroke-width="1"/>',
    ]

    if market_p is not None:
        kx = norm(market_p) * (w - 4) + 2
        parts.append(
            f'<line x1="{kx:.1f}" x2="{kx:.1f}" y1="{cy-5}" y2="{cy+5}" '
            f'stroke="var(--tdd-slate)" stroke-width="1"/>'
            f'<line x1="{kx:.1f}" x2="{mx:.1f}" y1="{cy}" y2="{cy}" '
            f'stroke="{ec}" stroke-width="2"/>'
        )

    parts.append(
        f'<circle cx="{mx:.1f}" cy="{cy}" r="3.2" fill="{ec}" '
        f'stroke="var(--tdd-dark)" stroke-width="1"/>'
    )
    parts.append('</svg>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _build_flat_picks(props: pd.DataFrame) -> pd.DataFrame:
    """Build a flat pick list -- one row per (player, stat, line, direction)."""
    dk = load_dk_props()
    pp = load_pp_props()
    pp = dedupe_pp_against_dk(dk, pp)
    thresholds = load_stat_tier_thresholds()

    # Game time lookup
    sched = load_todays_games()
    time_map: dict[int, str] = {}
    if not sched.empty and "game_datetime_utc" in sched.columns:
        for _, sg in sched.iterrows():
            gpk = sg.get("game_pk")
            if pd.notna(gpk):
                time_map[int(gpk)] = format_game_time(str(sg["game_datetime_utc"]))

    picks: list[dict] = []

    for _, row in props.iterrows():
        pid = row["player_id"]
        stat = row["stat"]
        ptype = row.get("player_type", "")

        # Collect available lines
        lines_seen: list[tuple[float, float | None]] = []

        dk_match = dk[
            (dk["player_id"] == pid) & (dk["stat"] == stat)
        ] if not dk.empty else pd.DataFrame()
        for _, dl in dk_match.iterrows():
            imp = _american_to_implied(dl.get("over_odds"))
            lines_seen.append((float(dl["line"]), imp))

        pp_match = pp[
            (pp["player_id"] == pid) & (pp["stat"] == stat)
        ] if not pp.empty else pd.DataFrame()
        for _, pl in pp_match.iterrows():
            pp_line = float(pl["line"])
            if not any(abs(l[0] - pp_line) < 0.01 for l in lines_seen):
                lines_seen.append((pp_line, None))

        if not lines_seen:
            default_line = row.get("line")
            if pd.isna(default_line):
                continue
            lines_seen.append((float(default_line), None))

        for book_line, dk_implied in lines_seen:
            model_p = _lookup_p_over(row, book_line)
            if model_p is None and book_line == row.get("line"):
                model_p = row.get("p_over")
            if model_p is None or pd.isna(model_p):
                continue
            model_p = float(model_p)

            edge_pp = None
            if dk_implied is not None and not pd.isna(dk_implied):
                edge_pp = (model_p - dk_implied) * 100

            # Determine direction
            direction = "over" if model_p >= 0.5 else "under"
            display_p = model_p if direction == "over" else 1.0 - model_p

            conf_tier = _confidence_tier(thresholds, ptype, stat, model_p)

            # Skip Pass tier by default
            if conf_tier == "Pass" and (edge_pp is None or abs(edge_pp) < 5):
                continue

            game_pk = row.get("game_pk")
            game_time = time_map.get(int(game_pk), "") if pd.notna(game_pk) else ""

            line_str = f"{book_line:.0f}" if book_line == int(book_line) else f"{book_line:.1f}"

            proj = row.get("projection", row.get("proj"))
            proj_str = ""
            if proj is not None and not pd.isna(proj):
                proj_str = f"{float(proj):.1f}" if float(proj) != int(float(proj)) else f"{int(float(proj))}"

            picks.append({
                "player_id": int(pid),
                "player_name": row.get("player_name", str(pid)),
                "team": row.get("team", ""),
                "opp": row.get("opponent", ""),
                "player_type": ptype,
                "type_badge": "P" if ptype == "pitcher" else "H",
                "stat": stat,
                "stat_label": _STAT_LABELS.get(stat, stat),
                "direction": direction,
                "line": line_str,
                "line_val": book_line,
                "model_p": model_p,
                "display_p": display_p,
                "market_p": dk_implied,
                "edge_pp": edge_pp,
                "conf_tier": conf_tier,
                "game_pk": game_pk,
                "game_time": game_time,
                "proj": proj_str,
            })

    if not picks:
        return pd.DataFrame()

    df = pd.DataFrame(picks)
    # Dedupe: keep best edge per (player, stat, line)
    df = df.drop_duplicates(
        subset=["player_id", "stat", "line_val", "direction"],
        keep="first",
    )
    return df


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _compute_trust(entry: dict | None) -> float:
    """Derive a 0-1 trust score from a threshold entry."""
    if not entry:
        return 0.40
    n_set = sum(1 for t in ("Lock", "Strong", "Lean") if entry.get(t) is not None)
    return 0.40 + n_set * 0.20


_CAL_MIN_N = 30  # hide stats with fewer than 30 backtest observations


def _render_cal_row(
    label: str, prefix: str, stats: list[str], cal_data: dict,
) -> str:
    """Render one calibration row (Hitters or Pitchers) across qualifying stats."""
    cells_html = ""
    shown = 0
    for stat in stats:
        key = f"{prefix}_{stat}"
        entry = cal_data.get(key)
        n_total = entry.get("n_total", 0) if entry else 0
        if n_total < _CAL_MIN_N:
            continue
        shown += 1
        trust = _compute_trust(entry)
        cal_lbl, cls = _cal_label(trust)
        pct = trust * 100
        cells_html += (
            f'<div class="pl-cal-cell">'
            f'<div class="pl-cal-stat">{stat}</div>'
            f'<div class="pl-cal-bar">'
            f'<div class="pl-cal-fill" style="width:{pct:.0f}%;background:var(--tdd-{cls})"></div>'
            f'</div>'
            f'<div class="pl-cal-label" style="color:var(--tdd-{cls})">{cal_lbl}'
            f'<span style="opacity:0.6;margin-left:0.3rem;font-size:0.55rem">n={n_total}</span>'
            f'</div>'
            f'</div>'
        )
    if not shown:
        return ""
    return (
        f'<div style="margin-bottom:0.65rem;">'
        f'<div style="font-family:var(--tdd-font-heading);font-size:0.6rem;'
        f'letter-spacing:1.5px;text-transform:uppercase;color:var(--tdd-slate);'
        f'margin-bottom:0.35rem;font-weight:700;">{label}</div>'
        f'<div class="pl-cal-stripe">{cells_html}</div>'
        f'</div>'
    )


# Pitcher-applicable and hitter-applicable stats
_HITTER_STATS = ["H", "TB", "HR", "HRR", "BB", "R", "RBI", "K"]
_PITCHER_STATS = ["H", "HR", "BB", "K", "Outs"]


def _render_calibration_banner(thresholds: dict) -> str:
    """Stat-level calibration trust grid, split by pitcher and hitter."""
    cal_data = thresholds.get("thresholds", {})
    if not cal_data:
        return ""

    # Determine summary text from combined trust
    all_trusts: dict[str, float] = {}
    for stat in _STAT_ORDER:
        best = 0.0
        for prefix in ("batter", "pitcher"):
            t = _compute_trust(cal_data.get(f"{prefix}_{stat}"))
            if t > best:
                best = t
        all_trusts[stat] = best

    calibrated = [s for s in _STAT_ORDER if all_trusts.get(s, 0) >= 0.80]
    soft = [s for s in _STAT_ORDER if all_trusts.get(s, 0) < 0.55]
    cal_text = ", ".join(calibrated[:3]).lower() if calibrated else "hits and total bases"
    soft_text = ", ".join(soft[:2]).lower() if soft else "strikeouts"

    hitter_row = _render_cal_row("Hitters", "batter", _HITTER_STATS, cal_data)
    pitcher_row = _render_cal_row("Pitchers", "pitcher", _PITCHER_STATS, cal_data)

    return (
        f'<div class="pl-cal-banner">'
        f'<div class="pl-cal-head">'
        f'<div class="pl-cal-eyebrow">Calibration &middot; last 60 days</div>'
        f'<div class="pl-cal-headline">'
        f'We\'re trusting <strong>{cal_text}</strong> tonight. '
        f'We\'re <em>not</em> leading with {soft_text}.'
        f'</div>'
        f'</div>'
        f'{hitter_row}'
        f'{pitcher_row}'
        f'<div class="pl-cal-foot">'
        f'<span class="pl-cal-key"><span class="dot sage"></span> Calibrated</span>'
        f'<span class="pl-cal-key"><span class="dot gold"></span> Reliable</span>'
        f'<span class="pl-cal-key"><span class="dot slate"></span> Watch</span>'
        f'<span class="pl-cal-key"><span class="dot ember"></span> Soft</span>'
        f'</div>'
        f'</div>'
    )


def _render_filter_summary(picks: pd.DataFrame) -> str:
    """Summary bar below filters."""
    n = len(picks)
    locks = len(picks[picks["conf_tier"] == "Lock"]) if not picks.empty else 0
    strong = len(picks[picks["conf_tier"] == "Strong"]) if not picks.empty else 0
    avg_edge = 0.0
    if not picks.empty and "edge_pp" in picks.columns:
        edges = picks["edge_pp"].dropna()
        if len(edges) > 0:
            avg_edge = edges.mean()

    # Best stat = stat with most Lock+Strong picks
    best_stat = ""
    soft_stat = ""
    if not picks.empty:
        tier_counts = picks[picks["conf_tier"].isin(["Lock", "Strong"])].groupby("stat").size()
        if not tier_counts.empty:
            best_stat = tier_counts.idxmax()
        all_counts = picks.groupby("stat").size()
        if not all_counts.empty:
            soft_stat = all_counts.idxmin()

    return (
        f'<div class="pl-filter-summary">'
        f'<span><span class="num">{n}</span> picks</span>'
        f'<span class="dot">&middot;</span>'
        f'<span><span class="num pl-tier-tag tier-Lock">{locks} Lock</span></span>'
        f'<span><span class="num pl-tier-tag tier-Strong">{strong} Strong</span></span>'
        f'<span class="dot">&middot;</span>'
        f'<span>Avg edge <span class="num">{avg_edge:+.1f}pp</span></span>'
        + (f'<span class="dot">&middot;</span>'
           f'<span>Strongest: <span class="pl-stat-tag">{best_stat}</span></span>'
           if best_stat else "")
        + f'</div>'
    )


def _render_pick_row(row: pd.Series, show_why: bool = False) -> str:
    """Render a single pick row."""
    name = row["player_name"]
    team = row["team"]
    opp = row["opp"]
    stat = row["stat"]
    direction = row["direction"]
    line = row["line"]
    model_p = row["model_p"]
    display_p = row["display_p"]
    market_p = row.get("market_p")
    edge_pp = row.get("edge_pp")
    tier = row["conf_tier"]
    game_time = row.get("game_time", "")
    proj = row.get("proj", "")
    type_badge = row.get("type_badge", "H")

    # Direction display
    dir_arrow = "\u25b2" if direction == "over" else "\u25bc"
    dir_label = "OVER" if direction == "over" else "UNDER"
    dir_cls = f"pl-dir pl-dir-{direction}"

    # Model % display
    pct = round(display_p * 100)

    # Edge display
    edge_str = "--"
    edge_color = SLATE
    edge_svg = ""
    if edge_pp is not None and not pd.isna(edge_pp):
        # Flip edge for unders
        display_edge = edge_pp if direction == "over" else -edge_pp
        edge_str = f"{display_edge:+.0f}pp"
        edge_color = _edge_color(abs(display_edge))
        edge_svg = _edge_bar_svg(model_p, market_p if not pd.isna(market_p) else None)

    # Avatar
    avatar = _avatar_html(name, team, 32)

    # Proj display
    proj_html = f'<span class="pl-pick-meta">proj <span class="num">{esc(proj)}</span></span>' if proj else ""

    html = (
        f'<div class="pl-row">'
        # Tier
        f'<div class="pl-cell pl-cell-rank">'
        f'<span class="pl-tier-pill tier-{tier}">{tier}</span>'
        f'</div>'
        # Player
        f'<div class="pl-cell pl-cell-player">'
        f'{avatar}'
        f'<div class="pl-player-info">'
        f'<div class="pl-player-name">{esc(name)}'
        f'<span class="pl-type-badge">{type_badge}</span></div>'
        f'<div class="pl-player-meta">'
        f'<span class="pl-team">{esc(team)}</span>'
        f'<span class="pl-vs">vs</span>'
        f'<span class="pl-team pl-team-opp">{esc(opp)}</span>'
        + (f'<span class="dot">&middot;</span><span>{esc(game_time)}</span>' if game_time else "")
        + f'</div></div></div>'
        # Pick
        f'<div class="pl-cell pl-cell-pick">'
        f'<div class="pl-pick-line">'
        f'<span class="{dir_cls}">{dir_arrow} {dir_label}</span>'
        f'<span class="pl-line">{esc(line)}</span>'
        f'<span class="pl-stat-tag">{esc(stat)}</span>'
        f'</div>'
        f'{proj_html}'
        f'</div>'
        # Model %
        f'<div class="pl-cell pl-cell-prob">'
        f'<div class="pl-prob-num">{pct}<span class="pct">%</span></div>'
        f'<div class="pl-prob-label">model</div>'
        f'</div>'
        # Edge
        f'<div class="pl-cell pl-cell-edge">'
        f'{edge_svg}'
        f'<div class="pl-edge-num" style="color:{edge_color}">{edge_str}</div>'
        f'</div>'
        # Dist placeholder
        f'<div class="pl-cell pl-cell-dist">'
        f'<span style="color:var(--tdd-slate);font-size:0.65rem">--</span>'
        f'</div>'
        f'</div>'
    )

    return html


def _render_table_header() -> str:
    return (
        '<div class="pl-thead">'
        '<div class="pl-th pl-cell-rank">Tier</div>'
        '<div class="pl-th pl-cell-player">Player &middot; Matchup</div>'
        '<div class="pl-th pl-cell-pick">Pick &middot; Line</div>'
        '<div class="pl-th pl-cell-prob">Model</div>'
        '<div class="pl-th pl-cell-edge">Edge vs market</div>'
        '<div class="pl-th pl-cell-dist">Dist</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def page_projected_performers() -> None:
    """Render the Props Lab page."""

    # ── Page header ────────────────────────────────────────────────
    st.markdown(
        '<div class="pl-page">'
        '<header class="pl-page-head">'
        '<div>'
        '<div class="pl-page-eyebrow">The Data Diamond</div>'
        '<h1 class="pl-page-title">Props Lab</h1>'
        '<p class="pl-page-sub">'
        'One row per pick, sortable across the universe. Every pick carries its '
        'model probability, edge vs market, and confidence tier.'
        '</p>'
        '</div>'
        '</header>',
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────
    all_props = load_game_props()
    if all_props.empty:
        st.markdown(
            '<div class="pl-empty">No game props data found.</div></div>',
            unsafe_allow_html=True,
        )
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

    today = _today_et()
    tomorrow = _tomorrow_et()
    today_props = all_props[all_props["game_date"] == today].copy()
    tomorrow_props = all_props[all_props["game_date"] == tomorrow].copy()

    if today_props.empty and tomorrow_props.empty:
        st.markdown(
            '<div class="pl-empty">No props for today\'s or tomorrow\'s games.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Add player names
    name_lookup: dict[int, str] = {}
    hp = load_projections("hitter")
    if not hp.empty and "batter_name" in hp.columns:
        for _, r in hp.iterrows():
            name_lookup[int(r["batter_id"])] = r["batter_name"]
    pp = load_projections("pitcher")
    if not pp.empty and "pitcher_name" in pp.columns:
        for _, r in pp.iterrows():
            name_lookup[int(r["pitcher_id"])] = r["pitcher_name"]

    for df in (today_props, tomorrow_props):
        if not df.empty:
            df["player_name"] = df["player_id"].map(
                lambda pid: name_lookup.get(int(pid), str(pid))
            )

    # Build unified picks from both days
    combined = pd.concat(
        [d for d in (today_props, tomorrow_props) if not d.empty],
        ignore_index=True,
    )
    all_picks = _build_flat_picks(combined)

    if all_picks.empty:
        st.markdown(
            '<div class="pl-empty">No picks match the current filters.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Calibration banner ─────────────────────────────────────────
    thresholds = load_stat_tier_thresholds()
    cal_html = _render_calibration_banner(thresholds)
    if cal_html:
        st.markdown(cal_html, unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1, 1, 1, 1, 2])

    with fcol1:
        type_filter = st.selectbox(
            "Type", ["All", "Hitters", "Pitchers"],
            key="pl_type", label_visibility="collapsed",
        )
    with fcol2:
        stat_options = ["All"] + _STAT_ORDER
        stat_filter = st.selectbox(
            "Stat", stat_options,
            key="pl_stat", label_visibility="collapsed",
        )
    with fcol3:
        tier_filter = st.selectbox(
            "Tier", ["All", "Lock", "Strong", "Lean", "Pass"],
            key="pl_tier", label_visibility="collapsed",
        )
    with fcol4:
        dir_filter = st.selectbox(
            "Direction", ["All", "Over", "Under"],
            key="pl_dir", label_visibility="collapsed",
        )
    with fcol5:
        search = st.text_input(
            "Search", placeholder="Search player or team...",
            key="pl_search", label_visibility="collapsed",
        )

    # Edge slider
    min_edge = st.slider(
        "Min edge (pp)", 0, 20, 0, key="pl_edge",
        label_visibility="collapsed",
    )

    # Apply filters
    filtered = all_picks.copy()
    if type_filter == "Hitters":
        filtered = filtered[filtered["player_type"] != "pitcher"]
    elif type_filter == "Pitchers":
        filtered = filtered[filtered["player_type"] == "pitcher"]

    if stat_filter != "All":
        filtered = filtered[filtered["stat"] == stat_filter]

    if tier_filter != "All":
        filtered = filtered[filtered["conf_tier"] == tier_filter]

    if dir_filter == "Over":
        filtered = filtered[filtered["direction"] == "over"]
    elif dir_filter == "Under":
        filtered = filtered[filtered["direction"] == "under"]

    if min_edge > 0:
        filtered = filtered[
            filtered["edge_pp"].notna() & (filtered["edge_pp"].abs() >= min_edge)
        ]

    if search:
        q = search.lower()
        filtered = filtered[
            filtered["player_name"].str.lower().str.contains(q, na=False)
            | filtered["team"].str.lower().str.contains(q, na=False)
            | filtered["opp"].str.lower().str.contains(q, na=False)
        ]

    # ── Sort ───────────────────────────────────────────────────────
    sort_col = st.selectbox(
        "Sort", ["Tier", "Edge", "Model %", "Stat"],
        key="pl_sort", label_visibility="collapsed",
    )
    tier_rank = {t: i for i, t in enumerate(_CONFIDENCE_TIER_ORDER)}

    if not filtered.empty:
        filtered = filtered.assign(_tier_rank=filtered["conf_tier"].map(tier_rank).fillna(99))
        if sort_col == "Edge":
            filtered = filtered.sort_values(
                ["_tier_rank", "edge_pp"], ascending=[True, False],
                na_position="last",
            )
        elif sort_col == "Model %":
            filtered = filtered.sort_values(
                ["_tier_rank", "display_p"], ascending=[True, False],
            )
        elif sort_col == "Stat":
            filtered = filtered.sort_values(["stat", "_tier_rank"])
        else:
            filtered = filtered.sort_values(
                ["_tier_rank", "display_p"], ascending=[True, False],
            )
        filtered = filtered.drop(columns="_tier_rank")

    # ── Filter summary ─────────────────────────────────────────────
    st.markdown(_render_filter_summary(filtered), unsafe_allow_html=True)

    # ── Table ──────────────────────────────────────────────────────
    if filtered.empty:
        st.markdown(
            '<div class="pl-empty">No picks match these filters.</div>',
            unsafe_allow_html=True,
        )
    else:
        table_html = '<div class="pl-table">'
        table_html += _render_table_header()
        table_html += '<div class="pl-tbody">'
        for _, row in filtered.iterrows():
            table_html += _render_pick_row(row)
        table_html += '</div></div>'
        st.markdown(table_html, unsafe_allow_html=True)

    # ── Explain a projection (attribution waterfall) ───────────────
    # The picks table is a single HTML blob, so per-row expanders don't
    # compose; offer a selector instead. Attribution exists for K only (P1).
    _attr = load_prop_attribution()
    _kpicks = filtered[filtered["stat"] == "K"] if not filtered.empty else filtered
    if not _attr.empty and not _kpicks.empty:
        st.markdown(
            '<div style="color:var(--tdd-gold);font-weight:700;font-size:0.7rem;'
            'letter-spacing:1px;text-transform:uppercase;margin:1.2rem 0 0.3rem">'
            'Why this number</div>',
            unsafe_allow_html=True,
        )
        _opts: dict[str, tuple[int, str, str]] = {}
        for _, _r in _kpicks.iterrows():
            _nm = str(_r["player_name"])
            _opts[f'{_nm} (K)'] = (
                int(_r["player_id"]), str(_r.get("player_type", "")), _nm,
            )
        _PLACEHOLDER = "Select a player..."
        _sel = st.selectbox(
            "Explain a K projection", [_PLACEHOLDER] + list(_opts),
            key="pl_explain", label_visibility="collapsed",
        )
        if _sel and _sel != _PLACEHOLDER:
            _pid, _ptype, _nm = _opts[_sel]
            _ar = _attr[(_attr["player_id"] == _pid) & (_attr["stat"] == "K")]
            if _ptype:
                _ar = _ar[_ar["player_type"] == _ptype]
            if not _ar.empty:
                st.markdown(
                    build_attribution_panel(_ar.iloc[0], name=_nm),
                    unsafe_allow_html=True,
                )

    # Close page wrapper
    st.markdown('</div>', unsafe_allow_html=True)
