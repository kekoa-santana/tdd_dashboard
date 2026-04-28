"""Player Rankings -- press box layout with variant tabs.

Design: eyebrow > masthead > variant tabs (Press Box | Editorial)
  > filter chips > side-by-side compact lists (hitters | pitchers)
  > expandable rows with component bars > methodology footer.

Prospect sections (Hitting Prospects, Pitching Prospects, Readiness)
are retained from the prior implementation with minimal changes.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM
from components.expandable_card import EXPANDABLE_CARD_CSS, expandable_card_html
from components.headshot import headshot_html
from utils.alerts import tdd_info, tdd_warn
from utils.html import esc, esc_attr
from utils.team_names import team_short
from lib.diamond_rating import score_to_diamonds
from services.data_loader import (
    load_core_rankings,
    load_rankings,
    load_player_teams,
    load_prospect_readiness,
    load_milb_factors,
    load_hitter_archetypes,
    load_pitcher_archetypes,
    load_hitter_grade_ci,
    load_pitcher_grade_ci,
)


# ---------------------------------------------------------------------------
# AL / NL team mapping
# ---------------------------------------------------------------------------
_AL_TEAMS = {
    "BAL", "BOS", "NYY", "TB", "TOR",
    "CLE", "CWS", "DET", "KC", "MIN",
    "HOU", "LAA", "OAK", "SEA", "TEX",
}
_NL_TEAMS = {
    "ATL", "MIA", "NYM", "PHI", "WSH",
    "CHC", "CIN", "MIL", "PIT", "STL",
    "ARI", "COL", "LAD", "SD", "SF",
}


# ---------------------------------------------------------------------------
# Hitter / pitcher component labels
# ---------------------------------------------------------------------------
_HITTER_COMP = ["Contact", "Power", "Discipline", "Speed", "Defense"]
_PITCHER_COMP = ["Stuff", "Command", "Deception", "Durability", "Splits"]

_HITTER_GRADE_COLS = [
    ("Contact", "grade_hit"),
    ("Power", "grade_power"),
    ("Speed", "grade_speed"),
    ("Fielding", "grade_fielding"),
    ("Discipline", "grade_discipline"),
]
_PITCHER_GRADE_COLS = [
    ("Stuff", "grade_stuff"),
    ("Command", "grade_command"),
    ("Durability", "grade_durability"),
]

# Key vitals for display
_HITTER_VITS = [
    ("wrc_plus", "wRC+", "int"),
    ("woba", "wOBA", ".000"),
    ("barrel_pct", "Brl%", "pct"),
    ("hard_hit_pct", "HH%", "pct"),
]
_PITCHER_VITS = [
    ("projected_era", "pERA", "0.00"),
    ("k_pct", "K%", "pct"),
    ("bb_pct", "BB%", "pct"),
    ("swstr_pct", "SwStr%", "pct"),
]


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def _fmt(val, fmt: str) -> str:
    if pd.isna(val):
        return ""
    if fmt == "int":
        return str(int(round(val)))
    if fmt == ".000":
        s = f"{val:.3f}"
        return s.lstrip("0") if abs(val) < 1.0 else s
    if fmt == "0.00":
        return f"{val:.2f}"
    if fmt == "pct":
        return f"{val:.1%}"
    return str(val)


def _delta_html(d: int) -> str:
    if d == 0:
        return '<span class="delta flat">--</span>'
    arrow = "&#9650;" if d > 0 else "&#9660;"
    cls = "up" if d > 0 else "down"
    return f'<span class="delta {cls}">{arrow}{abs(d)}</span>'


def _comp_bars_html(items: list[tuple[str, float]], max_v: float = 80) -> str:
    """Comparative bars for scouting grades (20-80 scale)."""
    lg_avg_pct = (50 / max_v) * 100  # 50 = average grade
    rows = []
    for label, v in items:
        pct = max(0, min(100, (v / max_v) * 100))
        rows.append(
            f'<div class="row">'
            f'<span class="lab">{label}</span>'
            f'<div class="bar">'
            f'<div class="fill" style="width:{pct:.1f}%"></div>'
            f'<div class="lg" style="left:{lg_avg_pct:.1f}%"></div>'
            f'</div>'
            f'<span class="v">{v:.0f}</span>'
            f'</div>'
        )
    return f'<div class="rk-comp">{"".join(rows)}</div>'


def _vitals_html(row: pd.Series, vit_config: list[tuple[str, str, str]]) -> str:
    """Build 4-column vitals strip."""
    cells = []
    for col, label, fmt in vit_config:
        val = row.get(col)
        v_str = _fmt(val, fmt) if pd.notna(val) else "--"
        cells.append(
            f'<div class="c">'
            f'<div class="v">{v_str}</div>'
            f'<div class="l">{label}</div>'
            f'</div>'
        )
    return f'<div class="vit">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# Press Box: compact player row
# ---------------------------------------------------------------------------

def _player_row_html(
    row: pd.Series,
    rank: int,
    kind: str,  # "hitter" or "pitcher"
    teams_lookup: dict[int, str],
    compact: bool = True,
) -> str:
    """Render a single player row + expandable detail."""
    id_col = "batter_id" if kind == "hitter" else "pitcher_id"
    name_col = "batter_name" if kind == "hitter" else "pitcher_name"
    pid = int(row[id_col])
    name = str(row[name_col])
    team = teams_lookup.get(pid, "")
    pos = str(row.get("position", row.get("role", "")))
    hand = str(row.get("batter_stand", row.get("pitch_hand", "")))
    age = row.get("age")
    age_str = str(int(age)) if pd.notna(age) else ""

    score_col = "current_value_score" if "current_value_score" in row.index else "tdd_value_score"
    score = float(row.get(score_col, 0) or 0)
    diamond = score_to_diamonds(score)

    # Rank delta (approximate from overall_rank vs talent_rank)
    d_rank = 0  # no week-over-week data in core rankings

    rk_cls = "top1" if rank == 1 else ("top5" if rank <= 5 else "")

    # Summary row
    vit_config = _HITTER_VITS if kind == "hitter" else _PITCHER_VITS
    vit_html = _vitals_html(row, vit_config)

    summary = (
        f'<div class="rk-row{"  compact" if compact else ""}">'
        f'<span class="rk {rk_cls}">{rank}</span>'
        f'{_delta_html(d_rank)}'
        f'<span class="pos">{pos}</span>'
        f'<span class="team-abbr" data-team="{esc_attr(team)}">{esc(team_short(team))}</span>'
        f'<span class="name">'
        f'<span class="nm">{esc(name)}</span>'
        f'</span>'
        f'<span class="score"><span class="glyph">&#9670;</span>{diamond:.1f}</span>'
        f'{vit_html if not compact else ""}'
        f'<span class="arr">&#8250;</span>'
        f'</div>'
    )

    # Expanded detail with comp bars
    grade_cols = _HITTER_GRADE_COLS if kind == "hitter" else _PITCHER_GRADE_COLS
    comp_items = []
    for label, col in grade_cols:
        gv = row.get(col)
        comp_items.append((label, float(gv) if pd.notna(gv) else 50))

    meta_cells = [
        (team, "Team"),
        (f"{pos} / {hand}", "Pos / Hand"),
        (age_str, "Age"),
    ]
    # Add key stats
    for col_name, lbl, fmt in vit_config[:4]:
        val = row.get(col_name)
        meta_cells.append((_fmt(val, fmt) if pd.notna(val) else "--", lbl))

    meta_html = "".join(
        f'<div class="c"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l in meta_cells
    )

    link_type = "hitter" if kind == "hitter" else "pitcher"
    profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"

    expand = (
        f'<div class="rk-expand">'
        f'<div>'
        f'<div class="meta-grid">{meta_html}</div>'
        f'<a class="open-link" href="{profile_url}">Open Profile &#8594;</a>'
        f'</div>'
        f'<div>'
        f'<div style="font-family:var(--tdd-font-heading); font-size:0.6rem; '
        f'letter-spacing:1.3px; color:var(--tdd-slate); font-weight:700; '
        f'text-transform:uppercase; margin-bottom:0.5rem;">'
        f'Scouting grades &middot; 20-80 &middot; vs lg avg</div>'
        f'{_comp_bars_html(comp_items)}'
        f'</div></div>'
    )

    return summary + expand


# ---------------------------------------------------------------------------
# Press Box layout
# ---------------------------------------------------------------------------

def _render_press_box(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    teams_lookup: dict[int, str],
) -> None:
    """Side-by-side compact hitter/pitcher lists."""

    # Column headers
    col_head = (
        '<div class="rk-col-head-row">'
        '<span class="ralign">Rk</span>'
        '<span class="calign">&Delta;</span>'
        '<span class="calign">Pos</span>'
        '<span>Tm</span>'
        '<span>Name</span>'
        '<span class="ralign">&#9670; Score</span>'
        '</div>'
    )

    # Build hitter rows
    h_score_col = "current_value_score" if "current_value_score" in hitters.columns else "tdd_value_score"
    hitters_sorted = hitters.sort_values(h_score_col, ascending=False).reset_index(drop=True)
    h_rows = "".join(
        _player_row_html(row, i + 1, "hitter", teams_lookup, compact=True)
        for i, (_, row) in enumerate(hitters_sorted.iterrows())
    )

    # Build pitcher rows
    p_score_col = "current_value_score" if "current_value_score" in pitchers.columns else "tdd_value_score"
    pitchers_sorted = pitchers.sort_values(p_score_col, ascending=False).reset_index(drop=True)
    p_rows = "".join(
        _player_row_html(row, i + 1, "pitcher", teams_lookup, compact=True)
        for i, (_, row) in enumerate(pitchers_sorted.iterrows())
    )

    html = (
        f'<div class="rk-split">'
        f'<div>'
        f'<div class="rk-col-head">'
        f'<span>Top Hitters</span>'
        f'<span class="meta">{len(hitters_sorted)} ranked</span>'
        f'</div>'
        f'<div class="rk-list">{col_head}{h_rows}</div>'
        f'</div>'
        f'<div>'
        f'<div class="rk-col-head">'
        f'<span>Top Pitchers</span>'
        f'<span class="meta">{len(pitchers_sorted)} ranked</span>'
        f'</div>'
        f'<div class="rk-list">{col_head}{p_rows}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Editorial layout -- podium + full ranks
# ---------------------------------------------------------------------------

def _podium_card_html(
    row: pd.Series,
    place: int,
    kind: str,
    teams_lookup: dict[int, str],
) -> str:
    """Top-3 podium hero card."""
    id_col = "batter_id" if kind == "hitter" else "pitcher_id"
    name_col = "batter_name" if kind == "hitter" else "pitcher_name"
    pid = int(row[id_col])
    name = str(row[name_col])
    team = teams_lookup.get(pid, "")
    pos = str(row.get("position", row.get("role", "")))
    hand = str(row.get("batter_stand", row.get("pitch_hand", "")))

    score_col = "current_value_score" if "current_value_score" in row.index else "tdd_value_score"
    score = float(row.get(score_col, 0) or 0)
    diamond = score_to_diamonds(score)

    cls = "" if place == 1 else ("silver" if place == 2 else "bronze")

    vit_config = _HITTER_VITS if kind == "hitter" else _PITCHER_VITS
    vit_cells = ""
    for col, label, fmt in vit_config:
        val = row.get(col)
        v_str = _fmt(val, fmt) if pd.notna(val) else "--"
        vit_cells += (
            f'<div class="c">'
            f'<div class="v">{v_str}</div>'
            f'<div class="l">{label}</div>'
            f'</div>'
        )

    link_type = "hitter" if kind == "hitter" else "pitcher"
    profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"

    return (
        f'<a href="{profile_url}" style="text-decoration:none; color:inherit;">'
        f'<div class="rk-podium-card {cls}">'
        f'<div class="rank-mark">No. {place}</div>'
        f'<div class="top">'
        f'<span class="ab" data-team="{esc_attr(team)}">{esc(team_short(team))}</span>'
        f'<span class="pos-lbl">{pos} &middot; {hand}</span>'
        f'</div>'
        f'<div class="nm">{esc(name)}</div>'
        f'<div class="score-row">'
        f'<span class="v">&#9670; {diamond:.1f}</span>'
        f'<span class="l">Diamond Score</span>'
        f'</div>'
        f'<div class="vit">{vit_cells}</div>'
        f'</div></a>'
    )


def _render_editorial(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    teams_lookup: dict[int, str],
    editorial_kind: str,
) -> None:
    """Editorial layout with top-3 podium + full ranked list."""
    if editorial_kind == "hitters":
        df = hitters
        kind = "hitter"
        label = "Hitters"
    else:
        df = pitchers
        kind = "pitcher"
        label = "Pitchers"

    score_col = "current_value_score" if "current_value_score" in df.columns else "tdd_value_score"
    sorted_df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    top3 = sorted_df.head(3)
    rest = sorted_df.iloc[3:]

    # Podium
    podium_cards = "".join(
        _podium_card_html(row, i + 1, kind, teams_lookup)
        for i, (_, row) in enumerate(top3.iterrows())
    )

    html = (
        f'<div class="rk-sec"><span class="num">01</span>'
        f'<h2>The Top Three {label}</h2>'
        f'<span class="sub">Diamond Score leaders</span></div>'
        f'<div class="rk-podium">{podium_cards}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    # Full rankings for the rest
    if not rest.empty:
        rows = "".join(
            _player_row_html(row, i + 4, kind, teams_lookup, compact=False)
            for i, (_, row) in enumerate(rest.iterrows())
        )
        rest_html = (
            f'<div class="rk-sec"><span class="num">02</span>'
            f'<h2>Full Rankings</h2>'
            f'<span class="sub">Ranks 4-{len(sorted_df)}</span></div>'
            f'<div class="rk-list">{rows}</div>'
        )
        st.markdown(rest_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _filter_hitters(df: pd.DataFrame, f: str) -> pd.DataFrame:
    if f == "all":
        return df
    if f in ("L", "R"):
        return df[df["batter_stand"].isin([f, "S"])]
    if f == "C":
        return df[df["position"] == "C"]
    if f == "INF":
        return df[df["position"].isin(["1B", "2B", "3B", "SS"])]
    if f == "OF":
        return df[df["position"].isin(["LF", "CF", "RF", "DH"])]
    return df


def _filter_pitchers(df: pd.DataFrame, f: str) -> pd.DataFrame:
    if f == "all":
        return df
    if f in ("L", "R"):
        return df[df["pitch_hand"] == f]
    if f in ("SP", "RP"):
        return df[df["role"] == f]
    return df


def _filter_by_league(
    df: pd.DataFrame,
    league: str,
    id_col: str,
    teams_lookup: dict[int, str],
) -> pd.DataFrame:
    if league == "AL":
        ids = {pid for pid, abbr in teams_lookup.items() if abbr in _AL_TEAMS}
        return df[df[id_col].isin(ids)]
    elif league == "NL":
        ids = {pid for pid, abbr in teams_lookup.items() if abbr in _NL_TEAMS}
        return df[df[id_col].isin(ids)]
    return df


# ===========================================================================
# Prospect sections (kept from prior implementation)
# ===========================================================================

_PROSPECT_TIER_COLORS = {
    "Elite": GOLD, "Impact": EMBER, "Solid": SAGE,
    "Developing": SLATE, "Org Filler": CREAM,
}
_READINESS_TIER_COLORS = {
    "Elite": GOLD, "Strong": EMBER, "Developing": SAGE,
    "Fringe": SLATE, "Long Shot": CREAM,
}
_LEVEL_ORDER = ["AAA", "AA", "A+", "A", "ROK"]
_POS_GROUP_ORDER = ["C", "MI", "Corner", "OF"]

PROSPECT_BATTER_DETAIL_STATS = [
    ("K%", "wtd_k_pct", "pct"), ("BB%", "wtd_bb_pct", "pct"), ("ISO", "wtd_iso", ".000"),
]
PROSPECT_PITCHER_DETAIL_STATS = [
    ("K%", "wtd_k_pct", "pct"), ("BB%", "wtd_bb_pct", "pct"), ("HR/BF", "wtd_hr_bf", ".000"),
]
BATTER_FUTURE_GRADES = [
    ("Hit", "future_hit"), ("Power", "future_power"),
    ("Speed", "future_speed"), ("Fielding", "future_fielding"),
    ("Discipline", "future_discipline"),
]
PITCHER_FUTURE_GRADES = [
    ("Stuff", "future_stuff"), ("Command", "future_command"),
    ("Durability", "future_durability"),
]


def _render_ranking_card(
    df: pd.DataFrame,
    title: str,
    rank_col: str,
    name_col: str,
    id_col: str,
    score_col: str,
    teams_lookup: dict[int, str],
    *,
    projected: bool = False,
    info_col: str | None = None,
    max_height: int = 0,
    detail_stats: list[tuple[str, str, str]] | None = None,
    wide: bool = False,
    link_type: str = "",
    expandable: bool = False,
    archetype_lookup: dict[int, tuple[str, str]] | None = None,
    hover_stats: list[tuple[str, str, str]] | None = None,
    future_grade_cols: list[tuple[str, str]] | None = None,
) -> None:
    """Render a scrollable ranking leaderboard card (used for prospects)."""
    if df.empty:
        return

    if rank_col in df.columns and df[rank_col].notna().any():
        work = df.sort_values(rank_col)
    elif score_col in df.columns and df[score_col].notna().any():
        work = df.sort_values(score_col, ascending=False)
    else:
        work = df.copy()
    has_detail = detail_stats is not None

    def _diamonds_html(rating: float, fill_color: str = "var(--tdd-gold)") -> str:
        if pd.isna(rating):
            return "".join(
                '<span style="color:var(--tdd-slate); opacity:0.35">&#9671;</span>'
                for _ in range(10)
            )
        parts = []
        for i in range(10):
            if i < int(rating) or (i == int(rating) and rating - int(rating) >= 0.5):
                parts.append(f'<span style="color:{fill_color}">&#9670;</span>')
            else:
                parts.append('<span style="color:var(--tdd-slate); opacity:0.35">&#9671;</span>')
        return "".join(parts)

    def _rating_val_html(score_val: float) -> str:
        rating = score_to_diamonds(score_val)
        if pd.isna(rating):
            return (
                f'<span class="lb-diamonds">{_diamonds_html(rating)}</span>'
                f'<span class="lb-rating-num" style="color:{SLATE}">&mdash;</span>'
            )
        if projected:
            fill = SAGE
            num_color = SAGE
        else:
            fill = "var(--tdd-gold)"
            num_color = GOLD if rating >= 7.5 else SAGE if rating >= 5.0 else SLATE
        return (
            f'<span class="lb-diamonds">{_diamonds_html(rating, fill_color=fill)}</span>'
            f'<span class="lb-rating-num" style="color:{num_color}">{rating:.1f}</span>'
        )

    rows_html = []
    for i, (_, row) in enumerate(work.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        rank_class = "lb-rank-top lb-rank" if i <= 5 else "lb-rank"

        hs = f'<span class="lb-headshot">{headshot_html(pid, size=50)}</span>'

        team = teams_lookup.get(pid, "")
        team_html = f'<span class="lb-team" data-team="{esc_attr(team)}">{esc(team_short(team))}</span>' if team else ""

        info_html = ""
        if info_col and info_col in row.index and pd.notna(row[info_col]):
            info_html = f'<span class="lb-info">{row[info_col]}</span>'

        val_html = _rating_val_html(row[score_col])

        stat_inline = ""
        if has_detail:
            for label, col_name, fmt_str in detail_stats:
                if col_name in row.index:
                    val = _fmt(row[col_name], fmt_str)
                    stat_inline += (
                        f'<span class="lb-stat-cell">'
                        f'<span class="lb-stat-lbl">{label}</span>'
                        f'<span class="lb-stat-val">{val}</span>'
                        f'</span>'
                    )

        if not expandable and link_type:
            profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"
            name_content = f'<a href="{profile_url}">{esc(name)}</a>'
        else:
            name_content = name

        summary_row = (
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{i}.</span>'
            f'{hs}'
            f'<span class="lb-name">{name_content}</span>'
            f'{info_html}{team_html}'
            f'<span class="lb-val">{val_html}</span>'
            f'{stat_inline}'
            f'</div>'
        )

        if expandable:
            detail_parts: list[str] = []
            detail_parts.append(
                f'<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">'
                f'{headshot_html(pid, size=80)}'
                f'<div>'
                f'<div style="color:var(--tdd-cream); font-size:1.1rem; font-weight:700;">{esc(name)}</div>'
                f'<div style="color:var(--tdd-slate); font-size:0.8rem;">'
                f'{row.get(info_col, "") if info_col else ""}'
                f'{" . " + team if team else ""}</div>'
                f'</div></div>'
            )

            if archetype_lookup and pid in archetype_lookup:
                arch_name, arch_desc = archetype_lookup[pid]
                detail_parts.append(
                    f'<div style="margin-bottom:0.5rem;">'
                    f'<span style="background:{GOLD}22; color:var(--tdd-gold); border:1px solid {GOLD}44; '
                    f'padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">'
                    f'{arch_name}</span>'
                    f'<span style="color:var(--tdd-slate); font-size:0.78rem; margin-left:0.5rem;">'
                    f'{arch_desc}</span></div>'
                )

            # Scouting grades
            if link_type == "pitcher":
                _gc = [("Stuff", "grade_stuff"), ("Command", "grade_command"), ("Durability", "grade_durability")]
            else:
                _gc = [("Contact", "grade_hit"), ("Power", "grade_power"), ("Speed", "grade_speed"),
                       ("Fielding", "grade_fielding"), ("Discipline", "grade_discipline")]
            _gp = []
            for _lbl, _col in _gc:
                _gv = row.get(_col)
                if pd.notna(_gv):
                    _ci = ""
                    _lo = row.get(f"{_col}_lo")
                    _hi = row.get(f"{_col}_hi")
                    if pd.notna(_lo) and pd.notna(_hi):
                        _ci = (f'<span style="color:var(--tdd-slate); font-size:0.68rem; '
                               f'opacity:0.65; margin-left:2px;">({int(_lo)}-{int(_hi)})</span>')
                    _gp.append(
                        f'<span style="color:var(--tdd-slate); font-size:0.78rem;">{_lbl}: </span>'
                        f'<span style="color:var(--tdd-cream); font-size:0.78rem; font-weight:600;">{int(_gv)}</span>'
                        f'{_ci}'
                    )
            if _gp:
                detail_parts.append(
                    f'<div style="display:flex; flex-wrap:wrap; gap:0.8rem; margin-bottom:0.5rem;">'
                    + "".join(_gp) + '</div>'
                )

            # Future grades
            if future_grade_cols:
                _fp = []
                for _lbl, _col in future_grade_cols:
                    _fv = row.get(_col)
                    if pd.notna(_fv):
                        _fp.append(
                            f'<span style="color:var(--tdd-slate); font-size:0.78rem;">{_lbl}: </span>'
                            f'<span style="color:var(--tdd-sage); font-size:0.78rem; font-weight:600;">{int(_fv)}</span>'
                        )
                if _fp:
                    detail_parts.append(
                        f'<div style="display:flex; flex-wrap:wrap; gap:0.8rem; margin-bottom:0.5rem;">'
                        f'<span style="color:var(--tdd-slate); font-size:0.68rem; font-weight:500; '
                        f'margin-right:0.3rem;">Future:</span>'
                        + "".join(_fp) + '</div>'
                    )

            if has_detail:
                stat_cells = []
                for label, col_name, fmt_str in detail_stats:
                    if col_name in row.index:
                        val = _fmt(row[col_name], fmt_str)
                        stat_cells.append(
                            f'<div style="text-align:center; min-width:55px;">'
                            f'<div style="color:var(--tdd-cream); font-size:0.9rem; font-weight:700;">{val}</div>'
                            f'<div style="color:var(--tdd-slate); font-size:0.65rem;">{label}</div></div>'
                        )
                if stat_cells:
                    detail_parts.append(
                        f'<div style="display:flex; flex-wrap:wrap; gap:0.6rem; '
                        f'margin-bottom:0.5rem;">{"".join(stat_cells)}</div>'
                    )

            if link_type:
                profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"
                detail_parts.append(
                    f'<a href="{profile_url}" style="color:var(--tdd-gold); font-size:0.85rem; '
                    f'font-weight:600; text-decoration:none;">View Full Profile &#8594;</a>'
                )

            rows_html.append(expandable_card_html(summary_row, "".join(detail_parts)))
        else:
            rows_html.append(summary_row)

    count_html = f'<span class="lb-subtitle">{len(work)}</span>'
    card_class = "lb-card lb-card-full" if wide else "lb-card"
    scroll_style = f' style="max-height:{max_height}px;"' if max_height > 0 else ""
    html = (
        f'<div class="{card_class}">'
        f'<div class="lb-scroll"{scroll_style}>'
        f'<div class="lb-title-row">'
        f'<span class="lb-title">{title}{count_html}</span>'
        f'</div>'
        + "".join(rows_html)
        + '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_prospect_rankings(df: pd.DataFrame, search: str = "") -> None:
    """Render hitting prospect rankings."""
    from components.metric_cards import metric_card

    col_tier, col_pos, col_level = st.columns([1, 1, 1])
    with col_tier:
        tiers = ["All", "Elite", "Impact", "Solid", "Developing", "Org Filler"]
        tier_filter = st.selectbox("Tier", tiers, key="rank_pr_tier")
    with col_pos:
        pos_groups = ["All"] + sorted(df["pos_group"].dropna().unique().tolist()) if "pos_group" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", pos_groups, key="rank_pr_pos")
    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_pr_level")

    filtered = df.copy()
    if tier_filter != "All" and "tdd_tier" in filtered.columns:
        filtered = filtered[filtered["tdd_tier"] == tier_filter]
    if pos_filter != "All" and "pos_group" in filtered.columns:
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if level_filter != "All" and "max_level" in filtered.columns:
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    if filtered.empty:
        tdd_info("No matching hitting prospects found.")
        return

    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Prospects Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["tdd_tier"] == "Elite").sum() if "tdd_tier" in filtered.columns else 0
        n_impact = (filtered["tdd_tier"] == "Impact").sum() if "tdd_tier" in filtered.columns else 0
        st.markdown(metric_card("Elite + Impact", f"{n_elite + n_impact}"), unsafe_allow_html=True)
    with cols[2]:
        n_fg = filtered["fg_overall_rank"].notna().sum() if "fg_overall_rank" in filtered.columns else 0
        st.markdown(metric_card("FG Ranked", f"{int(n_fg)}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["tdd_prospect_score"].mean() if "tdd_prospect_score" in filtered.columns and len(filtered) > 0 else 0
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f}"), unsafe_allow_html=True)

    tier_lookup: dict[int, tuple[str, str]] = {}
    for _, row in filtered.iterrows():
        pid = int(row["player_id"])
        tier = str(row.get("tdd_tier", ""))
        level = str(row.get("max_level", ""))
        age = f"Age {row['min_age']:.0f}" if pd.notna(row.get("min_age")) else ""
        tier_lookup[pid] = (tier, f"{level} . {age}" if level and age else level or age)

    level_lookup: dict[int, str] = {}
    for _, row in filtered.iterrows():
        if pd.notna(row.get("max_level")):
            level_lookup[int(row["player_id"])] = str(row["max_level"])

    _render_ranking_card(
        filtered, "Overall", "tdd_rank", "name", "player_id",
        "tdd_prospect_score", level_lookup,
        info_col="primary_position", max_height=600,
        detail_stats=PROSPECT_BATTER_DETAIL_STATS,
        wide=True, link_type="hitter", expandable=True,
        archetype_lookup=tier_lookup, future_grade_cols=BATTER_FUTURE_GRADES,
    )

    st.markdown('<div class="rk-sec"><span class="num">02</span><h2>By Position</h2></div>', unsafe_allow_html=True)
    positions = [p for p in _POS_GROUP_ORDER if p in filtered["pos_group"].unique()]
    positions += [p for p in sorted(filtered["pos_group"].dropna().unique()) if p not in positions]
    for i in range(0, len(positions), 3):
        batch = positions[i:i + 3]
        cols_st = st.columns(3)
        for col_st, pos in zip(cols_st, batch):
            with col_st:
                pos_df = filtered[filtered["pos_group"] == pos].copy()
                _render_ranking_card(
                    pos_df, pos, "tdd_rank", "name", "player_id",
                    "tdd_prospect_score", level_lookup,
                    info_col="primary_position", max_height=520,
                    hover_stats=PROSPECT_BATTER_DETAIL_STATS + [("Age", "min_age", "dec1")],
                )

    st.caption(
        "**Rating** = Diamond Rating from weighted composite. "
        "**K%/BB%/ISO** are MLB-translated MiLB stats."
    )


def _render_pitching_prospect_rankings(df: pd.DataFrame, search: str = "") -> None:
    """Render pitching prospect rankings."""
    from components.metric_cards import metric_card

    col_tier, col_role, col_level = st.columns([1, 1, 1])
    with col_tier:
        tiers = ["All", "Elite", "Impact", "Solid", "Developing", "Org Filler"]
        tier_filter = st.selectbox("Tier", tiers, key="rank_pp_tier")
    with col_role:
        role_filter = st.selectbox("Role", ["All", "SP", "RP"], key="rank_pp_role")
    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_pp_level")

    filtered = df.copy()
    if tier_filter != "All" and "tdd_tier" in filtered.columns:
        filtered = filtered[filtered["tdd_tier"] == tier_filter]
    if role_filter != "All" and "pitcher_role" in filtered.columns:
        filtered = filtered[filtered["pitcher_role"] == role_filter]
    if level_filter != "All" and "max_level" in filtered.columns:
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    if filtered.empty:
        tdd_info("No matching pitching prospects found.")
        return

    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Prospects Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["tdd_tier"] == "Elite").sum() if "tdd_tier" in filtered.columns else 0
        n_impact = (filtered["tdd_tier"] == "Impact").sum() if "tdd_tier" in filtered.columns else 0
        st.markdown(metric_card("Elite + Impact", f"{n_elite + n_impact}"), unsafe_allow_html=True)
    with cols[2]:
        n_fg = filtered["fg_overall_rank"].notna().sum() if "fg_overall_rank" in filtered.columns else 0
        st.markdown(metric_card("FG Ranked", f"{int(n_fg)}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["tdd_prospect_score"].mean() if "tdd_prospect_score" in filtered.columns and len(filtered) > 0 else 0
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f}"), unsafe_allow_html=True)

    tier_lookup: dict[int, tuple[str, str]] = {}
    for _, row in filtered.iterrows():
        pid = int(row["player_id"])
        tier = str(row.get("tdd_tier", ""))
        level = str(row.get("max_level", ""))
        age = f"Age {row['min_age']:.0f}" if pd.notna(row.get("min_age")) else ""
        tier_lookup[pid] = (tier, f"{level} . {age}" if level and age else level or age)

    level_lookup: dict[int, str] = {}
    for _, row in filtered.iterrows():
        if pd.notna(row.get("max_level")):
            level_lookup[int(row["player_id"])] = str(row["max_level"])

    sp_df = filtered[filtered["pitcher_role"] == "SP"].copy() if "pitcher_role" in filtered.columns else filtered
    if not sp_df.empty:
        _render_ranking_card(
            sp_df, "Starting Pitchers", "tdd_rank", "name", "player_id",
            "tdd_prospect_score", level_lookup,
            info_col="max_level", max_height=600,
            detail_stats=PROSPECT_PITCHER_DETAIL_STATS,
            wide=True, link_type="pitcher", expandable=True,
            archetype_lookup=tier_lookup, future_grade_cols=PITCHER_FUTURE_GRADES,
        )

    rp_df = filtered[filtered["pitcher_role"] == "RP"].copy() if "pitcher_role" in filtered.columns else pd.DataFrame()
    if not rp_df.empty:
        _render_ranking_card(
            rp_df, "Relief Pitchers", "tdd_rank", "name", "player_id",
            "tdd_prospect_score", level_lookup,
            info_col="max_level", max_height=600,
            detail_stats=PROSPECT_PITCHER_DETAIL_STATS,
            wide=True, link_type="pitcher", expandable=True,
            archetype_lookup=tier_lookup, future_grade_cols=PITCHER_FUTURE_GRADES,
        )

    st.caption(
        "**Rating** = Diamond Rating from weighted composite. "
        "**K%/BB%/HR/BF** are MLB-translated MiLB stats."
    )


def _style_readiness_tier(val: str) -> str:
    color = _READINESS_TIER_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


def _render_prospect_readiness(df: pd.DataFrame) -> None:
    """Render prospect readiness scores."""
    col_tier, col_pos, col_search = st.columns(3)
    with col_tier:
        tier_options = ["All", "Elite", "Strong", "Developing", "Fringe"]
        tier_filter = st.selectbox("Readiness Tier", tier_options, key="rank_rd_tier")
    with col_pos:
        pos_groups = ["All"] + sorted(df["pos_group"].dropna().unique().tolist()) if "pos_group" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", pos_groups, key="rank_rd_pos")
    with col_search:
        search = st.text_input("Search player", key="rank_rd_search")
    levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
    level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_rd_level")

    filtered = df.copy()
    if tier_filter != "All":
        filtered = filtered[filtered["readiness_tier"] == tier_filter]
    if pos_filter != "All" and "pos_group" in filtered.columns:
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if level_filter != "All" and "max_level" in filtered.columns:
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    from components.metric_cards import metric_card
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Prospects", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["readiness_tier"] == "Elite").sum() if "readiness_tier" in filtered.columns else 0
        n_strong = (filtered["readiness_tier"] == "Strong").sum() if "readiness_tier" in filtered.columns else 0
        st.markdown(metric_card("Elite + Strong", f"{n_elite + n_strong}"), unsafe_allow_html=True)
    with cols[2]:
        n_ranked = filtered["is_ranked"].sum() if "is_ranked" in filtered.columns else 0
        st.markdown(metric_card("FG Ranked", f"{int(n_ranked)}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["readiness_score"].mean() if len(filtered) > 0 else 0
        st.markdown(metric_card("Avg Readiness", f"{avg_score:.1%}"), unsafe_allow_html=True)

    display_map = {
        "readiness_score": "Score", "readiness_tier": "Tier", "name": "Player",
        "pos_group": "Pos", "max_level": "Level", "wtd_k_pct": "K%",
        "wtd_bb_pct": "BB%", "wtd_iso": "ISO", "sb_rate": "SB Rate",
        "youngest_age_rel": "Age vs Lvl", "min_age": "Age",
        "n_above": "Blocked By", "career_milb_pa": "MiLB PA",
    }
    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    display_df = display_df.sort_values("readiness_score", ascending=False)
    display_df.columns = [display_map[c] for c in available]

    fmt_map: dict[str, str] = {}
    for col, f in [
        ("Score", "{:.3f}"), ("K%", "{:.1%}"), ("BB%", "{:.1%}"),
        ("ISO", "{:.3f}"), ("SB Rate", "{:.3f}"), ("Age vs Lvl", "{:+.1f}"),
        ("Blocked By", "{:.0f}"), ("MiLB PA", "{:,.0f}"), ("Age", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt_map[col] = f

    styler = display_df.style.format(fmt_map, na_rep="")
    if "Tier" in display_df.columns:
        styler = styler.map(_style_readiness_tier, subset=["Tier"])
    st.dataframe(styler, use_container_width=True, hide_index=True, height=600)

    st.caption(
        "**Readiness Score** = probability of sticking in MLB (200+ PA season). "
        "**Blocked By** = prospects at same position ahead in the org pipeline."
    )

    with st.expander("Translation Factor Reference"):
        ptype = st.radio("Type", ["Batters", "Pitchers"], horizontal=True, key="rank_factor_type")
        factor_type = "batter" if ptype == "Batters" else "pitcher"
        factors = load_milb_factors(factor_type)
        if not factors.empty:
            st.caption("Translation factors convert MiLB stats to MLB equivalents.")
            pooled = factors[factors["pooled"] == True] if "pooled" in factors.columns else factors  # noqa: E712
            factor_cols = {"level": "Level", "stat": "Stat", "factor": "Factor", "n": "Sample Size", "p25": "P25", "p75": "P75"}
            f_avail = [c for c in factor_cols if c in pooled.columns]
            f_display = pooled[f_avail].copy()
            f_display.columns = [factor_cols[c] for c in f_avail]
            if "Level" in f_display.columns:
                level_cat = pd.CategoricalDtype(categories=_LEVEL_ORDER, ordered=True)
                f_display["Level"] = f_display["Level"].astype(level_cat)
                f_display = f_display.sort_values(["Level", "Stat"])
            st.dataframe(
                f_display.style.format({"Factor": "{:.3f}", "P25": "{:.3f}", "P75": "{:.3f}", "Sample Size": "{:,.0f}"}, na_rep=""),
                use_container_width=True, hide_index=True,
            )
        else:
            tdd_info("Translation factors not available.")


# ===========================================================================
# Main page entry
# ===========================================================================

# CSS for prospect leaderboard cards (lb-* classes)
_PROSPECT_CSS = """
<style>
.lb-title-row { display:flex; justify-content:space-between; align-items:baseline;
  margin-bottom:0; padding-bottom:0.4rem; border-bottom:1px solid var(--tdd-dark-border);
  background:var(--tdd-dark); position:sticky; top:0; z-index:10; }
.lb-title { color:var(--tdd-gold); font-size:1.0rem; font-weight:700; letter-spacing:0.5px; }
.lb-subtitle { color:var(--tdd-slate); font-size:0.70rem; font-weight:400; margin-left:0.4rem; }
.lb-scroll { overflow:visible; }
.lb-scroll[style*="max-height"] { overflow-y:auto; }
.lb-scroll::-webkit-scrollbar { width:6px; }
.lb-scroll::-webkit-scrollbar-track { background:transparent; }
.lb-scroll::-webkit-scrollbar-thumb { background:rgba(200,169,110,0.3); border-radius:3px; }
.lb-row { display:flex; align-items:center; padding:0.28rem 0;
  border-bottom:1px solid var(--tdd-dark-border-faint); }
.lb-row:last-child { border-bottom:none; }
.lb-rank { color:var(--tdd-slate); font-size:0.82rem; min-width:1.6rem;
  text-align:right; margin-right:0.5rem; }
.lb-rank-top { color:var(--tdd-gold); font-weight:700; }
.lb-headshot { margin-left:0.5rem; margin-right:0.5rem; }
.lb-name { color:var(--tdd-cream); font-size:0.95rem; font-weight:600; flex:1; }
.lb-name a { color:inherit; text-decoration:none; }
.lb-name a:hover { color:var(--tdd-gold); text-decoration:underline; }
.lb-info { color:var(--tdd-slate); font-size:0.72rem; background:rgba(123,143,166,0.12);
  padding:1px 6px; border-radius:3px; margin-right:0.5rem; }
.lb-team { color:var(--tdd-slate); font-size:0.80rem; margin-right:0.5rem; }
.lb-val { display:flex; align-items:center; min-width:5rem; justify-content:flex-end; }
.lb-diamonds { letter-spacing:1px; font-size:0.7rem; }
.lb-rating-num { font-weight:700; font-size:0.9rem; margin-left:3px; min-width:1.5rem; text-align:right; }
.lb-stat-cell { margin-left:0.6rem; }
.lb-stat-lbl { color:var(--tdd-slate); font-size:0.62rem; margin-right:2px; }
.lb-stat-val { color:var(--tdd-cream); font-size:0.72rem; font-weight:600; }
</style>
"""

# Backward-compat: breakout.py imports _CSS
_CSS = _PROSPECT_CSS


def page_player_rankings() -> None:
    """Render the Player Rankings page."""

    # ---- Data ----
    teams_df = load_player_teams()
    teams_lookup: dict[int, str] = {}
    if not teams_df.empty:
        teams_lookup = dict(zip(teams_df["player_id"].astype(int), teams_df["team_abbr"]))

    n_total = 0
    as_of = date.today().strftime("%b %d, %Y")

    # ---- Category selector ----
    category = st.radio(
        "Category",
        ["MLB Players", "Hitting Prospects", "Pitching Prospects", "Prospect Readiness"],
        horizontal=True,
        key="rankings_category",
        label_visibility="collapsed",
    )

    if category == "MLB Players":
        # Load MLB rankings
        hitters_df = load_core_rankings("hitters")
        pitchers_df = load_core_rankings("pitchers")

        if hitters_df.empty and pitchers_df.empty:
            tdd_warn("No player rankings data found. Run precompute first.")
            return

        # Merge grade CIs
        _h_ci = load_hitter_grade_ci()
        if not _h_ci.empty and not hitters_df.empty:
            _ci_cols = [c for c in _h_ci.columns if c.endswith("_lo") or c.endswith("_hi")]
            hitters_df = hitters_df.merge(
                _h_ci[["player_id"] + _ci_cols].rename(columns={"player_id": "batter_id"}),
                on="batter_id", how="left",
            )
        _p_ci = load_pitcher_grade_ci()
        if not _p_ci.empty and not pitchers_df.empty:
            _ci_cols = [c for c in _p_ci.columns if c.endswith("_lo") or c.endswith("_hi")]
            pitchers_df = pitchers_df.merge(
                _p_ci[["player_id"] + _ci_cols].rename(columns={"player_id": "pitcher_id"}),
                on="pitcher_id", how="left",
            )

        n_total = len(hitters_df) + len(pitchers_df)

        # ---- Eyebrow + masthead ----
        masthead = (
            '<div class="rk-page">'
            '<div class="rk-eyebrow">'
            '<span class="gold">&#9733;</span>'
            '<span>The Data Diamond</span>'
            '<span class="sep">/</span>'
            '<span>Rankings</span>'
            '<span class="sep">/</span>'
            '<span>Players</span>'
            '</div>'
            '<div class="rk-mast">'
            '<div>'
            '<h1>Player Rankings</h1>'
            '<div class="dek">'
            'The <span class="gold">Diamond Score</span> blends contact, power, plate discipline, '
            'baserunning, and defense for hitters; stuff, command, deception, durability, and splits '
            f'for pitchers. Updated weekly &middot; {as_of}.'
            '</div>'
            '</div>'
            '<div class="meta">'
            f'<div class="stat"><div class="v">{n_total}</div><div class="l">Ranked</div></div>'
            '<div class="stat"><div class="v">5.0</div><div class="l">Lg Avg</div></div>'
            '</div>'
            '</div>'
        )
        st.markdown(masthead, unsafe_allow_html=True)

        # ---- Variant tabs ----
        variant = st.radio(
            "Layout",
            ["Press Box", "Editorial"],
            horizontal=True,
            key="pr_variant",
            label_visibility="collapsed",
        )

        # ---- Filters ----
        if variant == "Press Box":
            fc1, fc2, fc3 = st.columns([2, 2, 1])
            with fc1:
                h_filter = st.radio(
                    "Hitters", ["all", "INF", "OF", "C", "L", "R"],
                    format_func=lambda x: {"all": "All Hitters", "INF": "Infield", "OF": "Outfield", "C": "Catchers", "L": "LHB", "R": "RHB"}[x],
                    horizontal=True, key="pr_h_filter", label_visibility="collapsed",
                )
            with fc2:
                p_filter = st.radio(
                    "Pitchers", ["all", "SP", "RP", "L", "R"],
                    format_func=lambda x: {"all": "All Pitchers", "SP": "Starters", "RP": "Bullpen", "L": "LHP", "R": "RHP"}[x],
                    horizontal=True, key="pr_p_filter", label_visibility="collapsed",
                )
            with fc3:
                league = st.radio(
                    "League", ["All", "AL", "NL"],
                    horizontal=True, key="pr_league", label_visibility="collapsed",
                )

            # Apply filters
            h_filtered = _filter_hitters(hitters_df, h_filter)
            p_filtered = _filter_pitchers(pitchers_df, p_filter)
            h_filtered = _filter_by_league(h_filtered, league, "batter_id", teams_lookup)
            p_filtered = _filter_by_league(p_filtered, league, "pitcher_id", teams_lookup)

            _render_press_box(h_filtered, p_filtered, teams_lookup)

        else:  # Editorial
            fc1, fc2 = st.columns([2, 1])
            with fc1:
                editorial_kind = st.radio(
                    "Show", ["hitters", "pitchers"],
                    format_func=lambda x: x.title(),
                    horizontal=True, key="pr_ed_kind", label_visibility="collapsed",
                )
            with fc2:
                league = st.radio(
                    "League", ["All", "AL", "NL"],
                    horizontal=True, key="pr_ed_league", label_visibility="collapsed",
                )

            h_filtered = _filter_by_league(hitters_df, league, "batter_id", teams_lookup)
            p_filtered = _filter_by_league(pitchers_df, league, "pitcher_id", teams_lookup)

            _render_editorial(h_filtered, p_filtered, teams_lookup, editorial_kind)

        # ---- Methodology footer ----
        methodology = (
            '<div class="rk-methodology">'
            '<span class="hdr">Methodology &middot;</span>'
            'Diamond Score is a 0-10 composite blending five sub-scores per role. '
            'For hitters: contact, power, discipline, speed, defense. '
            'For pitchers: stuff, command, deception, durability, splits. '
            'Each sub-score is normalized vs the league average (5.0) and combined with '
            'role-specific weights. Tap any row to expand component bars.'
            '</div></div>'  # closes rk-page
        )
        st.markdown(methodology, unsafe_allow_html=True)

    elif category == "Hitting Prospects":
        st.markdown(_PROSPECT_CSS, unsafe_allow_html=True)
        st.markdown(EXPANDABLE_CARD_CSS, unsafe_allow_html=True)
        search = st.text_input("Search", placeholder="Search prospect...", key="rank_search", label_visibility="collapsed")
        df = load_rankings("prospect")
        if df.empty:
            tdd_warn("No prospect rankings data found. Run precompute first.")
            return
        _render_prospect_rankings(df, search=search)

    elif category == "Pitching Prospects":
        st.markdown(_PROSPECT_CSS, unsafe_allow_html=True)
        st.markdown(EXPANDABLE_CARD_CSS, unsafe_allow_html=True)
        search = st.text_input("Search", placeholder="Search prospect...", key="rank_search", label_visibility="collapsed")
        df = load_rankings("pitching_prospect")
        if df.empty:
            tdd_warn("No pitching prospect rankings data found. Run precompute first.")
            return
        _render_pitching_prospect_rankings(df, search=search)

    else:  # Prospect Readiness
        st.markdown(_PROSPECT_CSS, unsafe_allow_html=True)
        df = load_prospect_readiness()
        if df.empty:
            tdd_warn("No prospect readiness data found. Run precompute first.")
            return
        _render_prospect_readiness(df)
