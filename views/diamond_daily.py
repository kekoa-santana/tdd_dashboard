"""The Diamond Daily -- editorial scroll page.

Redesigned to match the ScrollVariant from the design handoff:
  masthead > digest > pull quote > today hitters > today pitchers >
  pull quote > L14 boards (hitters + pitchers) > footer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

import pandas as pd
import streamlit as st

from utils.alerts import tdd_info
from utils.html import esc
from utils.team_names import team_short
from components.headshot import headshot_html
from services.data_loader import (
    load_roster,
    load_hitters_daily_standouts,
    load_pitchers_daily_standouts,
    load_hitters_weekly_form,
    load_pitchers_weekly_form,
    load_rankings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _teams_lookup() -> dict[int, str]:
    roster = load_roster()
    if roster.empty or "player_id" not in roster.columns:
        return {}
    return dict(zip(roster["player_id"].astype(int), roster["team_abbr"]))


def _attach_teams(df: pd.DataFrame, id_col: str, teams: dict[int, str]) -> pd.DataFrame:
    if df.empty or id_col not in df.columns:
        return df
    out = df.copy()
    if "team_abbr" not in out.columns:
        out["team_abbr"] = out[id_col].map(
            lambda pid: teams.get(int(pid), "") if pd.notna(pid) else ""
        )
    return out


def _fmt(value, fmt: str = "dec1") -> str:
    if pd.isna(value):
        return ""
    val = float(value)
    if fmt == "int":
        return str(int(round(val)))
    if fmt == "dec1":
        return f"{val:.1f}"
    if fmt == "dec2":
        return f"{val:.2f}"
    if fmt == "dec3":
        return f"{val:.3f}"
    if fmt == "pct0":
        return f"{val * 100:.0f}%"
    if fmt == "pct1":
        return f"{val * 100:.1f}%"
    return str(value)


def _top_val(df: pd.DataFrame, col: str, higher: bool = True) -> str:
    if df.empty or col not in df.columns:
        return ""
    work = df[pd.notna(df[col])]
    if work.empty:
        return ""
    idx = work[col].idxmax() if higher else work[col].idxmin()
    return str(work.loc[idx, col])


def _top_player(df: pd.DataFrame, col: str, name_col: str, higher: bool = True) -> tuple[str, str]:
    if df.empty or col not in df.columns:
        return ("", "")
    work = df[pd.notna(df[col])]
    if work.empty:
        return ("", "")
    idx = work[col].idxmax() if higher else work[col].idxmin()
    row = work.loc[idx]
    return (str(row.get(name_col, "")), str(row[col]))


def _diamond_rating_lookup() -> dict[int, float]:
    """Build player_id -> diamond rating (0-10) lookup from rankings."""
    from lib.diamond_rating import score_to_diamonds

    result: dict[int, float] = {}
    for ptype in ("hitters", "pitchers"):
        try:
            rk = load_rankings(ptype)
        except Exception:
            continue
        if rk.empty:
            continue
        id_col = "batter_id" if ptype == "hitters" else "pitcher_id"
        if id_col in rk.columns and "tdd_value_score" in rk.columns:
            for _, row in rk.iterrows():
                if pd.notna(row.get(id_col)) and pd.notna(row.get("tdd_value_score")):
                    result[int(row[id_col])] = score_to_diamonds(float(row["tdd_value_score"]))
    return result


# ---------------------------------------------------------------------------
# HTML renderers
#
# IMPORTANT: All returned HTML must be left-aligned (no leading whitespace
# on the first <div>). Streamlit's markdown parser treats 4+ spaces of
# indentation as a code block, which renders raw HTML as text.
# Use dedent() on multi-line strings or start them at column 0.
# ---------------------------------------------------------------------------

def _render_masthead(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    weekly_h: pd.DataFrame,
    weekly_p: pd.DataFrame,
) -> str:
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
    edition = f"Vol. IV  No. {now.timetuple().tm_yday}"

    top_name, _ = _top_player(hitters, "daily_standout_score", "batter_name")
    top_hits = ""
    if not hitters.empty and "hits" in hitters.columns:
        h_name, h_val = _top_player(hitters, "hits", "batter_name")
        top_hits = f"{h_name} led with {_fmt(h_val, 'int')} hits"

    headline = f"The board speaks: {top_name} leads all hitters" if top_name else "The Diamond Daily"
    dek = top_hits if top_hits else "Observed leaders from completed games and recent form trends."

    n_hitters = len(hitters) if not hitters.empty else 0
    n_pitchers = len(pitchers) if not pitchers.empty else 0
    hot_hitters = len(weekly_h[weekly_h["weekly_form_score"] > weekly_h["weekly_form_score"].median()]) if not weekly_h.empty and "weekly_form_score" in weekly_h.columns else 0
    avg_form = _fmt(weekly_h["weekly_form_score"].mean(), "dec2") if not weekly_h.empty and "weekly_form_score" in weekly_h.columns else "N/A"
    games = hitters["game_pk"].nunique() if not hitters.empty and "game_pk" in hitters.columns else 0

    stats = [
        (str(games), "Games"),
        (str(n_hitters), "Hitters"),
        (str(n_pitchers), "Pitchers"),
        (str(hot_hitters), "Hot (L14)"),
        (avg_form, "Avg Form"),
    ]

    stats_html = "".join(
        f'<div class="dd-mast-stat">'
        f'<div class="dd-mast-stat-v">{esc(v)}</div>'
        f'<div class="dd-mast-stat-l">{esc(l)}</div>'
        f'</div>'
        for v, l in stats
    )

    return dedent(f"""\
<div class="dd-masthead">
<div class="dd-mast-top">
<div class="dd-mast-eyebrow">
<span class="dd-pip"></span>
<span>The Diamond Daily</span>
<span class="dd-slash">/</span>
<span>{esc(edition)}</span>
</div>
<div class="dd-mast-meta">{esc(date_str)}</div>
</div>
<div class="dd-mast-head">
<div class="dd-mast-left">
<div class="dd-mast-date">{esc(date_str)}</div>
<h1 class="dd-mast-headline">{esc(headline)}</h1>
<div class="dd-mast-dek">{esc(dek)}</div>
<div class="dd-mast-byline">By <b>The Data Diamond</b></div>
</div>
<div class="dd-mast-stats">{stats_html}</div>
</div>
</div>""")


def _render_digest(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> str:
    items: list[tuple[str, str, str, str, str]] = []

    if not hitters.empty and "daily_standout_score" in hitters.columns:
        top = hitters.sort_values("daily_standout_score", ascending=False).iloc[0]
        name = str(top.get("batter_name", ""))
        hits = _fmt(top.get("hits", 0), "int")
        hr = _fmt(top.get("hr", 0), "int")
        score = _fmt(top.get("daily_standout_score", 0), "dec2")
        items.append((
            "Top Hitter",
            f"{name} dominates the board",
            f"{hits} H, {hr} HR in a standout performance. Score: {score}.",
            score, "Score",
        ))

    if not pitchers.empty and "daily_standout_score" in pitchers.columns:
        sp = pitchers[pitchers.get("role", pd.Series()) == "SP"] if "role" in pitchers.columns else pitchers
        if sp.empty:
            sp = pitchers
        top = sp.sort_values("daily_standout_score", ascending=False).iloc[0]
        name = str(top.get("pitcher_name", ""))
        k = _fmt(top.get("k", 0), "int")
        ip = _fmt(top.get("ip", 0), "dec1")
        era = _fmt(top.get("era", 0), "dec2")
        items.append((
            "Pitching",
            f"{name} shuts it down",
            f"{ip} IP, {k} K, {era} ERA. The kind of line the model loves.",
            k, "K",
        ))

    if not hitters.empty and "hr" in hitters.columns:
        hr_leaders = hitters[hitters["hr"] > 0].sort_values("hr", ascending=False)
        if not hr_leaders.empty:
            top = hr_leaders.iloc[0]
            name = str(top.get("batter_name", ""))
            hr_count = _fmt(top.get("hr", 0), "int")
            dist = _fmt(top.get("max_hr_distance", 0), "int") if pd.notna(top.get("max_hr_distance")) else ""
            sub = f"{name} goes deep with {hr_count} HR"
            if dist and dist != "0":
                sub += f". Longest: {dist} ft."
            else:
                sub += "."
            items.append(("Power", f"{name} launches {hr_count}", sub, hr_count, "HR"))

    if not items:
        return ""

    cards = ""
    for kicker, title, sub, stat_v, stat_l in items[:3]:
        cards += (
            '<div class="dd-digest-item">'
            f'<div class="dd-digest-kicker">{esc(kicker)}</div>'
            '<div>'
            f'<div class="dd-digest-title">{esc(title)}</div>'
            f'<div class="dd-digest-sub">{esc(sub)}</div>'
            '</div>'
            '<div class="dd-digest-stat">'
            f'<div class="dd-digest-stat-v">{esc(stat_v)}</div>'
            f'<div class="dd-digest-stat-l">{esc(stat_l)}</div>'
            '</div>'
            '</div>'
        )

    return f'<div class="dd-digest">{cards}</div>'


def _render_pull_quote(text: str) -> str:
    return (
        '<div class="dd-pull">'
        f'<div class="dd-pull-text">{esc(text)}</div>'
        '</div>'
    )


def _render_today_card(
    rank: int,
    name: str,
    team: str,
    edge_val: str,
    edge_label: str,
    quote: str,
    player_id: int | None = None,
) -> str:
    team_html = f'<span class="dd-team-abbr" data-team="{esc(team)}">{esc(team_short(team))}</span>' if team else ""

    return (
        '<div class="dd-today-card">'
        '<div class="dd-today-body">'
        '<div class="dd-today-line1">'
        f'<div class="dd-today-rank">{rank}</div>'
        '<div class="dd-today-ident">'
        f'<div class="dd-today-name">{esc(name)}</div>'
        f'<div class="dd-today-meta">{team_html}</div>'
        '</div>'
        '<div class="dd-today-edge">'
        f'<div class="dd-today-edge-v">{esc(edge_val)}</div>'
        f'<div class="dd-today-edge-l">{esc(edge_label)}</div>'
        '</div>'
        '</div>'
        '<div class="dd-today-line2">'
        f'<div class="dd-today-quote">{esc(quote)}</div>'
        '</div>'
        '</div>'
        '</div>'
    )


def _render_today_section(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    teams: dict[int, str],
) -> str:
    html = ""

    if not hitters.empty and "daily_standout_score" in hitters.columns:
        top5 = hitters.sort_values("daily_standout_score", ascending=False).head(5)
        html += (
            '<div class="dd-section-head">'
            '<div class="dd-section-t">Today - Hitters</div>'
            '<div class="dd-section-s">Top 5 by standout score</div>'
            '</div>'
            '<div class="dd-today-list" style="margin-bottom:2rem">'
        )
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            pid = int(row["batter_id"]) if pd.notna(row.get("batter_id")) else None
            name = str(row.get("batter_name", ""))
            team = teams.get(pid, row.get("team_abbr", "")) if pid else ""
            score = _fmt(row.get("daily_standout_score", 0), "dec2")
            hits = _fmt(row.get("hits", 0), "int")
            hr = _fmt(row.get("hr", 0), "int")
            rbi = _fmt(row.get("rbi", 0), "int")
            quote = f"{hits} H, {hr} HR, {rbi} RBI"
            if pd.notna(row.get("ops")):
                quote += f" / {_fmt(row['ops'], 'dec3')} OPS"
            html += _render_today_card(i, name, team, score, "Score", quote, pid)
        html += "</div>"

    if not pitchers.empty and "daily_standout_score" in pitchers.columns:
        top5 = pitchers.sort_values("daily_standout_score", ascending=False).head(5)
        html += (
            '<div class="dd-section-head">'
            '<div class="dd-section-t">Today - Pitchers</div>'
            '<div class="dd-section-s">Top 5 by standout score</div>'
            '</div>'
            '<div class="dd-today-list" style="margin-bottom:2rem">'
        )
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            pid = int(row["pitcher_id"]) if pd.notna(row.get("pitcher_id")) else None
            name = str(row.get("pitcher_name", ""))
            team = teams.get(pid, row.get("team_abbr", "")) if pid else ""
            score = _fmt(row.get("daily_standout_score", 0), "dec2")
            ip = _fmt(row.get("ip", 0), "dec1")
            k = _fmt(row.get("k", 0), "int")
            era = _fmt(row.get("era", 0), "dec2")
            role = str(row.get("role", ""))
            quote = f"{ip} IP, {k} K, {era} ERA"
            if role:
                quote += f" ({role})"
            html += _render_today_card(i, name, team, score, "Score", quote, pid)
        html += "</div>"

    return html


def _render_l14_board(
    df: pd.DataFrame,
    title: str,
    sub: str,
    id_col: str,
    name_col: str,
    kind: str,
    teams: dict[int, str],
    diamond_lookup: dict[int, float],
    n: int = 10,
) -> str:
    if df.empty or "weekly_form_score" not in df.columns:
        return ""

    top = df.sort_values("weekly_form_score", ascending=False).head(n)

    header = (
        '<div class="dd-board-cols">'
        '<div>#</div>'
        '<div>Player</div>'
        '<div>Form</div>'
        '<div>Stats</div>'
        '<div>GP</div>'
        '<div>&#9670;</div>'
        '</div>'
    )

    rows = ""
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        pid = int(row[id_col]) if pd.notna(row.get(id_col)) else 0
        name = str(row.get(name_col, ""))
        team = teams.get(pid, row.get("team_abbr", ""))
        form_score = _fmt(row.get("weekly_form_score", 0), "dec2")
        games = _fmt(row.get("games_14d", 0), "int")

        if kind == "hitter":
            hits = _fmt(row.get("hits_14d", 0), "int")
            hr = _fmt(row.get("hr_14d", 0), "int")
            woba = _fmt(row.get("woba_14d", 0), "dec3") if pd.notna(row.get("woba_14d")) else ""
            stats_str = f"{hits}H {hr}HR"
            if woba:
                stats_str += f" .{woba.lstrip('0.')}"
        else:
            era = _fmt(row.get("era_14d", 0), "dec2") if pd.notna(row.get("era_14d")) else ""
            whip = _fmt(row.get("whip_14d", 0), "dec2") if pd.notna(row.get("whip_14d")) else ""
            stats_str = f"{era} ERA" if era else ""
            if whip:
                stats_str += f" {whip} WHIP"

        dr = diamond_lookup.get(pid, 0)
        dr_str = f"{dr:.1f}" if dr else ""

        team_html = f'<span class="dd-team-abbr" data-team="{esc(team)}">{esc(team_short(team))}</span>' if team else ""

        rows += (
            '<div class="dd-board-row">'
            f'<div class="c-rank">{rank}</div>'
            '<div class="c-player">'
            '<div class="dd-row-ident">'
            f'<div class="dd-row-name">{esc(name)}</div>'
            f'<div class="dd-row-sub">{team_html}</div>'
            '</div>'
            '</div>'
            f'<div style="color:var(--tdd-gold);font-weight:700;font-variant-numeric:tabular-nums">{esc(form_score)}</div>'
            f'<div class="c-tot">{esc(stats_str)}</div>'
            f'<div style="color:var(--tdd-slate);font-variant-numeric:tabular-nums;text-align:center">{esc(games)}</div>'
            f'<div class="dd-diamond">{esc(dr_str)}</div>'
            '</div>'
        )

    return (
        '<div class="dd-board">'
        '<div class="dd-board-head">'
        f'<div class="dd-board-title">{esc(title)}</div>'
        f'<div class="dd-board-sub">{esc(sub)}</div>'
        '<div></div>'
        '</div>'
        f'{header}'
        f'<div class="dd-board-body">{rows}</div>'
        '</div>'
    )


def _render_footer() -> str:
    return (
        '<div class="dd-foot">'
        '<div>'
        '<span class="dd-foot-eyebrow">Methodology</span> '
        '<span class="dd-foot-method">'
        'All projections use hierarchical Bayesian models with Beta-Binomial '
        'conjugate updating. Daily standout scores weight counting stats, '
        'rate stats, and context (park, opponent, leverage). Weekly form '
        'uses a 14-day rolling window with reliability weighting by sample size.'
        '</span>'
        '</div>'
        '<div class="dd-foot-disc">'
        'For entertainment and research purposes. Not financial advice.'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def page_diamond_daily() -> None:
    """The Diamond Daily -- editorial scroll layout."""

    hitters = load_hitters_daily_standouts()
    pitchers = load_pitchers_daily_standouts()
    weekly_h = load_hitters_weekly_form()
    weekly_p = load_pitchers_weekly_form()

    if hitters.empty and pitchers.empty and weekly_h.empty and weekly_p.empty:
        tdd_info(
            "No Diamond Daily data found yet. Run the update pipeline to generate "
            "daily standout and weekly form parquets."
        )
        return

    teams = _teams_lookup()
    hitters = _attach_teams(hitters, "batter_id", teams)
    pitchers = _attach_teams(pitchers, "pitcher_id", teams)
    weekly_h = _attach_teams(weekly_h, "batter_id", teams)
    weekly_p = _attach_teams(weekly_p, "pitcher_id", teams)
    diamond_lookup = _diamond_rating_lookup()

    # Build entire page as one HTML string and render in a single
    # st.markdown() call. Multiple calls create separate Streamlit
    # container divs with their own margin, causing visible gaps.
    parts: list[str] = []
    parts.append('<div class="dd-page"><div class="dd-scroll">')

    # Masthead
    parts.append(_render_masthead(hitters, pitchers, weekly_h, weekly_p))

    # Digest
    digest = _render_digest(hitters, pitchers)
    if digest:
        parts.append(digest)

    # Pull quote 1
    if not hitters.empty:
        top_name, _ = _top_player(hitters, "daily_standout_score", "batter_name")
        if top_name:
            parts.append(
                _render_pull_quote(f"The numbers don't lie. {top_name} is locked in.")
            )

    # Today's Five
    parts.append(_render_today_section(hitters, pitchers, teams))

    # Pull quote 2
    if not weekly_h.empty and "weekly_form_score" in weekly_h.columns:
        hot_name, hot_score = _top_player(weekly_h, "weekly_form_score", "batter_name")
        if hot_name:
            parts.append(
                _render_pull_quote(
                    f"Two weeks of proof: {hot_name} sits atop the L14 board at {_fmt(hot_score, 'dec2')}."
                )
            )

    # L14 section header
    parts.append(
        '<div class="dd-section-head">'
        '<div class="dd-section-t">Rolling 14 - The Board</div>'
        '<div class="dd-section-s">Top 10 by weekly form score</div>'
        '</div>'
    )

    # L14 Hitters
    parts.append(_render_l14_board(
        weekly_h, title="Hitters - L14", sub="Ranked by weekly form score",
        id_col="batter_id", name_col="batter_name", kind="hitter",
        teams=teams, diamond_lookup=diamond_lookup,
    ))

    # L14 Pitchers
    parts.append(_render_l14_board(
        weekly_p, title="Pitchers - L14", sub="Ranked by weekly form score",
        id_col="pitcher_id", name_col="pitcher_name", kind="pitcher",
        teams=teams, diamond_lookup=diamond_lookup,
    ))

    # Footer
    parts.append(_render_footer())
    parts.append('</div></div>')

    st.markdown("".join(parts), unsafe_allow_html=True)
