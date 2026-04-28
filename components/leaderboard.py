"""Shared leaderboard card renderer.

Produces the HTML used by the .lb-* CSS classes (see assets/styles.css).
Consumers (views/stats.py, views/projections.py, views/diamond_daily.py, ...)
build a list of row dicts and hand them to render_card() instead of
assembling card markup inline.

Row dict keys (all optional except name and value):
    player_id: int      drives headshot img and profile link URL
    name:      str      display name
    value:     str      pre-formatted primary stat value (caller handles precision)
    team:      str      team abbr; adds data-team attribute for color theming
    range:     str      optional secondary text rendered after value, e.g. "(1.10-1.35)"
    link_type: str      "hitter" or "pitcher" to wrap name in a profile link
"""
from __future__ import annotations

from typing import Iterable, Mapping

from components.headshot import headshot_html
from utils.html import esc, esc_attr
from utils.team_names import team_short


def _row_html(
    rank: int,
    row: Mapping,
    *,
    headshot_size: int,
    top_rank_gold_n: int,
) -> str:
    """Render one standard leaderboard row."""
    pid = int(row["player_id"])
    name = esc(row["name"])
    value = esc(row["value"])
    team = esc_attr(row.get("team", "") or "")
    range_str = esc(row.get("range", "") or "")
    link_type = row.get("link_type", "") or ""

    rank_class = "lb-rank-top lb-rank" if rank <= top_rank_gold_n else "lb-rank"
    headshot = f'<span class="lb-headshot">{headshot_html(pid, size=headshot_size)}</span>'

    if link_type:
        lt = esc_attr(link_type)
        profile_url = f"?page=player_profile&amp;player_id={pid}&amp;player_type={lt}"
        name_html = f'<a href="{profile_url}">{name}</a>'
    else:
        name_html = name

    team_html = (
        f'<span class="lb-team" data-team="{team}">{team_short(team)}</span>' if team else ""
    )
    range_html = f'<span class="lb-range">{range_str}</span>' if range_str else ""

    return (
        f'<div class="lb-row">'
        f'<span class="{rank_class}">{rank}.</span>'
        f"{headshot}"
        f'<span class="lb-name">{name_html}</span>'
        f"{team_html}"
        f'<span class="lb-val">{value}</span>'
        f"{range_html}"
        f"</div>"
    )


def _watch_row_html(row: Mapping) -> str:
    """Render one muted "watch list" row (no rank, no headshot, team inline)."""
    name = esc(row["name"])
    value = esc(row["value"])
    team = esc_attr(row.get("team", "") or "")
    range_str = esc(row.get("range", "") or "")

    team_html = f' (<span data-team="{team}">{team_short(team)}</span>)' if team else ""
    range_html = f'<span class="lb-watch-range">{range_str}</span>' if range_str else ""

    return (
        f'<div class="lb-watch-row">'
        f'<span class="lb-watch-name">{name}{team_html}</span>'
        f'<span class="lb-watch-val">{value}</span>'
        f"{range_html}"
        f"</div>"
    )


def render_card(
    title: str,
    rows: Iterable[Mapping],
    *,
    subtitle: str = "",
    watch_rows: Iterable[Mapping] | None = None,
    watch_header: str = "Players to Watch",
    card_class: str = "lb-card",
    headshot_size: int = 50,
    top_rank_gold_n: int = 5,
) -> str:
    """Return the HTML string for a leaderboard card.

    The caller is responsible for pre-formatting values and ranges.
    The renderer handles rank numbering (1-based), top-N gold highlight,
    headshot, optional profile link, optional team pill, and the
    "watch list" section.
    """
    title = esc(title)
    subtitle_html = (
        f'<span class="lb-subtitle">{esc(subtitle)}</span>' if subtitle else ""
    )

    row_html = "".join(
        _row_html(
            rank,
            row,
            headshot_size=headshot_size,
            top_rank_gold_n=top_rank_gold_n,
        )
        for rank, row in enumerate(rows, 1)
    )

    html = (
        f'<div class="{card_class}">'
        f'<div class="lb-title-row">'
        f'<span class="lb-title">{title}{subtitle_html}</span>'
        f"</div>"
        f"{row_html}"
    )

    if watch_rows:
        watch_html = "".join(_watch_row_html(row) for row in watch_rows)
        if watch_html:
            html += f'<div class="lb-watch-header">{esc(watch_header)}</div>{watch_html}'

    html += "</div>"
    return html
