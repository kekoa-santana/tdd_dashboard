"""Team logo HTML component — mirrors headshot.py pattern."""
from __future__ import annotations


def _team_logo_url(team_id: int, size: int = 40) -> str:
    """MLB CDN team cap logo (dark background variant)."""
    return (
        f"https://www.mlbstatic.com/team-logos/team-cap-on-dark/{team_id}.svg"
    )


def team_logo_html(team_id: int, size: int = 40) -> str:
    """Return an inline HTML ``<img>`` tag for a team's cap logo.

    Parameters
    ----------
    team_id : int
        MLB Stats API team ID (e.g. 147 = NYY, 119 = LAD).
    size : int
        Display width in pixels.
    """
    url = _team_logo_url(team_id, size)
    return (
        f'<img src="{url}" '
        f'style="width:{size}px; height:{size}px; border-radius:4px; '
        f'background-color:transparent; display:inline-block; '
        f'vertical-align:middle; object-fit:contain;" />'
    )
