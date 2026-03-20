"""Player headshot component using MLB CDN."""
from __future__ import annotations

import streamlit as st


def _headshot_url(player_id: int, size: int = 60) -> str:
    return (
        f"https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/"
        f"w_{size},q_auto:best/v1/people/{player_id}/headshot/67/current"
    )


def render_headshot(player_id: int, size: int = 60, border_color: str = "#C8A96E") -> None:
    """Render a player headshot using st.image (reliable rendering)."""
    st.image(
        _headshot_url(player_id, size),
        width=size,
    )


def headshot_html(player_id: int, size: int = 40) -> str:
    """Return headshot as inline HTML string using CSS background-image."""
    url = _headshot_url(player_id, size)
    return (
        f'<div style="width:{size}px; height:{size}px; border-radius:6px; '
        f'background: url(\'{url}\') center/cover no-repeat, #181b23; '
        f'display:inline-block; vertical-align:middle;"></div>'
    )
