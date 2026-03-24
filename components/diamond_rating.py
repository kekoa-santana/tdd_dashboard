"""Diamond Rating visual component."""
from __future__ import annotations

import streamlit as st

GOLD = "#C8A96E"
SLATE = "#7B8FA6"


def _build_diamonds(rating: float, font_size: str) -> str:
    """Build the filled/empty diamond HTML for a given rating (0-10 scale)."""
    full_diamonds = int(rating)
    fractional = rating - full_diamonds

    parts: list[str] = []
    for i in range(10):
        if i < full_diamonds:
            parts.append(f'<span style="color:var(--tdd-gold);">&#9670;</span>')
        elif i == full_diamonds and fractional >= 0.5:
            parts.append(f'<span style="color:var(--tdd-gold);">&#9670;</span>')
        else:
            parts.append(f'<span style="color:var(--tdd-slate);">&#9671;</span>')
    return "".join(parts)


def render_diamond_rating(score: float, size: str = "md", show_numeric: bool = True) -> None:
    """Render diamond rating as HTML diamonds via ``st.markdown``.

    Args:
        score: TDD normalized score (0.0-1.0).
        size: "sm", "md", or "lg".
        show_numeric: whether to show the numeric value beside the diamonds.
    """
    from lib.diamond_rating import score_to_diamonds, diamond_tier

    rating = score_to_diamonds(score)
    tier = diamond_tier(rating)

    sizes = {"sm": "0.9rem", "md": "1.2rem", "lg": "1.6rem"}
    font_size = sizes.get(size, "1.2rem")
    diamonds = _build_diamonds(rating, font_size)

    numeric = f" {rating:.1f}" if show_numeric else ""

    html = (
        f'<span title="Diamond Rating: {rating:.1f} / 10.0 — {tier}" '
        f'style="font-size:{font_size}; letter-spacing:2px; cursor:help;">'
        f'{diamonds}<span style="color:var(--tdd-gold); font-size:0.8em;">{numeric}</span>'
        f'</span>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_diamond_rating_composite(
    composite: float, size: str = "md", show_numeric: bool = True,
) -> None:
    """Render diamond rating for a *signed* projection composite_score.

    This normalizes the signed delta (roughly -1 to +1.2) into 0-1 first.

    Args:
        composite: Raw signed composite_score from projection parquets.
        size: "sm", "md", or "lg".
        show_numeric: whether to show the numeric value beside the diamonds.
    """
    from lib.diamond_rating import normalize_composite

    render_diamond_rating(normalize_composite(composite), size=size, show_numeric=show_numeric)


def diamond_rating_html(
    score: float, size: str = "md", precomputed: float | None = None,
) -> str:
    """Return diamond rating as an HTML string (for use in tables/cards).

    Args:
        score: TDD normalized score (0.0-1.0), used as fallback.
        size: "sm", "md", or "lg".
        precomputed: Pre-computed diamond rating (0-10) from scouting grades.
            When provided, ``score`` is ignored and this value is used directly.
    """
    from lib.diamond_rating import score_to_diamonds, diamond_tier

    rating = precomputed if precomputed is not None else score_to_diamonds(score)
    tier = diamond_tier(rating)

    sizes = {"sm": "0.85rem", "md": "1.1rem", "lg": "1.4rem"}
    font_size = sizes.get(size, "1.1rem")
    diamonds = _build_diamonds(rating, font_size)

    return (
        f'<span title="Diamond Rating: {rating:.1f} / 10.0 — {tier}" '
        f'style="font-size:{font_size}; letter-spacing:2px; cursor:help;">'
        f'{diamonds} <span style="color:var(--tdd-gold); font-size:0.8em;">{rating:.1f}</span>'
        f'</span>'
    )


def diamond_rating_html_composite(composite: float, size: str = "md") -> str:
    """Return diamond rating HTML for a *signed* projection composite_score.

    Args:
        composite: Raw signed composite_score from projection parquets.
        size: "sm", "md", or "lg".
    """
    from lib.diamond_rating import normalize_composite

    return diamond_rating_html(normalize_composite(composite), size=size)


def diamond_rating_text(score: float) -> str:
    """Return a plain-text diamond rating string like '3.2 / 5'.

    Useful for ``st.dataframe`` cells where HTML is not supported.

    Args:
        score: TDD normalized score (0.0-1.0).
    """
    from lib.diamond_rating import score_to_diamonds

    rating = score_to_diamonds(score)
    return f"{rating:.1f}"


def diamond_rating_text_composite(composite: float) -> str:
    """Return plain-text diamond rating for a signed composite_score.

    Args:
        composite: Raw signed composite_score from projection parquets.
    """
    from lib.diamond_rating import normalize_composite

    return diamond_rating_text(normalize_composite(composite))
