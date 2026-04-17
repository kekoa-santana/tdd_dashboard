"""HTML escaping helpers for dynamic strings rendered via unsafe_allow_html."""
from __future__ import annotations

import html as _html


def esc(value) -> str:
    """Escape a value for use as HTML text content.

    Converts to str first so callers can pass ints, floats, etc.
    Returns an empty string for None.
    """
    if value is None:
        return ""
    return _html.escape(str(value), quote=False)


def esc_attr(value) -> str:
    """Escape a value for use inside an HTML attribute (quoted).

    Same as esc() but also escapes quotes.
    """
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)
