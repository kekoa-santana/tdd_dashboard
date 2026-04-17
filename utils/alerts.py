"""Branded alert components replacing Streamlit's native st.info/st.warning.

Use these instead of st.info(), st.warning() to maintain dark-theme
consistency. st.caption() is acceptable for footnotes.
"""
from __future__ import annotations

import streamlit as st

from utils.html import esc


def tdd_info(msg: str) -> None:
    """Branded info callout (slate accent, replaces st.info)."""
    st.markdown(
        f'<div class="tdd-callout-info">{esc(msg)}</div>',
        unsafe_allow_html=True,
    )


def tdd_warn(msg: str) -> None:
    """Branded warning callout (ember accent, replaces st.warning)."""
    st.markdown(
        f'<div class="tdd-callout-warn">{esc(msg)}</div>',
        unsafe_allow_html=True,
    )


def tdd_note(msg: str) -> None:
    """Branded gold callout (replaces st.info for important notices)."""
    st.markdown(
        f'<div class="tdd-callout">{esc(msg)}</div>',
        unsafe_allow_html=True,
    )
