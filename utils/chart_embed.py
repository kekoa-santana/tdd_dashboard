"""Utility for embedding matplotlib figures as base64 images in Streamlit HTML."""
from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt


def fig_to_base64(fig: plt.Figure, dpi: int = 100, fmt: str = "png") -> str:
    """Convert a matplotlib Figure to a base64-encoded data URI.

    Parameters
    ----------
    fig : plt.Figure
        The figure to encode.
    dpi : int
        Resolution for the rendered image.
    fmt : str
        Image format (png or jpeg).

    Returns
    -------
    str
        A ``data:image/...;base64,...`` string suitable for an HTML ``<img src=...>``.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none",
                transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"
