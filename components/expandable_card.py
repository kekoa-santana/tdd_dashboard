"""Expandable card component using native <details>/<summary>.

Zero JavaScript — mobile-friendly, accessible, and styled to match
the TDD dark-card aesthetic.
"""
from __future__ import annotations

EXPANDABLE_CARD_CSS = """
<style>
/* Expandable card — <details>/<summary> pattern */
.exp-card {
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--tdd-dark-border);
    border-radius: 0;
    margin-bottom: 0.35rem;
    overflow: hidden;
}
.exp-card details {
    width: 100%;
}
.exp-card summary {
    list-style: none;
    cursor: pointer;
    padding: 0.45rem 1rem;
    display: flex;
    align-items: center;
    gap: 0;
    user-select: none;
}
.exp-card summary::-webkit-details-marker { display: none; }
.exp-card summary::marker { display: none; content: ""; }
/* Expand indicator — right triangle rotates on open */
.exp-indicator {
    color: var(--tdd-slate);
    font-size: 0.6rem;
    margin-left: auto;
    padding-left: 0.5rem;
    transition: transform 0.2s ease;
    flex-shrink: 0;
}
.exp-card details[open] .exp-indicator {
    transform: rotate(90deg);
}
.exp-detail {
    padding: 0.8rem 1rem 1rem;
    border-top: 1px solid var(--tdd-dark-border);
}

/* Responsive — tablet */
@media (max-width: 768px) {
    .exp-card summary { padding: 0.4rem 0.8rem; }
    .exp-detail { padding: 0.6rem 0.8rem; }
}
/* Responsive — phone */
@media (max-width: 480px) {
    .exp-card summary { padding: 0.35rem 0.6rem; font-size: 0.85rem; }
    .exp-detail { padding: 0.5rem 0.6rem; }
}
</style>
"""


def expandable_card_html(
    summary_html: str,
    detail_html: str,
    *,
    open_default: bool = False,
) -> str:
    """Return an expandable card as an HTML string.

    Parameters
    ----------
    summary_html : str
        HTML for the always-visible summary row.
    detail_html : str
        HTML revealed when expanded.
    open_default : bool
        Whether the card starts expanded.
    """
    open_attr = " open" if open_default else ""
    return (
        f'<div class="exp-card">'
        f'<details{open_attr}>'
        f'<summary>{summary_html}<span class="exp-indicator">&#9654;</span></summary>'
        f'<div class="exp-detail">{detail_html}</div>'
        f'</details></div>'
    )
