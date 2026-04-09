"""Daily Preview -- game-by-game scouting narratives for today's slate."""
from __future__ import annotations

import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DASHBOARD_DIR
from lib.fantasy_report import (
    load_report_data,
    analyze_game,
    PitcherReport,
    GameReport,
)


def _pitcher_section_html(report: PitcherReport) -> str:
    """Render a single pitcher's scouting report as styled HTML."""
    parts: list[str] = []

    # Header
    label = f"vs {report.opp_abbr}"
    parts.append(
        f'<div style="font-weight:700; font-size:1rem; margin-bottom:0.3rem;">'
        f'{report.pitcher_name} '
        f'<span style="color:{SLATE};">({report.team_abbr})</span> '
        f'<span style="color:{SLATE}; font-weight:400;">{label}</span>'
        f'</div>'
    )

    # Stat line
    stat_parts = []
    if report.expected_k > 0:
        stat_parts.append(f"{report.expected_k:.1f} K")
    if report.expected_ip > 0:
        stat_parts.append(f"{report.expected_ip:.1f} IP")
    if report.expected_bb > 0:
        stat_parts.append(f"{report.expected_bb:.1f} BB")
    if report.dk_mean > 0:
        stat_parts.append(f"{report.dk_mean:.1f} DK pts")
    if report.archetype:
        stat_parts.append(report.archetype)
    if stat_parts:
        lineup_note = "" if report.has_lineup else " &middot; projected lineup"
        parts.append(
            f'<div style="font-size:0.8rem; color:{SLATE}; margin-bottom:0.6rem;">'
            f'{" &middot; ".join(stat_parts)}{lineup_note}</div>'
        )

    # Bullet sections
    sections = [
        ("ADVANTAGES", SAGE, report.advantages),
        ("STRUGGLES", EMBER, report.struggles),
        ("BATTERS TO WATCH", GOLD, report.key_batters),
        ("BATTERS WHO WILL STRUGGLE", SLATE, report.struggling_batters),
        ("BULLPEN", SLATE, report.bullpen_notes),
        (f"{report.opp_abbr} BATTING OUTLOOK", GOLD, report.batting_outlook),
    ]

    for title, color, bullets in sections:
        if not bullets:
            continue
        bullet_html = "".join(
            f'<li style="margin-bottom:0.25rem;">{b}</li>' for b in bullets
        )
        parts.append(
            f'<div style="margin-bottom:0.6rem;">'
            f'<div style="color:{color}; font-weight:600; font-size:0.7rem; '
            f'letter-spacing:0.5px; margin-bottom:0.2rem;">{title}</div>'
            f'<ul style="margin:0; padding-left:1.2rem; font-size:0.82rem; '
            f'color:{CREAM};">{bullet_html}</ul></div>'
        )

    return "".join(parts)


def _game_card_html(game: GameReport) -> str:
    """Render a full game card."""
    parts: list[str] = []

    # Game header
    parts.append(
        f'<div style="font-size:1.1rem; font-weight:700; margin-bottom:0.5rem; '
        f'color:{GOLD};">'
        f'{game.away_abbr} @ {game.home_abbr} '
        f'<span style="font-weight:400; font-size:0.85rem; color:{SLATE};">'
        f'{game.game_time}</span></div>'
    )

    # Two-column layout content
    if game.away:
        parts.append(_pitcher_section_html(game.away))
    else:
        parts.append(
            f'<div style="color:{SLATE}; font-size:0.85rem; margin-bottom:0.5rem;">'
            f'{game.away_abbr} starter: TBD</div>'
        )

    parts.append('<hr style="border-color:var(--tdd-dark-border); margin:0.6rem 0;">')

    if game.home:
        parts.append(_pitcher_section_html(game.home))
    else:
        parts.append(
            f'<div style="color:{SLATE}; font-size:0.85rem;">'
            f'{game.home_abbr} starter: TBD</div>'
        )

    return (
        f'<div style="background:var(--tdd-dark-card); border:1px solid '
        f'var(--tdd-dark-border); border-radius:8px; padding:1rem; '
        f'margin-bottom:1rem;">{"".join(parts)}</div>'
    )


def page_daily_preview() -> None:
    """Render the Daily Preview page."""
    st.markdown(
        f'<h2 style="margin-bottom:0.2rem;">Daily Preview</h2>'
        f'<p style="color:{SLATE}; font-size:0.85rem; margin-bottom:1rem;">'
        f'Game-by-game scouting narratives for today\'s slate</p>',
        unsafe_allow_html=True,
    )

    data = load_report_data(DASHBOARD_DIR)

    if data.games.empty:
        st.info("No games scheduled today.")
        return

    st.markdown(
        f'<div style="color:{SLATE}; font-size:0.82rem; margin-bottom:1rem;">'
        f'{len(data.games)} games on today\'s slate</div>',
        unsafe_allow_html=True,
    )

    for _, game_row in data.games.iterrows():
        game_report = analyze_game(game_row, data)
        st.markdown(_game_card_html(game_report), unsafe_allow_html=True)
