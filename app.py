"""
The Data Diamond -- MLB Bayesian Projection Dashboard.

Interactive Streamlit app for exploring hierarchical Bayesian player
projections, posterior distributions, and game-level K predictions.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    GOLD, EMBER, SAGE, SLATE, DARK_CARD, DARK_BORDER, CREAM, DARK, ICON_PATH,
    POSITIVE, NEGATIVE,
    CURRENT_SEASON, TRAINING_RANGE,
)
from services.data_loader import load_update_metadata  # noqa: E402
from utils.helpers import check_data_exists  # noqa: E402
from components.charts import apply_dark_mpl  # noqa: E402

# Page imports
from views.schedule import page_schedule  # noqa: E402
from views.projections import page_projections  # noqa: E402
from views.stats import page_stats  # noqa: E402
from views.player_profile import page_player_profile  # noqa: E402
from views.team_overview import page_team_overview  # noqa: E402
from views.matchup_explorer import page_matchup_explorer  # noqa: E402
from views.data_health import page_data_health  # noqa: E402
from views.model_performance import page_model_performance  # noqa: E402
from views.player_rankings import page_player_rankings  # noqa: E402
from views.team_rankings import page_team_rankings  # noqa: E402
from views.compare import page_compare  # noqa: E402
from views.breakout import page_breakout  # noqa: E402
from views.methodology import page_methodology  # noqa: E402

# Apply dark matplotlib theme at import time
apply_dark_mpl()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="The Data Diamond | MLB Projections",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — load styles.css, then inject :root variables (only dynamic values)
# ---------------------------------------------------------------------------
_css_path = PROJECT_ROOT / "assets" / "styles.css"
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

st.markdown(f"""
<style>
    :root {{
        --tdd-gold: {GOLD};
        --tdd-ember: {EMBER};
        --tdd-sage: {SAGE};
        --tdd-slate: {SLATE};
        --tdd-cream: {CREAM};
        --tdd-dark: {DARK};
        --tdd-dark-card: {DARK_CARD};
        --tdd-dark-border: {DARK_BORDER};
        --tdd-dark-border-faint: {DARK_BORDER}22;
        --tdd-positive: {POSITIVE};
        --tdd-negative: {NEGATIVE};
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Color palettes — switch via sidebar dropdown
# ---------------------------------------------------------------------------
_PALETTES = {
    "Original (Gold & Cream)": {
        "gold": GOLD, "ember": EMBER, "sage": SAGE, "slate": SLATE,
        "cream": CREAM, "dark": DARK, "dark_card": DARK_CARD,
        "dark_border": DARK_BORDER, "positive": POSITIVE, "negative": NEGATIVE,
    },
    "Midnight Arctic": {
        "gold": "#64B5F6", "ember": "#FF6B6B", "sage": "#4DB6AC", "slate": "#8899AA",
        "cream": "#E8EEF2", "dark": "#0B1120", "dark_card": "#111B2E",
        "dark_border": "#1E2D44", "positive": "#4DB6AC", "negative": "#FF6B6B",
    },
    "Obsidian Copper": {
        "gold": "#D4856A", "ember": "#C45C5C", "sage": "#5E9E8B", "slate": "#8A8580",
        "cream": "#D4CFC9", "dark": "#101014", "dark_card": "#1A1A22",
        "dark_border": "#2A2A32", "positive": "#5E9E8B", "negative": "#C45C5C",
    },
    "Deep Space": {
        "gold": "#9C7CF4", "ember": "#F59E0B", "sage": "#4ADE80", "slate": "#8B8FA6",
        "cream": "#E0D8F0", "dark": "#0D0D1A", "dark_card": "#151528",
        "dark_border": "#252540", "positive": "#4ADE80", "negative": "#F59E0B",
    },
    "Darkroom": {
        "gold": "#C45C5C", "ember": "#D4856A", "sage": "#7A99AC", "slate": "#8A8580",
        "cream": "#F0EDE6", "dark": "#0C0C0C", "dark_card": "#161616",
        "dark_border": "#282828", "positive": "#7A99AC", "negative": "#C45C5C",
    },
    "Press Box": {
        "gold": "#C8A96E", "ember": "#C47A5A", "sage": "#6BA38E", "slate": "#9B9488",
        "cream": "#E8E4DC", "dark": "#121210", "dark_card": "#1A1917",
        "dark_border": "#2A2924", "positive": "#6BA38E", "negative": "#C47A5A",
    },
    "Pacific Dusk": {
        "gold": "#E8875C", "ember": "#D4562A", "sage": "#5B8FA8", "slate": "#8B8FA6",
        "cream": "#E5E1DB", "dark": "#0F1923", "dark_card": "#172230",
        "dark_border": "#243040", "positive": "#5B8FA8", "negative": "#D4562A",
    },
}

# ---------------------------------------------------------------------------
# Font pairings — heading + body, switch via Settings
# ---------------------------------------------------------------------------
_FONT_PAIRINGS = {
    "System Default": {
        "heading": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        "body": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        "imports": [],
    },
    "Inter + IBM Plex Mono": {
        "heading": '"Inter", sans-serif',
        "body": '"IBM Plex Mono", monospace',
        "imports": ["Inter:wght@400;500;600;700;800", "IBM+Plex+Mono:wght@400;500;600"],
    },
    "DM Sans + DM Mono": {
        "heading": '"DM Sans", sans-serif',
        "body": '"DM Mono", monospace',
        "imports": ["DM+Sans:wght@400;500;600;700;800", "DM+Mono:wght@400;500"],
    },
    "Space Grotesk + IBM Plex Sans": {
        "heading": '"Space Grotesk", sans-serif',
        "body": '"IBM Plex Sans", sans-serif',
        "imports": ["Space+Grotesk:wght@400;500;600;700", "IBM+Plex+Sans:wght@400;500;600"],
    },
    "Sora + Inter": {
        "heading": '"Sora", sans-serif',
        "body": '"Inter", sans-serif',
        "imports": ["Sora:wght@400;500;600;700;800", "Inter:wght@400;500;600"],
    },
    "Outfit + Source Sans 3": {
        "heading": '"Outfit", sans-serif',
        "body": '"Source Sans 3", sans-serif',
        "imports": ["Outfit:wght@400;500;600;700;800", "Source+Sans+3:wght@400;500;600"],
    },
    "Bricolage Grotesque + IBM Plex Mono": {
        "heading": '"Bricolage Grotesque", sans-serif',
        "body": '"IBM Plex Mono", monospace',
        "imports": ["Bricolage+Grotesque:wght@400;500;600;700;800", "IBM+Plex+Mono:wght@400;500;600"],
    },
    "Fraunces + DM Mono": {
        "heading": '"Fraunces", serif',
        "body": '"DM Mono", monospace',
        "imports": ["Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,800", "DM+Mono:wght@400;500"],
    },
    "Space Grotesk + JetBrains Mono": {
        "heading": '"Space Grotesk", sans-serif',
        "body": '"JetBrains Mono", monospace',
        "imports": ["Space+Grotesk:wght@400;500;600;700", "JetBrains+Mono:wght@400;500;600"],
    },
    "Instrument Serif + IBM Plex Mono": {
        "heading": '"Instrument Serif", serif',
        "body": '"IBM Plex Mono", monospace',
        "imports": ["Instrument+Serif", "IBM+Plex+Mono:wght@400;500;600"],
    },
}

_font_name = st.session_state.get("tdd_font", "Inter + IBM Plex Mono")
_font_pair = _FONT_PAIRINGS.get(_font_name, _FONT_PAIRINGS["System Default"])

# Load Google Fonts
if _font_pair["imports"]:
    families = "&family=".join(_font_pair["imports"])
    st.markdown(
        f'<link href="https://fonts.googleapis.com/css2?family={families}&display=swap" '
        f'rel="stylesheet">',
        unsafe_allow_html=True,
    )

# Apply fonts via CSS variables — wildcard selectors to override Streamlit
st.markdown(f"""
<style>
    :root {{
        --tdd-font-heading: {_font_pair["heading"]};
        --tdd-font-body: {_font_pair["body"]};
    }}
    /* Body font — everything by default */
    html, body, .stApp, .stMarkdown, .stMarkdown p,
    div, td, th, li, label, input, select, textarea, button,
    [data-testid="stAppViewContainer"],
    [data-baseweb] {{
        font-family: var(--tdd-font-body) !important;
    }}
    /* Heading font — brand, section headers, player/team names */
    .topbar-brand-text,
    .tdd-section-hdr,
    .tdd-team-abbr,
    .tdd-player-name,
    .tdd-player-name-lg,
    .tdd-edge-label,
    .brand-title,
    .lb-title,
    .lb-name,
    .proj-title,
    h1, h2, h3 {{
        font-family: var(--tdd-font-heading) !important;
    }}
    /* Tabular nums for stat values */
    .tdd-stat-value, .lb-val, .metric-value {{
        font-variant-numeric: tabular-nums;
    }}
</style>
""", unsafe_allow_html=True)

_palette_name = st.session_state.get("tdd_palette", "Press Box")
_pal = _PALETTES.get(_palette_name, _PALETTES["Original (Gold & Cream)"])

# Override :root variables for the active palette
# (styles.css uses var(--tdd-*) everywhere, so this recolors the whole site)
if _palette_name != "Original (Gold & Cream)":
    st.markdown(f"""
    <style>
        :root {{
            --tdd-gold: {_pal["gold"]};
            --tdd-ember: {_pal["ember"]};
            --tdd-sage: {_pal["sage"]};
            --tdd-slate: {_pal["slate"]};
            --tdd-cream: {_pal["cream"]};
            --tdd-dark: {_pal["dark"]};
            --tdd-dark-card: {_pal["dark_card"]};
            --tdd-dark-border: {_pal["dark_border"]};
            --tdd-dark-border-faint: {_pal["dark_border"]}22;
            --tdd-positive: {_pal["positive"]};
            --tdd-negative: {_pal["negative"]};
        }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
PAGES = {
    "Schedule": page_schedule,
    "Player Profile": page_player_profile,
    "Player Rankings": page_player_rankings,
    "Stats": page_stats,
    "Projections": page_projections,
    "Breakout Candidates": page_breakout,
    "Team Overview": page_team_overview,
    "Team Rankings": page_team_rankings,
    "Matchup Explorer": page_matchup_explorer,
    "Compare Players": page_compare,
    "Model Performance": page_model_performance,
    "Data Health": page_data_health,
    "Methodology": page_methodology,
}

PAGE_URL_MAP = {name.lower().replace(" ", "_"): name for name in PAGES}

# Nav structure: standalone items + dropdown groups
_NAV = [
    ("Schedule", None),  # standalone
    ("Players", [
        "Player Profile", "Player Rankings", "Stats",
        "Projections", "Breakout Candidates",
    ]),
    ("Teams", ["Team Overview", "Team Rankings"]),
    ("Tools", ["Matchup Explorer", "Compare Players"]),
    ("About", ["Methodology", "Model Performance", "Data Health"]),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Always sync from query params so browser back/forward works
    page_param = st.query_params.get("page", "")
    if page_param in PAGE_URL_MAP:
        st.session_state.active_page = PAGE_URL_MAP[page_param]
    elif "active_page" not in st.session_state:
        st.session_state.active_page = list(PAGES.keys())[0]

    page = st.session_state.active_page

    # --- Sticky topbar with dropdown menus ---
    nav_html = ""
    for label, children in _NAV:
        if children is None:
            # Standalone button
            slug = label.lower().replace(" ", "_")
            cls = "topbar-btn topbar-btn-active" if page == label else "topbar-btn"
            nav_html += f'<a href="?page={slug}" target="_self" class="{cls}">{label}</a>'
        else:
            # Dropdown group — highlight if any child is active
            group_active = page in children
            parent_cls = "topbar-btn topbar-btn-active" if group_active else "topbar-btn"
            items = ""
            for child in children:
                slug = child.lower().replace(" ", "_")
                child_cls = "topbar-dd-item topbar-dd-active" if page == child else "topbar-dd-item"
                items += f'<a href="?page={slug}" target="_self" class="{child_cls}">{child}</a>'
            nav_html += (
                f'<div class="topbar-dropdown">'
                f'<span class="{parent_cls}">{label} <span class="topbar-arrow">&#9662;</span></span>'
                f'<div class="topbar-dd-menu">{items}</div>'
                f'</div>'
            )

    # Social links
    socials_html = (
        '<div class="topbar-socials">'
        '<a href="https://x.com/TheDataDiamond" target="_blank" class="topbar-social" title="X / Twitter">𝕏</a>'
        '<a href="https://instagram.com/TheDataDiamond" target="_blank" class="topbar-social" title="Instagram">⌘</a>'
        '</div>'
    )

    _logo_b64 = ""
    if ICON_PATH.exists():
        import base64
        _logo_b64 = base64.b64encode(ICON_PATH.read_bytes()).decode()

    _logo_html = (
        f'<img src="data:image/png;base64,{_logo_b64}" class="topbar-logo">'
        if _logo_b64 else ""
    )

    st.markdown(f"""
    <div class="topbar-wrap">
    <div class="topbar">
        {_logo_html}
        <span class="topbar-brand-text">The Data Diamond</span>
        {nav_html}
        <div class="topbar-right">
            {socials_html}
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Settings bar
    with st.expander("Settings", expanded=False):
        col_pal, col_font, col_meta = st.columns([2, 2, 3])
        with col_pal:
            palette_choice = st.selectbox(
                "Color Palette",
                list(_PALETTES.keys()),
                index=list(_PALETTES.keys()).index(_palette_name),
                key="palette_selector",
                label_visibility="collapsed",
            )
            if palette_choice != _palette_name:
                st.session_state["tdd_palette"] = palette_choice
                st.rerun()
        with col_font:
            font_choice = st.selectbox(
                "Font Pairing",
                list(_FONT_PAIRINGS.keys()),
                index=list(_FONT_PAIRINGS.keys()).index(_font_name),
                key="font_selector",
                label_visibility="collapsed",
            )
            if font_choice != _font_name:
                st.session_state["tdd_font"] = font_choice
                st.rerun()
        with col_meta:
            meta = load_update_metadata()
            updated_str = ""
            if meta.get("last_updated"):
                try:
                    from datetime import datetime as _dt
                    ts = _dt.fromisoformat(meta["last_updated"])
                    updated_str = f"Updated: {ts.strftime('%b %d, %I:%M %p')} | "
                except Exception:
                    pass
            st.markdown(
                f'<div class="tdd-meta" style="text-align:right; padding-top:0.3rem;">'
                f'{updated_str}v2.0 | {CURRENT_SEASON} | {TRAINING_RANGE}</div>',
                unsafe_allow_html=True,
            )

    if not check_data_exists():
        st.error(
            "Dashboard data not found. Run the pre-computation first:\n\n"
            "```bash\n"
            "python scripts/precompute_dashboard_data.py\n"
            "```"
        )
        return

    # Dispatch to selected page
    st.query_params["page"] = page.lower().replace(" ", "_")
    PAGES[page]()


if __name__ == "__main__":
    main()
