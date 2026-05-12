"""
The Data Diamond -- MLB Bayesian Projection Dashboard.

Interactive Streamlit app for exploring hierarchical Bayesian player
projections, posterior distributions, and game-level stat predictions.

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
)
from utils.helpers import check_data_exists  # noqa: E402
from components.charts import apply_dark_mpl  # noqa: E402

# Page imports
from views.schedule import page_schedule  # noqa: E402
from views.projections import page_projections  # noqa: E402
from views.stats import page_stats  # noqa: E402
from views.diamond_daily import page_diamond_daily  # noqa: E402
from views.player_profile import page_player_profile  # noqa: E402
from views.team_overview import page_team_overview  # noqa: E402
from views.matchup_explorer import page_matchup_explorer  # noqa: E402
from views.data_health import page_data_health  # noqa: E402
from views.model_performance import page_model_performance  # noqa: E402
from views.player_rankings import page_player_rankings  # noqa: E402
from views.team_rankings import page_team_rankings  # noqa: E402
from views.division_standings import page_division_standings  # noqa: E402
from views.compare import page_compare  # noqa: E402
from views.breakout import page_breakout  # noqa: E402
from views.lineup_creator import page_lineup_creator  # noqa: E402
from views.methodology import page_methodology  # noqa: E402
from views.projected_performers import page_projected_performers  # noqa: E402
from views.daily_preview import page_daily_preview  # noqa: E402
from views.game import page_game  # noqa: E402
from views.game_prep import page_game_prep  # noqa: E402
from views.home import page_home  # noqa: E402

# Apply dark matplotlib theme at import time
apply_dark_mpl()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="The Data Diamond | MLB Projections",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS: load styles.css (owns defaults); app.py only injects dynamic overrides
# ---------------------------------------------------------------------------
_css_path = PROJECT_ROOT / "assets" / "styles.css"
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# Remove Streamlit Community Cloud branding bar via JS
# (st.markdown strips <script> tags; components.html allows JS execution)
import streamlit.components.v1 as _components
_components.html("""
<script>
(function removeBranding() {
    const parent = window.parent.document;
    // Inject CSS into the parent frame to catch Cloud-injected elements
    if (!parent.getElementById('tdd-hide-branding')) {
        const style = parent.createElement('style');
        style.id = 'tdd-hide-branding';
        style.textContent = `
            footer, [data-testid="stBottom"],
            div[class*="viewerBadge"], a[class*="viewerBadge"],
            div[class*="profileContainer"], div[class*="StatusWidget"],
            [data-testid="stToolbar"], [data-testid="stDecoration"],
            div[class*="BottomBar"], div[class*="bottomBar"],
            div[class*="cloudToolbar"], div[class*="CloudToolbar"],
            [data-testid="stMainBottom"],
            div[class*="stBottom"], div[class*="toolbar"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                overflow: hidden !important;
                position: absolute !important;
                pointer-events: none !important;
            }
        `;
        parent.head.appendChild(style);
    }
    // Also walk the DOM for text-based matching
    function hide() {
        parent.querySelectorAll('a, span, div').forEach(el => {
            const txt = el.textContent || '';
            if (txt.includes('Hosted with Streamlit') ||
                txt.includes('Created by') && el.closest('[class*="bottom"], [class*="Bottom"], footer')) {
                let t = el;
                for (let i = 0; i < 8 && t.parentElement && t.parentElement !== parent.body; i++)
                    t = t.parentElement;
                t.style.cssText = 'display:none!important;height:0!important';
            }
        });
    }
    hide();
    new MutationObserver(hide).observe(parent.body, {childList:true, subtree:true});
})();
</script>
""", height=0)

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

# Load Google Fonts for the active pairing
if _font_pair["imports"]:
    families = "&family=".join(_font_pair["imports"])
    st.markdown(
        f'<link href="https://fonts.googleapis.com/css2?family={families}&display=swap" '
        f'rel="stylesheet">',
        unsafe_allow_html=True,
    )

_palette_name = st.session_state.get("tdd_palette", "Press Box")
_pal = _PALETTES.get(_palette_name, _PALETTES["Original (Gold & Cream)"])

# Override :root vars for the active palette + font pairing.
# styles.css owns the default values and all structural rules; this block
# only pushes dynamic values (user-selected theme) into CSS variables.
if _palette_name != "Original (Gold & Cream)" or _font_name != "Inter + IBM Plex Mono":
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
            --tdd-font-heading: {_font_pair["heading"]};
            --tdd-font-body: {_font_pair["body"]};
        }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
PAGES = {
    "Home": page_home,
    "Schedule": page_schedule,
    "Player Profile": page_player_profile,
    "Player Rankings": page_player_rankings,
    "Stats": page_stats,
    "The Diamond Daily": page_diamond_daily,
    "Projections": page_projections,
    "Breakout Candidates": page_breakout,
    "Team Overview": page_team_overview,
    "Team Rankings": page_team_rankings,
    "Division Standings": page_division_standings,
    "Matchup Explorer": page_matchup_explorer,
    "Compare Players": page_compare,
    "Lineup Creator": page_lineup_creator,
    "Model Performance": page_model_performance,
    "Data Health": page_data_health,
    "Daily Preview": page_daily_preview,
    "Props Lab": page_projected_performers,
    "Methodology": page_methodology,
    "Game Analysis": page_game,
    "Game Prep": page_game_prep,
}

PAGE_URL_MAP = {name.lower().replace(" ", "_"): name for name in PAGES}

# Nav structure: standalone items + dropdown groups
_NAV = [
    ("Home", None),
    ("Games", ["Schedule", "Game Analysis", "Game Prep", "Daily Preview", "Props Lab"]),
    ("Players", [
        "Player Profile", "Player Rankings", "Stats",
        "The Diamond Daily", "Projections", "Breakout Candidates",
    ]),
    ("Teams", ["Team Overview", "Team Rankings", "Division Standings"]),
    ("Tools", ["Matchup Explorer", "Compare Players", "Lineup Creator"]),
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
            active_attr = ' aria-current="page"' if page == label else ""
            cls = "topbar-btn topbar-btn-active" if page == label else "topbar-btn"
            nav_html += f'<a href="?page={slug}" target="_self" class="{cls}"{active_attr}>{label}</a>'
        else:
            # Dropdown group: trigger is a <button> so it is keyboard-focusable.
            # :focus-within on the wrapper keeps the menu visible while tabbing.
            group_active = page in children
            parent_cls = "topbar-btn topbar-btn-active" if group_active else "topbar-btn"
            items = ""
            for child in children:
                slug = child.lower().replace(" ", "_")
                active_attr = ' aria-current="page"' if page == child else ""
                child_cls = "topbar-dd-item topbar-dd-active" if page == child else "topbar-dd-item"
                items += f'<a href="?page={slug}" target="_self" class="{child_cls}" role="menuitem"{active_attr}>{child}</a>'
            nav_html += (
                f'<div class="topbar-dropdown">'
                f'<button type="button" class="{parent_cls}" aria-haspopup="true">'
                f'{label} <span class="topbar-arrow">&#9662;</span>'
                f'</button>'
                f'<div class="topbar-dd-menu" role="menu">{items}</div>'
                f'</div>'
            )

    # Mobile navigation: flat list inside details/summary
    mobile_nav_html = ""
    for label, children in _NAV:
        if children is None:
            slug = label.lower().replace(" ", "_")
            active = " topbar-mob-active" if page == label else ""
            aria = ' aria-current="page"' if page == label else ""
            mobile_nav_html += (
                f'<a href="?page={slug}" target="_self"'
                f' class="topbar-mob-item{active}"{aria}>{label}</a>'
            )
        else:
            mobile_nav_html += f'<div class="topbar-mob-group">{label}</div>'
            for child in children:
                slug = child.lower().replace(" ", "_")
                active = " topbar-mob-active" if page == child else ""
                aria = ' aria-current="page"' if page == child else ""
                mobile_nav_html += (
                    f'<a href="?page={slug}" target="_self"'
                    f' class="topbar-mob-item{active}"{aria}>{child}</a>'
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
        f'<img src="data:image/png;base64,{_logo_b64}" class="topbar-logo" alt="The Data Diamond">'
        if _logo_b64 else ""
    )

    st.markdown(f"""
    <div class="topbar-wrap">
    <nav class="topbar" aria-label="Main navigation">
        <a href="?page=home" target="_self" class="topbar-home">
            {_logo_html}
            <span class="topbar-brand-text">The Data Diamond</span>
        </a>
        <div class="topbar-nav-desktop">
            {nav_html}
        </div>
        <div class="topbar-right">
            {socials_html}
        </div>
        <details class="topbar-mobile-menu">
            <summary class="topbar-hamburger">&#9776;</summary>
            <div class="topbar-mobile-nav">
                {mobile_nav_html}
                <div class="topbar-mob-socials">{socials_html}</div>
            </div>
        </details>
    </nav>
    </div>
    """, unsafe_allow_html=True)

    if not check_data_exists():
        st.error(
            "Dashboard data not found. Run the pre-computation first:\n\n"
            "```bash\n"
            "python scripts/precompute_dashboard_data.py\n"
            "```"
        )
        return

    # Always write the page slug so the browser URL bar stays in sync.
    # (Streamlit may intercept anchor-tag clicks internally without updating
    # the URL bar; unconditionally setting the param forces replaceState.)
    _slug = page.lower().replace(" ", "_")
    st.query_params["page"] = _slug
    PAGES[page]()

    # Site-wide disclaimer
    st.markdown(
        '<div class="tdd-footer-disclaimer">For entertainment and research purposes. Not financial advice.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
