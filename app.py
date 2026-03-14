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
    GOLD, SLATE, DARK_CARD, DARK_BORDER, CREAM, DARK, ICON_PATH,
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
from views.game_k_sim import page_game_k_sim  # noqa: E402
from views.preseason_snapshot import page_preseason_snapshot  # noqa: E402
from views.data_health import page_data_health  # noqa: E402
from views.model_performance import page_model_performance  # noqa: E402

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
# Custom CSS — variables injected inline, rules loaded from external file
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    :root {{
        --tdd-gold: {GOLD};
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

_css_path = PROJECT_ROOT / "assets" / "styles.css"
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
PAGES = {
    "Schedule": page_schedule,
    "Projections": page_projections,
    "Stats": page_stats,
    "Player Profile": page_player_profile,
    "Team Overview": page_team_overview,
    "Matchup Explorer": page_matchup_explorer,
    "Game K Simulator": page_game_k_sim,
    "Preseason Snapshot": page_preseason_snapshot,
    "Model Performance": page_model_performance,
    "Data Health": page_data_health,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Sidebar
    with st.sidebar:
        if ICON_PATH.exists():
            _, icon_col, _ = st.columns([1, 2, 1])
            with icon_col:
                st.image(str(ICON_PATH), width=110)
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">THE DATA DIAMOND</div>
            <div class="sidebar-brand-sub">KEKOA SANTANA</div>
            <div class="sidebar-brand-sub">Bayesian MLB Projections</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        # Initialize active page in session state
        if "active_page" not in st.session_state:
            st.session_state.active_page = list(PAGES.keys())[0]

        with st.container():
            st.markdown('<div class="nav-container">', unsafe_allow_html=True)
            for page_name in PAGES:
                is_active = st.session_state.active_page == page_name
                btn_type = "primary" if is_active else "secondary"
                if st.button(page_name, key=f"nav_{page_name}", type=btn_type,
                             use_container_width=True):
                    st.session_state.active_page = page_name
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        page = st.session_state.active_page
        st.markdown("---")
        # Update timestamp
        meta = load_update_metadata()
        updated_str = ""
        if meta.get("last_updated"):
            try:
                from datetime import datetime as _dt
                ts = _dt.fromisoformat(meta["last_updated"])
                updated_str = f"<br>Updated: {ts.strftime('%b %d, %I:%M %p')}"
            except Exception:
                pass
        st.markdown(
            f'<div style="color:{SLATE}; font-size:0.75rem; text-align:center;">'
            f'v2.0 | {CURRENT_SEASON} Season<br>'
            f'Trained on {TRAINING_RANGE}{updated_str}</div>',
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
    PAGES[page]()


if __name__ == "__main__":
    main()
