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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Brand header */
    .brand-header {{
        background: {DARK_CARD};
        border: 1px solid {DARK_BORDER};
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .brand-title {{
        color: {GOLD};
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    .brand-subtitle {{
        color: {SLATE};
        font-size: 0.9rem;
        margin-top: 2px;
    }}

    /* Metric cards */
    .metric-card {{
        background: {DARK_CARD};
        border: 1px solid {DARK_BORDER};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .metric-value {{
        color: {GOLD};
        font-size: 1.6rem;
        font-weight: 700;
    }}
    .metric-label {{
        color: {SLATE};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}
    .metric-delta {{
        font-size: 0.85rem;
        margin-top: 4px;
    }}
    .metric-pctile {{
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 2px;
        letter-spacing: 0.5px;
    }}

    /* Insight cards */
    .insight-card {{
        background: {DARK_CARD};
        border: 1px solid {DARK_BORDER};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .insight-title {{
        color: {GOLD};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }}
    .insight-bullet {{
        color: {CREAM};
        font-size: 0.92rem;
        line-height: 1.7;
        padding-left: 0.5rem;
    }}
    .insight-bullet .dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }}

    /* Percentile bars */
    .pctile-row {{
        display: flex;
        align-items: center;
        margin: 10px 0;
    }}
    .pctile-label {{
        width: 65px;
        color: {SLATE};
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .pctile-bar-bg {{
        flex: 1;
        background: {DARK};
        border-radius: 6px;
        height: 22px;
        position: relative;
        overflow: hidden;
        border: 1px solid {DARK_BORDER};
    }}
    .pctile-bar-fill {{
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }}
    .pctile-info {{
        width: 220px;
        text-align: right;
        color: {SLATE};
        font-size: 0.8rem;
        padding-left: 12px;
    }}

    /* Delta colors */
    .delta-pos {{ color: {POSITIVE}; font-weight: 600; }}
    .delta-neg {{ color: {NEGATIVE}; font-weight: 600; }}
    .delta-neutral {{ color: {SLATE}; font-weight: 600; }}

    /* Table styling */
    .stDataFrame {{ font-size: 0.85rem; }}

    /* Sidebar branding */
    [data-testid="stSidebar"] {{
        padding-top: 0;
    }}
    [data-testid="stSidebar"] [data-testid="stImage"] {{
        display: flex;
        justify-content: center;
        padding-top: 1rem;
    }}
    .sidebar-brand {{
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }}
    .sidebar-brand-name {{
        color: {GOLD};
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 2px;
    }}
    .sidebar-brand-sub {{
        color: {SLATE};
        font-size: 0.75rem;
        margin-top: 4px;
    }}

    /* Sidebar nav buttons */
    div[data-testid="stSidebar"] .nav-container .stButton > button {{
        background: none;
        border: none;
        color: {SLATE};
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0.45rem 0.8rem;
        width: 100%;
        text-align: left;
        cursor: pointer;
        border-radius: 6px;
        transition: background 0.15s, color 0.15s;
    }}
    div[data-testid="stSidebar"] .nav-container .stButton > button:hover {{
        background: {DARK_BORDER};
        color: {CREAM};
    }}
    div[data-testid="stSidebar"] .nav-container .stButton > button[kind="primary"] {{
        background: {DARK_BORDER};
        color: {GOLD};
        font-weight: 600;
    }}

    /* Section dividers */
    .section-header {{
        color: {GOLD};
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid {DARK_BORDER};
    }}

    /* Pitch profile tables */
    .pitch-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }}
    .pitch-table th {{
        color: {SLATE};
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 6px 10px;
        border-bottom: 1px solid {DARK_BORDER};
        text-align: left;
    }}
    .pitch-table td {{
        color: {CREAM};
        padding: 8px 10px;
        border-bottom: 1px solid {DARK_BORDER}22;
        vertical-align: middle;
    }}
    .pitch-table tr:last-child td {{
        border-bottom: none;
    }}
    .pitch-table .pt-name {{
        font-weight: 600;
        white-space: nowrap;
    }}
    .pitch-table .pt-n {{
        color: {SLATE};
        font-size: 0.75rem;
    }}
    .spark-cell {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .spark-bar {{
        height: 14px;
        border-radius: 7px;
        min-width: 3px;
    }}
    .spark-val {{
        font-size: 0.82rem;
        font-weight: 500;
        min-width: 36px;
    }}

    /* Matchup edge indicator */
    .edge-dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin: 0 auto;
    }}
    .matchup-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }}
    .matchup-table th {{
        color: {SLATE};
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 6px 8px;
        border-bottom: 1px solid {DARK_BORDER};
        text-align: left;
    }}
    .matchup-table td {{
        color: {CREAM};
        padding: 7px 8px;
        border-bottom: 1px solid {DARK_BORDER}22;
        vertical-align: middle;
    }}
    .matchup-table tr:last-child td {{
        border-bottom: none;
    }}
    .matchup-table .pt-name {{
        font-weight: 600;
        white-space: nowrap;
    }}
    .matchup-table .pt-n {{
        color: {SLATE};
        font-size: 0.75rem;
    }}
</style>
""", unsafe_allow_html=True)

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
