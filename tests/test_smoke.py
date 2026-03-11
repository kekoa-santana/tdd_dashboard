"""Smoke tests for the TDD Dashboard.

These tests verify that:
- All modules import without errors
- Config constants have the right types and sane values
- Utility/formatter functions produce expected output
- Data loaders can read fixture parquet files
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path (conftest.py also does this)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# 1. Import tests
# =====================================================================
class TestImports:
    """Verify that every module imports without error."""

    def test_import_config(self):
        from config import CURRENT_SEASON, PROJECTION_LABEL, AVAILABLE_SEASONS  # noqa: F401

    def test_import_data_loader(self):
        from services.data_loader import load_projections  # noqa: F401

    def test_import_charts(self):
        from components.charts import apply_dark_mpl  # noqa: F401

    def test_import_metric_cards(self):
        from components.metric_cards import metric_card  # noqa: F401

    def test_import_tables(self):
        from components.tables import build_hitter_profile_table  # noqa: F401

    def test_import_scouting(self):
        from components.scouting import generate_scouting_bullets  # noqa: F401

    def test_import_helpers(self):
        from utils.helpers import strip_accents  # noqa: F401

    def test_import_formatters(self):
        from utils.formatters import fmt_pct, fmt_stat  # noqa: F401

    def test_import_page_schedule(self):
        from views.schedule import page_schedule  # noqa: F401

    def test_import_page_projections(self):
        from views.projections import page_projections  # noqa: F401

    def test_import_page_player_profile(self):
        from views.player_profile import page_player_profile  # noqa: F401

    def test_import_page_team_overview(self):
        from views.team_overview import page_team_overview  # noqa: F401

    def test_import_page_matchup_explorer(self):
        from views.matchup_explorer import page_matchup_explorer  # noqa: F401

    def test_import_page_game_k_sim(self):
        from views.game_k_sim import page_game_k_sim  # noqa: F401

    def test_import_page_preseason_snapshot(self):
        from views.preseason_snapshot import page_preseason_snapshot  # noqa: F401

    def test_import_lib_theme(self):
        from lib.theme import GOLD, EMBER, SAGE, SLATE, CREAM, DARK  # noqa: F401

    def test_import_lib_constants(self):
        from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE  # noqa: F401

    def test_import_lib_matchup(self):
        from lib.matchup import score_matchup  # noqa: F401

    def test_import_lib_bf_model(self):
        from lib.bf_model import get_bf_distribution  # noqa: F401

    def test_import_lib_game_k_model(self):
        from lib.game_k_model import simulate_game_ks  # noqa: F401


# =====================================================================
# 2. Config tests
# =====================================================================
class TestConfig:
    """Verify that runtime config values are sane."""

    def test_current_season_is_int(self):
        from config import CURRENT_SEASON
        assert isinstance(CURRENT_SEASON, int)
        assert CURRENT_SEASON > 2020

    def test_available_seasons_descending(self):
        from config import AVAILABLE_SEASONS
        assert isinstance(AVAILABLE_SEASONS, list)
        assert len(AVAILABLE_SEASONS) > 0
        assert AVAILABLE_SEASONS == sorted(AVAILABLE_SEASONS, reverse=True)

    def test_projection_label_contains_season(self):
        from config import PROJECTION_LABEL, CURRENT_SEASON
        assert str(CURRENT_SEASON) in PROJECTION_LABEL

    def test_dashboard_dir_attribute(self):
        from config import DASHBOARD_DIR
        assert isinstance(DASHBOARD_DIR, Path)

    def test_brand_colors_are_hex(self):
        from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK
        for color in (GOLD, EMBER, SAGE, SLATE, CREAM, DARK):
            assert isinstance(color, str)
            assert color.startswith("#")


# =====================================================================
# 3. Formatter tests
# =====================================================================
class TestFormatters:
    """Verify that formatting functions produce expected output."""

    def test_fmt_pct(self):
        from utils.formatters import fmt_pct
        result = fmt_pct(0.253)
        assert isinstance(result, str)
        assert "25" in result

    def test_fmt_pct_zero(self):
        from utils.formatters import fmt_pct
        result = fmt_pct(0.0)
        assert "0" in result

    def test_fmt_stat_k_rate(self):
        from utils.formatters import fmt_stat
        result = fmt_stat(0.253, "k_rate")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fmt_stat_avg_exit_velo(self):
        from utils.formatters import fmt_stat
        result = fmt_stat(93.5, "avg_exit_velo")
        assert "93" in result

    def test_strip_accents(self):
        from utils.helpers import strip_accents
        assert strip_accents("Jose") == "Jose"
        # Decomposed form: e + combining acute accent
        assert strip_accents("Jose\u0301") == "Jose"
        # Pre-composed form: e-with-acute
        assert strip_accents("Jos\u00e9") == "Jose"
        # Verify output contains no non-ASCII
        result = strip_accents("Jos\u00e9")
        assert all(ord(c) < 128 for c in result)

    def test_metric_card_returns_html(self):
        from components.metric_cards import metric_card
        html = metric_card("K%", "25.3%")
        assert isinstance(html, str)
        assert "metric-card" in html
        assert "25.3%" in html

    def test_delta_html(self):
        from utils.formatters import delta_html
        result = delta_html(0.025, higher_is_better=True)
        assert "pp" in result

    def test_fmt_trad(self):
        from utils.formatters import fmt_trad
        assert fmt_trad(0.301, ".000") == "0.301"
        assert fmt_trad(3.45, "0.00") == "3.45"


# =====================================================================
# 4. Data loader tests (using fixture data)
# =====================================================================
class TestDataLoaders:
    """Verify data loaders can read fixture parquet files."""

    def test_load_hitter_projections(self, dashboard_dir):
        from services.data_loader import load_projections
        df = load_projections("hitter")
        assert not df.empty
        assert "batter_id" in df.columns
        assert "batter_name" in df.columns
        assert "projected_k_rate" in df.columns
        assert "projected_bb_rate" in df.columns
        assert "composite_score" in df.columns
        assert len(df) == 2  # fixture has exactly 2 rows

    def test_load_pitcher_projections(self, dashboard_dir):
        from services.data_loader import load_projections
        df = load_projections("pitcher")
        assert not df.empty
        assert "pitcher_id" in df.columns
        assert "pitcher_name" in df.columns
        assert "projected_k_rate" in df.columns
        assert "is_starter" in df.columns
        assert len(df) == 2  # fixture has exactly 2 rows

    def test_load_player_teams(self, dashboard_dir):
        from services.data_loader import load_player_teams
        df = load_player_teams()
        assert not df.empty
        assert "player_id" in df.columns
        assert "team_abbr" in df.columns

    def test_load_update_metadata(self, dashboard_dir):
        from services.data_loader import load_update_metadata
        meta = load_update_metadata()
        assert isinstance(meta, dict)
        assert "season" in meta
        assert meta["season"] == 2026
        assert "last_updated" in meta

    def test_load_projections_missing_file(self, dashboard_dir):
        """Loader returns empty DataFrame for missing files."""
        from services.data_loader import load_projections
        df = load_projections("nonexistent_type")
        assert df.empty

    def test_load_metadata_missing_file(self, tmp_path, monkeypatch):
        """Loader returns empty dict for missing metadata."""
        import config
        import services.data_loader
        monkeypatch.setattr(config, "DASHBOARD_DIR", tmp_path)
        monkeypatch.setattr(services.data_loader, "DASHBOARD_DIR", tmp_path)
        from services.data_loader import load_update_metadata
        meta = load_update_metadata()
        assert meta == {}

    def test_check_data_exists(self, dashboard_dir):
        from utils.helpers import check_data_exists
        assert check_data_exists() is True

    def test_check_data_exists_missing(self, tmp_path, monkeypatch):
        import config
        import utils.helpers
        monkeypatch.setattr(config, "DASHBOARD_DIR", tmp_path)
        monkeypatch.setattr(utils.helpers, "DASHBOARD_DIR", tmp_path)
        from utils.helpers import check_data_exists
        assert check_data_exists() is False
