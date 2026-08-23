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

    def test_import_page_diamond_daily(self):
        from views.diamond_daily import page_diamond_daily  # noqa: F401

    def test_import_page_player_profile(self):
        from views.player_profile import page_player_profile  # noqa: F401

    def test_import_page_team_overview(self):
        from views.team_overview import page_team_overview  # noqa: F401

    def test_import_page_game_k_sim(self):
        """game_k_sim merged into schedule — verify schedule still imports."""
        from views.schedule import page_schedule  # noqa: F401

    def test_import_page_preseason_snapshot(self):
        """preseason_snapshot merged into model_performance."""
        from views.model_performance import page_model_performance  # noqa: F401

    def test_import_lib_theme(self):
        from lib.theme import GOLD, EMBER, SAGE, SLATE, CREAM, DARK  # noqa: F401

    def test_import_lib_constants(self):
        from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE  # noqa: F401

    def test_import_lib_bf_model(self):
        from lib.bf_model import get_bf_distribution  # noqa: F401

    def test_import_lib_game_k_model(self):
        from lib.game_k_model import simulate_game_ks, build_tto_logit_lifts  # noqa: F401

    def test_import_lib_rest_adjustment(self):
        from lib.rest_adjustment import get_rest_adjustment, apply_rest_to_bf  # noqa: F401

    def test_import_model_performance(self):
        from views.model_performance import page_model_performance  # noqa: F401

    def test_import_prospects(self):
        """prospects merged into player_rankings."""
        from views.player_rankings import page_player_rankings  # noqa: F401

    def test_import_compare(self):
        from views.compare import page_compare  # noqa: F401


    def test_import_simulate_game_stat(self):
        from lib.game_k_model import simulate_game_stat  # noqa: F401

    def test_import_compute_over_probs(self):
        from lib.game_k_model import compute_over_probs  # noqa: F401

    def test_import_leaderboard_renderer(self):
        from components.leaderboard import render_card  # noqa: F401

    def test_import_html_escape(self):
        from utils.html import esc, esc_attr  # noqa: F401


# =====================================================================
# 1b. Shared component functional tests
# =====================================================================
class TestLeaderboardRenderer:
    """Verify leaderboard renderer produces expected markup."""

    def test_render_card_basic(self):
        from components.leaderboard import render_card

        html = render_card(
            "Top 3 HR",
            [
                {"player_id": 1, "name": "Test", "value": "40", "team": "NYY"},
                {"player_id": 2, "name": "Other", "value": "38"},
            ],
        )
        assert 'class="lb-card"' in html
        assert 'class="lb-title"' in html
        assert "Top 3 HR" in html
        assert 'data-team="NYY"' in html

    def test_render_card_escapes_html(self):
        from components.leaderboard import render_card

        html = render_card(
            "<script>alert(1)</script>",
            [{"player_id": 1, "name": 'O"Brien <b>', "value": "5", "team": "A&B"}],
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert 'O"Brien &lt;b&gt;' in html
        assert "A&amp;B" in html

    def test_render_card_watch_rows(self):
        from components.leaderboard import render_card

        html = render_card(
            "Title",
            [{"player_id": 1, "name": "Main", "value": "10"}],
            watch_rows=[{"player_id": 2, "name": "Watcher", "value": "5", "team": "BOS"}],
        )
        assert "Players to Watch" in html
        assert "Watcher" in html
        assert 'class="lb-watch-row"' in html


class TestHtmlEscape:
    """Verify HTML escape helpers."""

    def test_esc_basic(self):
        from utils.html import esc
        assert esc("a<b>c") == "a&lt;b&gt;c"
        assert esc("a&b") == "a&amp;b"
        assert esc(None) == ""
        assert esc(42) == "42"

    def test_esc_attr_quotes(self):
        from utils.html import esc_attr
        assert esc_attr('a"b') == "a&quot;b"
        assert esc_attr("a'b") == "a&#x27;b"
        assert esc_attr(None) == ""


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


# =====================================================================
# 5. Backtest & snapshot tests
# =====================================================================
class TestBacktestLoaders:
    """Verify backtest and snapshot data loaders."""

    def test_load_backtest_pitcher_k(self, dashboard_dir):
        from services.data_loader import load_backtest
        df = load_backtest("pitcher_k_backtest")
        assert not df.empty
        assert "test_season" in df.columns
        assert "bayes_mae" in df.columns

    def test_load_backtest_game_k(self, dashboard_dir):
        from services.data_loader import load_backtest
        df = load_backtest("game_k_backtest")
        assert not df.empty
        assert "n_games" in df.columns

    def test_load_backtest_missing(self, dashboard_dir):
        from services.data_loader import load_backtest
        df = load_backtest("nonexistent_backtest")
        assert df.empty

    def test_load_weekly_snapshots(self, dashboard_dir):
        from services.data_loader import load_weekly_snapshots
        snaps = load_weekly_snapshots("pitcher")
        assert isinstance(snaps, dict)
        assert len(snaps) == 1
        assert "2026-03-01" in snaps
        assert not snaps["2026-03-01"].empty

    def test_load_weekly_snapshots_empty(self, dashboard_dir):
        from services.data_loader import load_weekly_snapshots
        snaps = load_weekly_snapshots("nonexistent")
        assert snaps == {}

class TestBacktestCharts:
    """Verify chart functions return plotly Figure objects."""

    def test_create_coverage_chart(self, dashboard_dir):
        import plotly.graph_objects as go
        from services.data_loader import load_backtest
        from components.backtest_charts import create_coverage_chart

        df = load_backtest("pitcher_k_backtest")
        fig = create_coverage_chart(df, ["coverage_95"], ["95% CI"])
        assert isinstance(fig, go.Figure)

    def test_create_game_k_model_comparison(self, dashboard_dir):
        import plotly.graph_objects as go
        from services.data_loader import load_backtest
        from components.backtest_charts import create_game_k_model_comparison

        df = load_backtest("game_k_backtest")
        fig = create_game_k_model_comparison(df)
        assert isinstance(fig, go.Figure)

    def test_create_movers_chart(self):
        import plotly.graph_objects as go
        from components.backtest_charts import create_movers_chart

        fig = create_movers_chart(
            ["Player A", "Player B", "Player C"],
            [2.5, 1.8, -0.3],
            "Test Movers",
        )
        assert isinstance(fig, go.Figure)


# =====================================================================
# 7. Config schedule refresh tests
# =====================================================================
class TestScheduleConfig:
    """Verify schedule refresh config values."""

    def test_schedule_refresh_minutes(self):
        from config import SCHEDULE_REFRESH_MINUTES
        assert isinstance(SCHEDULE_REFRESH_MINUTES, int)
        assert SCHEDULE_REFRESH_MINUTES > 0

    def test_game_window_hours(self):
        from config import GAME_WINDOW_START_HOUR, GAME_WINDOW_END_HOUR
        assert 0 <= GAME_WINDOW_START_HOUR < 24
        assert 0 < GAME_WINDOW_END_HOUR <= 24
        assert GAME_WINDOW_START_HOUR < GAME_WINDOW_END_HOUR


# =====================================================================
# 8. MiLB data loader tests
# =====================================================================
class TestMilbLoaders:
    """Verify MiLB data loaders."""

    def test_load_milb_factors_missing(self, dashboard_dir):
        import pandas as _pd
        from services.data_loader import load_milb_factors
        df = load_milb_factors("batter")
        assert isinstance(df, _pd.DataFrame)


# =====================================================================
# 9. Rest adjustment tests
# =====================================================================
class TestRestAdjustment:
    """Verify rest adjustment module."""

    def test_classify_rest_bucket(self):
        from lib.rest_adjustment import classify_rest_bucket
        assert classify_rest_bucket(3) == "short"
        assert classify_rest_bucket(5) == "normal"
        assert classify_rest_bucket(7) == "extended"
        assert classify_rest_bucket(None) == "normal"

    def test_apply_rest_to_bf(self):
        from lib.rest_adjustment import apply_rest_to_bf
        mu, sigma = apply_rest_to_bf(23.0, 3.5, 3)  # short rest
        assert mu < 23.0  # should reduce BF
        assert sigma > 3.5  # should increase variance

        mu_n, sigma_n = apply_rest_to_bf(23.0, 3.5, 5)  # normal rest
        assert mu_n == 23.0
        assert sigma_n == 3.5


# =====================================================================
# 10. Game stat simulator tests
# =====================================================================
class TestSimulateGameStat:
    """Verify simulate_game_stat returns correct shapes and dtypes."""

    def test_k_simulation(self):
        import numpy as np
        from lib.game_k_model import simulate_game_stat
        rng = np.random.default_rng(0)
        k_samples = rng.beta(5, 17, size=500)  # ~22% K rate
        result = simulate_game_stat(
            rate_samples=k_samples,
            opp_mu=24.0,
            opp_sigma=3.0,
            n_draws=200,
            random_seed=42,
        )
        assert result.shape == (200,)
        assert np.issubdtype(result.dtype, np.integer)

    def test_k_bb_hr_ordering(self):
        import numpy as np
        from lib.game_k_model import simulate_game_stat
        rng = np.random.default_rng(0)
        k_samples = rng.beta(5, 17, size=500)
        bb_samples = rng.beta(2, 22, size=500)
        hr_samples = rng.beta(1, 32, size=500)
        kwargs = dict(opp_mu=24.0, opp_sigma=3.0, n_draws=200, random_seed=42)
        k_result = simulate_game_stat(rate_samples=k_samples, **kwargs)
        bb_result = simulate_game_stat(rate_samples=bb_samples, **kwargs)
        hr_result = simulate_game_stat(rate_samples=hr_samples, **kwargs)
        for r in (k_result, bb_result, hr_result):
            assert r.shape == (200,)
            assert np.issubdtype(r.dtype, np.integer)
        # Sanity: K mean > BB mean > HR mean
        assert np.mean(k_result) > np.mean(bb_result)
        assert np.mean(bb_result) > np.mean(hr_result)
