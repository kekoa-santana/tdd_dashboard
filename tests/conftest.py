"""Shared fixtures for TDD Dashboard smoke tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path so imports work like they do at runtime
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock st.cache_data before any dashboard module is imported.
# This avoids Streamlit's runtime dependency on an active event loop.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()


def _passthrough_decorator(*args, **kwargs):
    """No-op replacement for st.cache_data / st.cache_resource."""
    if args and callable(args[0]):
        return args[0]
    return lambda fn: fn


_st_mock.cache_data = _passthrough_decorator
_st_mock.cache_resource = _passthrough_decorator
# Let attribute access for st.selectbox, st.radio, etc. return a MagicMock
# so that module-level code that references st.X doesn't crash on import.
sys.modules.setdefault("streamlit", _st_mock)


# ---------------------------------------------------------------------------
# Fixture: temporary dashboard data directory with minimal parquet files
# ---------------------------------------------------------------------------
@pytest.fixture()
def dashboard_dir(tmp_path, monkeypatch):
    """Create a temp data/dashboard/ directory with minimal fixture files
    and patch config.DASHBOARD_DIR to point to it."""
    d = tmp_path / "data" / "dashboard"
    d.mkdir(parents=True)

    # -- hitter_projections.parquet --
    hitter_proj = pd.DataFrame({
        "batter_id": [660271, 665742],
        "batter_name": ["Shohei Ohtani", "Juan Soto"],
        "batter_stand": ["L", "L"],
        "age": [31, 27],
        "composite_score": [92.3, 89.1],
        "projected_k_rate": [0.220, 0.185],
        "projected_bb_rate": [0.095, 0.155],
        "observed_k_rate": [0.230, 0.190],
        "observed_bb_rate": [0.090, 0.150],
        "delta_k_rate": [-0.010, -0.005],
        "delta_bb_rate": [0.005, 0.005],
        "projected_k_rate_2_5": [0.180, 0.155],
        "projected_k_rate_97_5": [0.260, 0.215],
        "projected_bb_rate_2_5": [0.065, 0.125],
        "projected_bb_rate_97_5": [0.125, 0.185],
        "whiff_rate": [0.28, 0.22],
        "chase_rate": [0.30, 0.24],
        "z_contact_pct": [0.85, 0.90],
        "avg_exit_velo": [93.5, 91.2],
        "hard_hit_pct": [0.48, 0.42],
        "sprint_speed": [27.5, 26.8],
        "fb_pct": [0.38, 0.35],
    })
    hitter_proj.to_parquet(d / "hitter_projections.parquet", index=False)

    # -- pitcher_projections.parquet --
    pitcher_proj = pd.DataFrame({
        "pitcher_id": [543037, 477132],
        "pitcher_name": ["Gerrit Cole", "Clayton Kershaw"],
        "pitch_hand": ["R", "L"],
        "age": [35, 38],
        "composite_score": [91.0, 82.5],
        "projected_k_rate": [0.295, 0.260],
        "projected_bb_rate": [0.055, 0.065],
        "observed_k_rate": [0.300, 0.255],
        "observed_bb_rate": [0.050, 0.070],
        "delta_k_rate": [-0.005, 0.005],
        "delta_bb_rate": [0.005, -0.005],
        "projected_k_rate_2_5": [0.260, 0.225],
        "projected_k_rate_97_5": [0.330, 0.295],
        "projected_bb_rate_2_5": [0.035, 0.045],
        "projected_bb_rate_97_5": [0.075, 0.085],
        "is_starter": [True, True],
        "whiff_rate": [0.32, 0.28],
        "avg_velo": [97.2, 91.5],
        "release_extension": [6.5, 6.2],
        "zone_pct": [0.48, 0.51],
        "gb_pct": [0.38, 0.42],
    })
    pitcher_proj.to_parquet(d / "pitcher_projections.parquet", index=False)

    # -- player_teams.parquet --
    player_teams = pd.DataFrame({
        "player_id": [660271, 665742, 543037, 477132],
        "team_abbr": ["LAD", "NYY", "NYY", "LAD"],
    })
    player_teams.to_parquet(d / "player_teams.parquet", index=False)

    # -- update_metadata.json --
    metadata = {
        "last_updated": "2026-03-11T08:00:00",
        "game_date": "2026-03-11",
        "season": 2026,
        "hitters_updated": 200,
        "pitchers_updated": 150,
    }
    with open(d / "update_metadata.json", "w") as f:
        json.dump(metadata, f)

    # -- backtest fixture: pitcher_k_backtest.parquet --
    bt_pitcher_k = pd.DataFrame({
        "test_season": [2023, 2024],
        "bayes_mae": [0.032, 0.035],
        "marcel_mae": [0.033, 0.033],
        "mae_improvement_pct": [3.0, -5.7],
        "bayes_rmse": [0.042, 0.045],
        "marcel_rmse": [0.041, 0.043],
        "rmse_improvement_pct": [-0.8, -5.0],
        "coverage_95": [0.88, 0.81],
        "bayes_brier": [0.17, 0.21],
        "marcel_brier": [0.20, 0.25],
        "n_players": [427, 424],
        "converged": [True, True],
    })
    bt_pitcher_k.to_parquet(d / "backtest_pitcher_k_backtest.parquet", index=False)

    # -- backtest fixture: game_k_backtest.parquet --
    bt_game_k = pd.DataFrame({
        "test_season": [2023, 2024],
        "n_games": [3782, 3789],
        "rmse": [2.31, 2.25],
        "mae": [1.83, 1.80],
        "avg_brier": [0.188, 0.189],
        "coverage_50": [0.48, 0.48],
        "coverage_80": [0.79, 0.79],
        "coverage_90": [0.89, 0.90],
        "log_score": [-2.24, -2.21],
        "naive_rmse": [2.29, 2.26],
        "naive_avg_brier": [0.184, 0.188],
        "poisson_rmse": [2.34, 2.30],
        "poisson_avg_brier": [0.190, 0.193],
        "model_no_matchup_rmse": [2.34, 2.30],
        "model_no_matchup_avg_brier": [0.190, 0.193],
        "full_model_rmse": [2.31, 2.25],
        "full_model_avg_brier": [0.188, 0.189],
    })
    bt_game_k.to_parquet(d / "backtest_game_k_backtest.parquet", index=False)

    # -- snapshots/weekly/ fixture --
    weekly_dir = d / "snapshots" / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    snap_pitcher = pitcher_proj.copy()
    snap_pitcher["projected_k_rate"] = [0.290, 0.255]
    snap_pitcher.to_parquet(
        weekly_dir / "pitcher_projections_2026-03-01.parquet", index=False,
    )

    # Patch DASHBOARD_DIR everywhere it's been imported
    import config
    monkeypatch.setattr(config, "DASHBOARD_DIR", d)

    import services.data_loader
    monkeypatch.setattr(services.data_loader, "DASHBOARD_DIR", d)

    import utils.helpers
    monkeypatch.setattr(utils.helpers, "DASHBOARD_DIR", d)

    return d
