#!/usr/bin/env python
"""
Dashboard post-update script.

Runs the projection engine's in-season update (conjugate updating,
schedule fetch, game simulations), then handles dashboard-specific
bookkeeping: weekly snapshots, update metadata, and artifact manifest.

All model work (DB queries, conjugate updating, K samples, matchup
simulation) lives in the player_profiles repo. This script delegates
to it via subprocess.

Usage
-----
    python scripts/update_in_season.py                    # today's date (full update)
    python scripts/update_in_season.py --date 2026-04-15  # specific date
    python scripts/update_in_season.py --skip-schedule    # skip API calls
    python scripts/update_in_season.py --skip-engine      # skip projection engine, just do bookkeeping
    python scripts/update_in_season.py --snapshot          # force a weekly snapshot
    python scripts/update_in_season.py --schedule-only    # refresh schedule/lineups/sims only (hourly mode)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CURRENT_SEASON  # noqa: E402

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
PLAYER_PROFILES_DIR = PROJECT_ROOT.parent / "player_profiles"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEASON = CURRENT_SEASON


# ---------------------------------------------------------------------------
# Weather / umpire parsing helpers
# ---------------------------------------------------------------------------

def _parse_temp_bucket(temp_str: object) -> str:
    """Convert temperature string to weather bucket."""
    if not temp_str:
        return "warm"
    try:
        temp = int(temp_str)
    except (ValueError, TypeError):
        return "warm"
    if temp < 55:
        return "cold"
    if temp < 70:
        return "cool"
    if temp < 85:
        return "warm"
    return "hot"


def _parse_wind_category(wind_str: object) -> str:
    """Convert wind string to category."""
    if not wind_str:
        return "none"
    w = str(wind_str).lower()
    if "calm" in w or w.strip() == "":
        return "none"
    if "out" in w:
        return "out"
    if "in from" in w or "in," in w:
        return "in"
    if "l to r" in w or "r to l" in w:
        return "cross"
    return "none"


# ---------------------------------------------------------------------------
# Roster export
# ---------------------------------------------------------------------------

_ORG_TO_ABBR = {
    108: "LAA", 109: "AZ", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def export_roster() -> bool:
    """Export production.dim_roster to a dashboard parquet.

    Returns True on success, False on failure.
    """
    try:
        from lib.db import read_sql
        import pandas as pd

        df = read_sql("""
            SELECT player_id, player_name, org_id, roster_status,
                   primary_position, is_starter
            FROM production.dim_roster
            WHERE level = 'MLB'
              AND roster_status NOT IN ('released', 'restricted', 'minors')
        """)
        df["team_abbr"] = df["org_id"].map(_ORG_TO_ABBR)
        df.to_parquet(DASHBOARD_DIR / "roster.parquet", index=False)
        logger.info("Exported roster: %d players", len(df))
        return True
    except Exception as e:
        logger.warning("Roster export failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Schedule-only refresh (hourly mode)
# ---------------------------------------------------------------------------

def run_schedule_refresh(game_date: str) -> None:
    """Fetch schedule/lineups and re-run sims using existing projections.

    This is the lightweight hourly mode: no DB queries, no conjugate
    updates — just MLB API calls and Monte Carlo sims with whatever
    projections and K samples are already on disk.

    Uses the PA-by-PA game simulator (Layer 3 v2) for multi-stat
    projections: K, BB, H, HR, IP, pitches, fantasy points.
    """
    import numpy as np
    import pandas as pd
    from lib.schedule import fetch_todays_schedule, fetch_all_lineups
    from lib.game_sim.simulator import simulate_game
    from lib.game_sim.exit_model import ExitModel
    from lib.game_sim.tto_model import build_all_tto_lifts
    from lib.game_sim.pitch_count_model import build_pitch_count_features
    from lib.game_sim.fantasy_scoring import compute_pitcher_fantasy
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    # Fetch schedule
    logger.info("Fetching schedule for %s...", game_date)
    schedule = fetch_todays_schedule(game_date)
    if schedule.empty:
        logger.info("No games scheduled for %s", game_date)
        return

    schedule.to_parquet(DASHBOARD_DIR / "todays_games.parquet", index=False)
    logger.info("Saved schedule: %d games", len(schedule))

    # Fetch lineups
    lineups = fetch_all_lineups(schedule)
    if not lineups.empty:
        lineups.to_parquet(DASHBOARD_DIR / "todays_lineups.parquet", index=False)
        logger.info("Saved lineups: %d batters across %d games",
                     len(lineups), lineups["game_pk"].nunique())
    else:
        logger.info("No lineups available yet")

    # --- Load existing projections and posterior samples ---
    p_path = DASHBOARD_DIR / "pitcher_projections.parquet"
    k_path = DASHBOARD_DIR / "pitcher_k_samples.npz"

    if not p_path.exists() or not k_path.exists():
        logger.warning("Missing projections or K samples — cannot simulate. "
                        "Run a full update first.")
        return

    pitcher_proj = pd.read_parquet(p_path)

    k_data = np.load(k_path)
    k_samples = {k: k_data[k] for k in k_data.files}

    bb_path = DASHBOARD_DIR / "pitcher_bb_samples.npz"
    bb_samples: dict[str, np.ndarray] = {}
    if bb_path.exists():
        _bb = np.load(bb_path)
        bb_samples = {k: _bb[k] for k in _bb.files}

    hr_path = DASHBOARD_DIR / "pitcher_hr_samples.npz"
    hr_samples: dict[str, np.ndarray] = {}
    if hr_path.exists():
        _hr = np.load(hr_path)
        hr_samples = {k: _hr[k] for k in _hr.files}

    arsenal_path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    vuln_path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
    arsenal_df = pd.read_parquet(arsenal_path) if arsenal_path.exists() else pd.DataFrame()
    vuln_df = pd.read_parquet(vuln_path) if vuln_path.exists() else pd.DataFrame()

    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    # --- Load game simulator component data ---
    exit_model = ExitModel()
    exit_pkl = DASHBOARD_DIR / "exit_model.pkl"
    if exit_pkl.exists():
        exit_model.load(exit_pkl)
        logger.info("Loaded exit model")
    else:
        logger.warning("No exit model — using fallback sigmoid")

    pitcher_pc_path = DASHBOARD_DIR / "pitcher_pitch_count_features.parquet"
    batter_pc_path = DASHBOARD_DIR / "batter_pitch_count_features.parquet"
    pitcher_pc = pd.read_parquet(pitcher_pc_path) if pitcher_pc_path.exists() else pd.DataFrame()
    batter_pc = pd.read_parquet(batter_pc_path) if batter_pc_path.exists() else pd.DataFrame()

    tto_path = DASHBOARD_DIR / "tto_profiles.parquet"
    tto_profiles = pd.read_parquet(tto_path) if tto_path.exists() else pd.DataFrame()

    tend_path = DASHBOARD_DIR / "pitcher_exit_tendencies.parquet"
    exit_tendencies = pd.read_parquet(tend_path) if tend_path.exists() else pd.DataFrame()

    # --- Load umpire tendencies and weather effects ---
    ump_path = DASHBOARD_DIR / "umpire_tendencies.parquet"
    wx_path = DASHBOARD_DIR / "weather_effects.parquet"
    ump_lookup: dict[str, float] = {}
    if ump_path.exists():
        ump_df = pd.read_parquet(ump_path)
        for _, ur in ump_df.iterrows():
            ump_lookup[ur["hp_umpire_name"]] = float(ur["k_logit_lift"])
        logger.info("Loaded %d umpire tendencies", len(ump_lookup))

    wx_lookup: dict[tuple[str, str], dict] = {}
    if wx_path.exists():
        wx_df = pd.read_parquet(wx_path)
        for _, wr in wx_df.iterrows():
            wx_lookup[(wr["temp_bucket"], wr["wind_category"])] = {
                "k_multiplier": float(wr["k_multiplier"]),
                "overall_k_rate": float(wr["overall_k_rate"]),
            }
        logger.info("Loaded %d weather effect combos", len(wx_lookup))

    # --- Helper: get pitcher avg exit pitches ---
    def _pitcher_avg_pitches(pid: int) -> float:
        if exit_tendencies.empty:
            return 88.0
        row = exit_tendencies[
            (exit_tendencies["pitcher_id"] == pid)
            & (exit_tendencies["season"] == SEASON - 1)
        ]
        if not row.empty:
            return float(row.iloc[0]["avg_pitches"])
        return 88.0

    # --- Helper: generate fallback rate samples ---
    _rng_fallback = np.random.default_rng(99)

    def _fallback_samples(rate: float, n: int = 4000) -> np.ndarray:
        """Beta(a, b) centered on rate with moderate spread."""
        r = np.clip(rate, 0.01, 0.99)
        kappa = 200  # concentration
        return _rng_fallback.beta(r * kappa, (1 - r) * kappa, size=n).astype(np.float32)

    # --- Simulate each starter ---
    logger.info("Simulating game props for today's starters...")
    results = []
    for _, game in schedule.iterrows():
        gpk = game["game_pk"]

        # Per-game umpire lift
        hp_ump_name = game.get("hp_umpire_name", "")
        ump_k_lift = ump_lookup.get(hp_ump_name, 0.0) if hp_ump_name else 0.0

        # Per-game weather lift
        wx_k_lift = 0.0
        temp_bucket = _parse_temp_bucket(game.get("weather_temp"))
        wind_cat = _parse_wind_category(game.get("weather_wind"))
        wx_info = wx_lookup.get((temp_bucket, wind_cat))
        if wx_info:
            from scipy.special import logit as _logit
            k_mult = wx_info["k_multiplier"]
            overall_k = wx_info["overall_k_rate"]
            adj_k = np.clip(overall_k * k_mult, 1e-6, 1 - 1e-6)
            wx_k_lift = float(_logit(adj_k) - _logit(np.clip(overall_k, 1e-6, 1 - 1e-6)))

        for side in ("away", "home"):
            pid = game.get(f"{side}_pitcher_id")
            pname = game.get(f"{side}_pitcher_name", "")

            if pd.isna(pid):
                continue
            pid = int(pid)
            pid_str = str(pid)

            if pid_str not in k_samples:
                logger.debug("No K samples for pitcher %s (%s) — skipping", pname, pid)
                continue

            k_samp = k_samples[pid_str]

            # BB / HR samples (fallback to projection-based Beta if missing)
            p_row = pitcher_proj[pitcher_proj["pitcher_id"] == pid]
            proj_k_rate = float(p_row.iloc[0]["projected_k_rate"]) if not p_row.empty else float(np.mean(k_samp))
            proj_bb = float(p_row.iloc[0].get("projected_bb_rate", 0.08)) if not p_row.empty else 0.08
            proj_hr = float(p_row.iloc[0].get("projected_hr_per_bf", 0.03)) if not p_row.empty else 0.03
            composite = float(p_row.iloc[0].get("composite_score", 0)) if not p_row.empty else 0.0

            bb_samp = bb_samples.get(pid_str) if bb_samples else None
            if bb_samp is None:
                bb_samp = _fallback_samples(proj_bb)
            hr_samp = hr_samples.get(pid_str) if hr_samples else None
            if hr_samp is None:
                hr_samp = _fallback_samples(proj_hr)

            # Lineup matchup lifts
            opp_side = "home" if side == "away" else "away"
            opp_team_id = game.get(f"{opp_side}_team_id")
            lineup_matchup_lifts: dict[str, np.ndarray] = {}
            has_lineup = False
            lineup_batter_ids: list[int] = []

            if not lineups.empty and not arsenal_df.empty and not vuln_df.empty:
                game_lu = lineups[
                    (lineups["game_pk"] == gpk) &
                    (lineups["team_id"] == opp_team_id)
                ].sort_values("batting_order")

                if len(game_lu) >= 9:
                    has_lineup = True
                    k_lifts, bb_lifts, hr_lifts = [], [], []
                    for _, brow in game_lu.head(9).iterrows():
                        bid = int(brow["batter_id"])
                        lineup_batter_ids.append(bid)

                        k_m = score_matchup(pid, bid, arsenal_df, vuln_df, baselines_pt)
                        kl = k_m.get("matchup_k_logit_lift", 0.0)
                        k_lifts.append(0.0 if np.isnan(kl) else kl)

                        bb_m = score_matchup_bb(pid, bid, arsenal_df, vuln_df, baselines_pt)
                        bl = bb_m.get("matchup_bb_logit_lift", 0.0)
                        bb_lifts.append(0.0 if np.isnan(bl) else bl)

                        hr_m = score_matchup_hr(pid, bid, arsenal_df, vuln_df, baselines_pt)
                        hl = hr_m.get("matchup_hr_logit_lift", 0.0)
                        hr_lifts.append(0.0 if np.isnan(hl) else hl)

                    lineup_matchup_lifts = {
                        "k": np.array(k_lifts),
                        "bb": np.array(bb_lifts),
                        "hr": np.array(hr_lifts),
                    }

            # TTO lifts
            tto_lifts = build_all_tto_lifts(
                tto_profiles if not tto_profiles.empty else None,
                pid, SEASON - 1,
            )

            # Pitch count features
            pitcher_ppa_adj = 0.0
            batter_ppa_adjs = np.zeros(9)
            if not pitcher_pc.empty and not batter_pc.empty and lineup_batter_ids:
                pitcher_ppa_adj, batter_ppa_adjs = build_pitch_count_features(
                    pitcher_pc, batter_pc, pid, lineup_batter_ids, SEASON - 1,
                )

            avg_pitches = _pitcher_avg_pitches(pid)

            # Run PA-by-PA simulation
            result = simulate_game(
                pitcher_k_rate_samples=k_samp,
                pitcher_bb_rate_samples=bb_samp,
                pitcher_hr_rate_samples=hr_samp,
                lineup_matchup_lifts=lineup_matchup_lifts,
                tto_lifts=tto_lifts,
                pitcher_ppa_adj=pitcher_ppa_adj,
                batter_ppa_adjs=batter_ppa_adjs,
                exit_model=exit_model,
                pitcher_avg_pitches=avg_pitches,
                umpire_k_lift=ump_k_lift,
                weather_k_lift=wx_k_lift,
                n_sims=10_000,
                random_seed=42 + gpk + (0 if side == "away" else 1),
            )

            # Fantasy points
            fantasy = compute_pitcher_fantasy(result)
            dk = fantasy.dk_summary()
            espn = fantasy.espn_summary()

            # K prop lines
            k_over = result.over_probs("k", lines=[4.5, 5.5, 6.5, 7.5])
            p_over_dict = {}
            for _, kr in k_over.iterrows():
                col = f"p_over_{kr['line']:.1f}".replace(".", "_")
                p_over_dict[col] = kr["p_over"]

            # BB and HR prop lines
            bb_over = result.over_probs("bb", lines=[1.5, 2.5, 3.5])
            for _, br in bb_over.iterrows():
                col = f"p_over_bb_{br['line']:.1f}".replace(".", "_")
                p_over_dict[col] = br["p_over"]

            hr_over = result.over_probs("hr", lines=[0.5, 1.5])
            for _, hr_r in hr_over.iterrows():
                col = f"p_over_hr_{hr_r['line']:.1f}".replace(".", "_")
                p_over_dict[col] = hr_r["p_over"]

            # H prop lines
            h_over = result.over_probs("h", lines=[4.5, 5.5, 6.5, 7.5])
            for _, hr_row in h_over.iterrows():
                col = f"p_over_h_{hr_row['line']:.1f}".replace(".", "_")
                p_over_dict[col] = hr_row["p_over"]

            avg_matchup = float(np.mean(lineup_matchup_lifts["k"])) if has_lineup else 0.0

            results.append({
                "game_pk": gpk,
                "side": side,
                "pitcher_id": pid,
                "pitcher_name": pname,
                "team_abbr": game.get(f"{side}_abbr", ""),
                "opp_abbr": game.get(f"{opp_side}_abbr", ""),
                "projected_k_rate": proj_k_rate,
                "composite_score": composite,
                # K
                "expected_k": float(np.mean(result.k_samples)),
                "k_std": float(np.std(result.k_samples)),
                "median_k": float(np.median(result.k_samples)),
                # BB
                "expected_bb": float(np.mean(result.bb_samples)),
                "bb_std": float(np.std(result.bb_samples)),
                # H
                "expected_h": float(np.mean(result.h_samples)),
                "h_std": float(np.std(result.h_samples)),
                # HR
                "expected_hr": float(np.mean(result.hr_samples)),
                "hr_std": float(np.std(result.hr_samples)),
                # IP / Pitches
                "expected_ip": float(np.mean(result.ip_samples())),
                "expected_pitches": float(np.mean(result.pitch_count_samples)),
                "expected_bf": float(np.mean(result.bf_samples)),
                # Fantasy
                "dk_mean": dk["mean"],
                "dk_median": dk["median"],
                "dk_q10": dk["q10"],
                "dk_q90": dk["q90"],
                "espn_mean": espn["mean"],
                "espn_median": espn["median"],
                # Context
                "has_lineup": has_lineup,
                "avg_matchup_lift": avg_matchup,
                "umpire_k_logit_lift": ump_k_lift,
                "weather_k_logit_lift": wx_k_lift,
                "bf_mu": float(np.mean(result.bf_samples)),
                "bf_sigma": float(np.std(result.bf_samples)),
                **p_over_dict,
            })

    if results:
        sim_df = pd.DataFrame(results)
        sim_df.to_parquet(DASHBOARD_DIR / "todays_sims.parquet", index=False)
        logger.info("Saved game simulations for %d pitcher appearances", len(sim_df))

        n_with_lineup = sim_df["has_lineup"].sum()
        logger.info("  %d with lineup data, %d without",
                     n_with_lineup, len(sim_df) - n_with_lineup)
    else:
        logger.warning("No pitchers could be simulated (missing K samples?)")


# ---------------------------------------------------------------------------
# Projection engine delegation
# ---------------------------------------------------------------------------

def run_projection_engine(game_date: str, skip_schedule: bool = False) -> bool:
    """Run the player_profiles in-season update script.

    Returns True on success, False on failure.
    """
    engine_script = PLAYER_PROFILES_DIR / "scripts" / "update_in_season.py"
    if not engine_script.exists():
        logger.error(
            "Projection engine script not found at %s. "
            "Ensure player_profiles repo is at %s",
            engine_script, PLAYER_PROFILES_DIR,
        )
        return False

    # Use player_profiles' own virtualenv, not the dashboard's
    engine_python = PLAYER_PROFILES_DIR / "myenv" / "Scripts" / "python.exe"
    if not engine_python.exists():
        # Fall back to Unix layout
        engine_python = PLAYER_PROFILES_DIR / "myenv" / "bin" / "python"
    if not engine_python.exists():
        logger.error("player_profiles virtualenv not found at %s", engine_python.parent)
        return False

    cmd = [str(engine_python), str(engine_script), "--date", game_date]
    if skip_schedule:
        cmd.append("--skip-schedule")

    logger.info("Running projection engine: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PLAYER_PROFILES_DIR))

    if result.returncode != 0:
        logger.error("Projection engine exited with code %d", result.returncode)
        return False

    logger.info("Projection engine completed successfully")
    return True


# ---------------------------------------------------------------------------
# Weekly snapshots
# ---------------------------------------------------------------------------

def _is_snapshot_day(game_date: str) -> bool:
    """Return True if game_date falls on a Monday (weekly snapshot day)."""
    d = date.fromisoformat(game_date)
    return d.weekday() == 0  # Monday


def save_weekly_snapshot(game_date: str) -> None:
    """Copy current projections to a dated weekly snapshot.

    Idempotent: skips if a snapshot already exists for this date.
    """
    weekly_dir = DASHBOARD_DIR / "snapshots" / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    for ptype in ("hitter", "pitcher"):
        src = DASHBOARD_DIR / f"{ptype}_projections.parquet"
        dst = weekly_dir / f"{ptype}_projections_{game_date}.parquet"
        if dst.exists():
            logger.info("Weekly snapshot already exists: %s — skipping", dst.name)
            continue
        if not src.exists():
            logger.warning("No %s projections to snapshot", ptype)
            continue
        shutil.copy2(src, dst)
        logger.info("Saved weekly snapshot: %s", dst.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard in-season update")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--skip-schedule", action="store_true",
                        help="Skip fetching schedule/lineups from MLB API.")
    parser.add_argument("--skip-engine", action="store_true",
                        help="Skip projection engine; only run dashboard bookkeeping.")
    parser.add_argument("--snapshot", action="store_true",
                        help="Force saving a weekly projection snapshot.")
    parser.add_argument("--schedule-only", action="store_true",
                        help="Refresh schedule/lineups/sims only (no projection updates).")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info("=" * 60)
    logger.info("Dashboard update for %s (season %d)", game_date, SEASON)
    logger.info("=" * 60)

    # Schedule-only mode: lightweight refresh, then exit
    if args.schedule_only:
        logger.info("Mode: schedule-only (hourly refresh)")
        run_schedule_refresh(game_date)

        # Update metadata timestamp
        meta_path = DASHBOARD_DIR / "update_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata["last_schedule_refresh"] = datetime.now().isoformat()
        metadata["game_date"] = game_date
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("Done! (schedule-only)")
        return

    # Step 1: Run projection engine (model work lives in player_profiles)
    if not args.skip_engine:
        success = run_projection_engine(game_date, skip_schedule=args.skip_schedule)
        if not success:
            logger.warning(
                "Projection engine failed or not found. "
                "Continuing with dashboard bookkeeping using existing parquets."
            )
    else:
        logger.info("Step 1: Skipped (--skip-engine)")

    # Step 2: Export roster from DB to parquet
    logger.info("Step 2: Exporting roster...")
    export_roster()

    # Step 3: Weekly snapshot
    if args.snapshot or _is_snapshot_day(game_date):
        logger.info("Saving weekly projection snapshot...")
        save_weekly_snapshot(game_date)

    # Step 4: Save update metadata
    import pandas as pd
    h_path = DASHBOARD_DIR / "hitter_projections.parquet"
    p_path = DASHBOARD_DIR / "pitcher_projections.parquet"
    k_path = DASHBOARD_DIR / "pitcher_k_samples.npz"

    metadata = {
        "last_updated": datetime.now().isoformat(),
        "game_date": game_date,
        "season": SEASON,
        "hitters_updated": len(pd.read_parquet(h_path)) if h_path.exists() else 0,
        "pitchers_updated": len(pd.read_parquet(p_path)) if p_path.exists() else 0,
        "k_samples_count": len(dict(__import__("numpy").load(k_path))) if k_path.exists() else 0,
    }
    meta_path = DASHBOARD_DIR / "update_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved update metadata to %s", meta_path)

    # Step 5: Generate artifact manifest
    from services.manifest import generate_manifest
    manifest = generate_manifest(DASHBOARD_DIR)
    manifest_path = DASHBOARD_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest with %d artifacts to %s",
                len(manifest.get("artifacts", [])), manifest_path)

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
