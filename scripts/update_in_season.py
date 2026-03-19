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
# Schedule-only refresh (hourly mode)
# ---------------------------------------------------------------------------

def run_schedule_refresh(game_date: str) -> None:
    """Fetch schedule/lineups and re-run sims using existing projections.

    This is the lightweight hourly mode: no DB queries, no conjugate
    updates — just MLB API calls and Monte Carlo sims with whatever
    projections and K samples are already on disk.
    """
    import numpy as np
    import pandas as pd
    from lib.schedule import fetch_todays_schedule, fetch_all_lineups
    from lib.game_k_model import simulate_game_ks, compute_k_over_probs
    from lib.bf_model import get_bf_distribution
    from lib.matchup import score_matchup
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

    # Load existing projections and K samples from disk
    p_path = DASHBOARD_DIR / "pitcher_projections.parquet"
    k_path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    bf_path = DASHBOARD_DIR / "bf_priors.parquet"
    arsenal_path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    vuln_path = DASHBOARD_DIR / "hitter_vuln_career.parquet"

    if not p_path.exists() or not k_path.exists():
        logger.warning("Missing projections or K samples — cannot simulate. "
                        "Run a full update first.")
        return

    pitcher_proj = pd.read_parquet(p_path)
    k_data = np.load(k_path)
    k_samples = {k: k_data[k] for k in k_data.files}
    bf_priors = pd.read_parquet(bf_path) if bf_path.exists() else pd.DataFrame()
    arsenal_df = pd.read_parquet(arsenal_path) if arsenal_path.exists() else pd.DataFrame()
    vuln_df = pd.read_parquet(vuln_path) if vuln_path.exists() else pd.DataFrame()

    baselines_pt = {
        pt: {"whiff_rate": vals.get("whiff_rate", 0.25)}
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    # Load umpire tendencies and weather effects for adjustments
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

    # Simulate each starter
    logger.info("Simulating K props for today's starters...")
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

            samples = k_samples[pid_str]

            # BF distribution
            bf_info = get_bf_distribution(pid, SEASON - 1, bf_priors)
            bf_mu = bf_info["mu_bf"]
            bf_sigma = bf_info["sigma_bf"]

            # Lineup matchup lifts
            lineup_lifts = None
            opp_side = "home" if side == "away" else "away"
            opp_team_id = game.get(f"{opp_side}_team_id")

            if not lineups.empty and not arsenal_df.empty and not vuln_df.empty:
                game_lu = lineups[
                    (lineups["game_pk"] == gpk) &
                    (lineups["team_id"] == opp_team_id)
                ].sort_values("batting_order")

                if len(game_lu) >= 9:
                    lifts = []
                    for _, brow in game_lu.head(9).iterrows():
                        bid = int(brow["batter_id"])
                        m = score_matchup(
                            pid, bid, arsenal_df, vuln_df, baselines_pt,
                        )
                        lift = m.get("matchup_k_logit_lift", 0.0)
                        lifts.append(0.0 if np.isnan(lift) else lift)
                    lineup_lifts = np.array(lifts)

            # Simulate
            game_ks = simulate_game_ks(
                pitcher_k_rate_samples=samples,
                bf_mu=float(bf_mu),
                bf_sigma=float(bf_sigma),
                lineup_matchup_lifts=lineup_lifts,
                umpire_k_logit_lift=ump_k_lift,
                weather_k_logit_lift=wx_k_lift,
                n_draws=10000,
                random_seed=42 + gpk,
            )

            k_over = compute_k_over_probs(game_ks)

            p_over_dict = {}
            for _, kr in k_over.iterrows():
                line = kr["line"]
                if line in (4.5, 5.5, 6.5, 7.5):
                    p_over_dict[f"p_over_{line:.1f}".replace(".", "_")] = kr["p_over"]

            p_row = pitcher_proj[pitcher_proj["pitcher_id"] == pid]
            proj_k_rate = float(p_row.iloc[0]["projected_k_rate"]) if not p_row.empty else np.mean(samples)
            composite = float(p_row.iloc[0].get("composite_score", 0)) if not p_row.empty else 0.0

            results.append({
                "game_pk": gpk,
                "side": side,
                "pitcher_id": pid,
                "pitcher_name": pname,
                "team_abbr": game.get(f"{side}_abbr", ""),
                "opp_abbr": game.get(f"{opp_side}_abbr", ""),
                "projected_k_rate": proj_k_rate,
                "composite_score": composite,
                "expected_k": float(np.mean(game_ks)),
                "k_std": float(np.std(game_ks)),
                "median_k": float(np.median(game_ks)),
                "has_lineup": lineup_lifts is not None,
                "avg_matchup_lift": float(np.mean(lineup_lifts)) if lineup_lifts is not None else 0.0,
                "umpire_k_logit_lift": ump_k_lift,
                "weather_k_logit_lift": wx_k_lift,
                "bf_mu": float(bf_mu),
                "bf_sigma": float(bf_sigma),
                **p_over_dict,
            })

    if results:
        sim_df = pd.DataFrame(results)
        sim_df.to_parquet(DASHBOARD_DIR / "todays_sims.parquet", index=False)
        logger.info("Saved K simulations for %d pitcher appearances", len(sim_df))

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

    cmd = [sys.executable, str(engine_script), "--date", game_date]
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

    # Step 2: Weekly snapshot
    if args.snapshot or _is_snapshot_day(game_date):
        logger.info("Saving weekly projection snapshot...")
        save_weekly_snapshot(game_date)

    # Step 3: Save update metadata
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

    # Step 4: Generate artifact manifest
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
