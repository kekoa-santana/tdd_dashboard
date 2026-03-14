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
    python scripts/update_in_season.py                    # today's date
    python scripts/update_in_season.py --date 2026-04-15  # specific date
    python scripts/update_in_season.py --skip-schedule    # skip API calls
    python scripts/update_in_season.py --skip-engine      # skip projection engine, just do bookkeeping
    python scripts/update_in_season.py --snapshot          # force a weekly snapshot
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
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info("=" * 60)
    logger.info("Dashboard update for %s (season %d)", game_date, SEASON)
    logger.info("=" * 60)

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
