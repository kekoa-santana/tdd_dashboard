"""
Sync, check, and verify lib/ modules against the player_profiles source.

Usage:
    python scripts/sync_lib.py              # default: --check
    python scripts/sync_lib.py --check      # dry-run comparison
    python scripts/sync_lib.py --sync       # copy + fix imports
    python scripts/sync_lib.py --sync --force  # overwrite without confirmation
    python scripts/sync_lib.py --verify     # test imports
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
LIB_DIR = DASHBOARD_ROOT / "lib"
PLAYER_PROFILES_ROOT = DASHBOARD_ROOT.parent / "player_profiles"
SRC_DIR = PLAYER_PROFILES_ROOT / "src"

# ---------------------------------------------------------------------------
# File mapping: lib/ filename -> relative path under player_profiles/src/
# ---------------------------------------------------------------------------

FILE_MAP: dict[str, str] = {
    "constants.py":         "utils/constants.py",
    "theme.py":             "viz/theme.py",
    "matchup.py":           "models/matchup.py",
    "bf_model.py":          "models/bf_model.py",
    "game_k_model.py":      "models/game_k_model.py",
    "zone_charts.py":       "viz/zone_charts.py",
    "in_season_updater.py": "models/in_season_updater.py",
    "schedule.py":          "data/schedule.py",
    "db.py":                "data/db.py",
}

# Verify targets: module name -> list of names to import
# Used by --verify to confirm each module loads correctly.
VERIFY_IMPORTS: dict[str, list[str]] = {
    "constants":         ["PITCH_TO_FAMILY", "LEAGUE_AVG_OVERALL"],
    "theme":             ["GOLD", "EMBER", "SAGE", "SLATE", "CREAM", "DARK", "add_watermark"],
    "matchup":           ["score_matchup", "score_matchups_batch"],
    "bf_model":          ["get_bf_distribution", "draw_bf_samples"],
    "game_k_model":      ["simulate_game_ks", "compute_k_over_probs"],
    "zone_charts":       ["plot_pitcher_location_heatmap", "plot_hitter_zone_grid", "plot_matchup_overlay"],
    "in_season_updater": ["conjugate_update", "update_projections", "update_pitcher_k_samples"],
    "schedule":          ["fetch_todays_schedule", "fetch_game_lineups", "fetch_all_lineups"],
    "db":                ["get_engine", "read_sql"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_imports(content: str) -> str:
    """Replace ``from src.`` / ``import src.`` with ``from lib.`` / ``import lib.``."""
    content = re.sub(r'\bfrom\s+src\.', 'from lib.', content)
    content = re.sub(r'\bimport\s+src\.', 'import lib.', content)
    return content


def _content_hash(text: str) -> str:
    """SHA-256 of text after normalising line endings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalised_content(path: Path) -> str | None:
    """Read a file and return content with normalised line endings, or None."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _comparable_content(source_text: str) -> str:
    """Return source content with ``from src.`` already fixed so we can
    compare it fairly against the lib/ copy."""
    return _fix_imports(source_text)

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_check() -> int:
    """Dry-run: report which lib files are up to date, need sync, or missing."""
    print()
    print(f"  Source root : {SRC_DIR}")
    print(f"  Dest root   : {LIB_DIR}")
    print()

    header = f"  {'lib/ file':<25} {'Status':<18} {'Detail'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    needs_sync = 0

    for lib_name, src_rel in FILE_MAP.items():
        src_path = SRC_DIR / src_rel
        dst_path = LIB_DIR / lib_name

        src_text = _normalised_content(src_path)
        dst_text = _normalised_content(dst_path)

        if src_text is None:
            print(f"  {lib_name:<25} {'SOURCE MISSING':<18} {src_path}")
            needs_sync += 1
            continue

        if dst_text is None:
            print(f"  {lib_name:<25} {'DEST MISSING':<18} (not yet synced)")
            needs_sync += 1
            continue

        comparable_src = _comparable_content(src_text)

        if _content_hash(comparable_src) == _content_hash(dst_text):
            print(f"  {lib_name:<25} {'UP TO DATE':<18}")
        else:
            print(f"  {lib_name:<25} {'NEEDS SYNC':<18} content differs")
            needs_sync += 1

    print()
    if needs_sync == 0:
        print("  All files are up to date.")
    else:
        print(f"  {needs_sync} file(s) need attention. Run --sync to update.")
    print()
    return 0 if needs_sync == 0 else 1


def cmd_sync(*, force: bool = False) -> int:
    """Copy source files to lib/, fixing imports."""
    print()
    print(f"  Source root : {SRC_DIR}")
    print(f"  Dest root   : {LIB_DIR}")
    print()

    if not SRC_DIR.exists():
        print(f"  ERROR: source directory does not exist: {SRC_DIR}")
        return 1

    LIB_DIR.mkdir(parents=True, exist_ok=True)

    to_sync: list[tuple[str, Path, Path]] = []

    for lib_name, src_rel in FILE_MAP.items():
        src_path = SRC_DIR / src_rel
        dst_path = LIB_DIR / lib_name

        if not src_path.exists():
            print(f"  SKIP  {lib_name:<25} source not found: {src_path}")
            continue

        src_text = _normalised_content(src_path)
        dst_text = _normalised_content(dst_path)
        assert src_text is not None  # we just checked exists

        comparable_src = _comparable_content(src_text)

        if dst_text is not None and _content_hash(comparable_src) == _content_hash(dst_text):
            print(f"  OK    {lib_name:<25} already up to date")
            continue

        to_sync.append((lib_name, src_path, dst_path))

    if not to_sync:
        print("\n  Nothing to sync — all files up to date.")
        return 0

    print()
    print(f"  {len(to_sync)} file(s) to sync:")
    for lib_name, src_path, _ in to_sync:
        print(f"    {src_path}  ->  lib/{lib_name}")

    if not force:
        try:
            answer = input("\n  Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return 1
        if answer != "y":
            print("  Aborted.")
            return 1

    print()
    errors = 0
    for lib_name, src_path, dst_path in to_sync:
        try:
            src_text = src_path.read_text(encoding="utf-8")
            fixed = _fix_imports(src_text)
            dst_path.write_text(fixed, encoding="utf-8", newline="\n")
            print(f"  SYNCED  {lib_name}")
        except Exception as exc:
            print(f"  ERROR   {lib_name}: {exc}")
            errors += 1

    print()
    if errors:
        print(f"  Done with {errors} error(s).")
    else:
        print("  All files synced successfully.")
        print("  Run --verify to confirm imports work.")
    return 1 if errors else 0


def cmd_verify() -> int:
    """Test that each lib module can be imported."""
    print()
    print(f"  Working dir : {DASHBOARD_ROOT}")
    print()

    header = f"  {'Module':<25} {'Status':<10} {'Detail'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    failures = 0

    for module, names in VERIFY_IMPORTS.items():
        import_stmt = f"from lib.{module} import {', '.join(names)}"
        result = subprocess.run(
            [sys.executable, "-c", import_stmt],
            capture_output=True,
            text=True,
            cwd=str(DASHBOARD_ROOT),
        )
        if result.returncode == 0:
            print(f"  {module:<25} {'PASS':<10}")
        else:
            # Grab last line of stderr for a compact error
            err_lines = result.stderr.strip().splitlines()
            short_err = err_lines[-1] if err_lines else "(unknown error)"
            print(f"  {module:<25} {'FAIL':<10} {short_err}")
            failures += 1

    print()
    if failures == 0:
        print("  All modules import successfully.")
    else:
        print(f"  {failures} module(s) failed to import.")
    print()
    return 0 if failures == 0 else 1

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync and verify lib/ modules from player_profiles.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: show which files differ (default action)",
    )
    group.add_argument(
        "--sync",
        action="store_true",
        help="Copy source files to lib/ and fix imports",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="Test that all lib modules import correctly",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt during --sync",
    )

    args = parser.parse_args()

    # Default to --check when no flag given
    if not (args.check or args.sync or args.verify):
        args.check = True

    if args.check:
        rc = cmd_check()
    elif args.sync:
        rc = cmd_sync(force=args.force)
    elif args.verify:
        rc = cmd_verify()
    else:
        rc = 0

    sys.exit(rc)


if __name__ == "__main__":
    main()
