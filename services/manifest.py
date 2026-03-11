"""Manifest validation for dashboard data artifacts.

Validates that pre-computed parquet/npz files in data/dashboard/ match
the manifest contract produced by the player_profiles precompute pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DASHBOARD_DIR, RUNTIME

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA_VERSION: int = RUNTIME["schema_version"]
STALE_THRESHOLD_HOURS: float = 24.0
ARTIFACT_EXTENSIONS: set[str] = {".parquet", ".npz"}


@dataclass
class ManifestStatus:
    """Result of manifest validation."""

    valid: bool
    missing_artifacts: list[str] = field(default_factory=list)
    extra_artifacts: list[str] = field(default_factory=list)
    row_count_mismatches: list[str] = field(default_factory=list)
    manifest_age_hours: float | None = None
    warnings: list[str] = field(default_factory=list)


def _column_hash(columns: list[str]) -> str:
    """Compute sha256 of sorted column names joined with commas."""
    payload = ",".join(sorted(columns))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_git_commit() -> str | None:
    """Return short git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _scan_artifact_files(dashboard_dir: Path) -> list[str]:
    """Return sorted list of artifact filenames (parquet/npz) on disk."""
    files: list[str] = []
    if not dashboard_dir.exists():
        return files
    for p in dashboard_dir.iterdir():
        if p.is_file() and p.suffix in ARTIFACT_EXTENSIONS:
            files.append(p.name)
    return sorted(files)


def load_manifest(dashboard_dir: Path) -> dict | None:
    """Load manifest.json from *dashboard_dir*. Returns None if not found."""
    manifest_path = dashboard_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read manifest.json: %s", exc)
        return None


def validate_manifest(
    dashboard_dir: Path,
    strict: bool = False,
) -> ManifestStatus:
    """Validate dashboard artifacts against manifest.

    In strict mode, missing artifacts and row-count mismatches cause
    ``valid=False``. In lenient mode (default), they are reported as
    warnings but ``valid`` remains True.

    Checks
    ------
    1. manifest.json exists
    2. schema_version matches ``RUNTIME["schema_version"]``
    3. All artifacts listed in manifest exist on disk
    4. Row counts match (parquet files only)
    5. Manifest is not stale (warn if > 24 h old)
    """
    manifest = load_manifest(dashboard_dir)

    # 1. Manifest existence
    if manifest is None:
        return ManifestStatus(
            valid=not strict,
            warnings=["manifest.json not found — skipping artifact validation"],
        )

    warnings: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    row_mismatches: list[str] = []
    age_hours: float | None = None

    # 2. Schema version
    manifest_version = manifest.get("schema_version")
    if manifest_version != EXPECTED_SCHEMA_VERSION:
        warnings.append(
            f"Schema version mismatch: manifest has {manifest_version}, "
            f"expected {EXPECTED_SCHEMA_VERSION}"
        )

    # 5. Staleness (check early so we always report age)
    generated_at = manifest.get("generated_at")
    if generated_at:
        try:
            gen_dt = datetime.fromisoformat(generated_at)
            # If naive, assume UTC
            if gen_dt.tzinfo is None:
                gen_dt = gen_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age_hours = (now - gen_dt).total_seconds() / 3600.0
            if age_hours > STALE_THRESHOLD_HOURS:
                warnings.append(
                    f"Manifest is {age_hours:.1f}h old (threshold: "
                    f"{STALE_THRESHOLD_HOURS}h)"
                )
        except (ValueError, TypeError):
            warnings.append("Could not parse manifest generated_at timestamp")

    # Build lookup of manifest artifacts
    artifacts: list[dict] = manifest.get("artifacts", [])
    manifest_names: set[str] = set()
    artifact_map: dict[str, dict] = {}
    for art in artifacts:
        name = art.get("artifact_name", "")
        manifest_names.add(name)
        artifact_map[name] = art

    # Files on disk
    disk_names = set(_scan_artifact_files(dashboard_dir))

    # 3. Missing artifacts (in manifest but not on disk)
    for name in sorted(manifest_names - disk_names):
        missing.append(name)

    # Extra artifacts (on disk but not in manifest)
    for name in sorted(disk_names - manifest_names):
        extra.append(name)

    if missing:
        warnings.append(f"{len(missing)} artifact(s) missing from disk")
    if extra:
        warnings.append(f"{len(extra)} artifact(s) on disk not in manifest")

    # 4. Row-count validation (parquet only)
    for name, art in artifact_map.items():
        if not name.endswith(".parquet"):
            continue
        file_path = dashboard_dir / name
        if not file_path.exists():
            continue  # already captured as missing
        expected_rows = art.get("row_count")
        if expected_rows is None:
            continue
        try:
            actual_rows = len(pd.read_parquet(file_path))
            if actual_rows != expected_rows:
                msg = (
                    f"{name}: row count {actual_rows} != manifest {expected_rows}"
                )
                row_mismatches.append(msg)
        except Exception as exc:
            warnings.append(f"Could not read {name} for row check: {exc}")

    if row_mismatches:
        warnings.append(
            f"{len(row_mismatches)} artifact(s) have row-count mismatches"
        )

    # Determine validity
    has_errors = bool(missing) or bool(row_mismatches)
    valid = not has_errors if strict else True

    return ManifestStatus(
        valid=valid,
        missing_artifacts=missing,
        extra_artifacts=extra,
        row_count_mismatches=row_mismatches,
        manifest_age_hours=age_hours,
        warnings=warnings,
    )


def generate_manifest(dashboard_dir: Path) -> dict:
    """Generate a manifest from the current state of files on disk.

    Scans all parquet/npz files, counts rows (parquet), and hashes
    column names. Used by ``update_in_season.py`` to produce a fresh
    manifest after each update run.
    """
    artifacts: list[dict] = []

    for name in _scan_artifact_files(dashboard_dir):
        file_path = dashboard_dir / name
        entry: dict = {
            "artifact_name": name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if name.endswith(".parquet"):
            try:
                df = pd.read_parquet(file_path)
                entry["row_count"] = len(df)
                entry["column_hash"] = _column_hash(list(df.columns))
            except Exception as exc:
                logger.warning("Could not read %s for manifest: %s", name, exc)
                entry["row_count"] = None
                entry["column_hash"] = None
        elif name.endswith(".npz"):
            entry["row_count"] = None
            entry["column_hash"] = None

        artifacts.append(entry)

    manifest: dict = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "producer_repo": "tdd-dashboard",
        "producer_commit": _get_git_commit(),
        "artifacts": artifacts,
    }
    return manifest
