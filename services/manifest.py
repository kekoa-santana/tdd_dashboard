"""Manifest validation for dashboard data artifacts.

Validates that pre-computed parquet/npz files in data/dashboard/ match
the manifest contract produced by the player_profiles precompute pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from config import RUNTIME

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
    schema_mismatches: list[str] = field(default_factory=list)
    invalid_artifacts: list[str] = field(default_factory=list)
    manifest_age_hours: float | None = None
    release_status: str | None = None
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
    2. contract version matches ``RUNTIME["schema_version"]``
    3. All artifacts listed in manifest exist on disk
    4. Row counts match (parquet files only)
    5. Required columns and column hashes match
    6. Producer validation and release status are safe
    7. Manifest is not stale (warn if > 24 h old)
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
    schema_mismatches: list[str] = []
    invalid_artifacts: list[str] = []
    age_hours: float | None = None

    # 2. Schema version
    manifest_version = manifest.get("contract_version", manifest.get("schema_version"))
    if manifest_version != EXPECTED_SCHEMA_VERSION:
        schema_mismatches.append(
            f"Schema version mismatch: manifest has {manifest_version}, "
            f"expected {EXPECTED_SCHEMA_VERSION}"
        )

    release_status = manifest.get("release_status")
    if release_status == "experimental":
        warnings.append("Artifacts were produced by an experimental/quick run")
    elif release_status == "internal-only":
        schema_mismatches.append("Internal-only artifacts cannot power the dashboard")

    if manifest.get("validation_status") not in {None, "validated"}:
        invalid_artifacts.append("Manifest producer validation did not pass")

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

    # 4-5. Parquet row-count and schema validation
    for name, art in artifact_map.items():
        if art.get("validation_status") not in {None, "validated"}:
            invalid_artifacts.append(f"{name}: producer validation did not pass")
        if not name.endswith(".parquet"):
            continue
        file_path = dashboard_dir / name
        if not file_path.exists():
            continue  # already captured as missing
        try:
            frame = pd.read_parquet(file_path)
            expected_rows = art.get("row_count")
            if expected_rows is not None and len(frame) != expected_rows:
                msg = (
                    f"{name}: row count {len(frame)} != manifest {expected_rows}"
                )
                row_mismatches.append(msg)

            required_columns = set(art.get("required_columns", []))
            missing_columns = sorted(required_columns - set(frame.columns))
            if missing_columns:
                schema_mismatches.append(
                    f"{name}: missing required columns {missing_columns}"
                )

            expected_hash = art.get("column_hash")
            actual_hash = _column_hash([str(column) for column in frame.columns])
            if expected_hash is not None and actual_hash != expected_hash:
                schema_mismatches.append(f"{name}: column hash does not match manifest")
        except Exception as exc:
            invalid_artifacts.append(f"Could not validate {name}: {exc}")

    if row_mismatches:
        warnings.append(
            f"{len(row_mismatches)} artifact(s) have row-count mismatches"
        )
    if schema_mismatches:
        warnings.append(f"{len(schema_mismatches)} schema contract issue(s)")
    if invalid_artifacts:
        warnings.append(f"{len(invalid_artifacts)} artifact validation issue(s)")

    # Determine validity
    has_errors = bool(
        missing or row_mismatches or schema_mismatches or invalid_artifacts
    )
    valid = not has_errors if strict else True

    return ManifestStatus(
        valid=valid,
        missing_artifacts=missing,
        extra_artifacts=extra,
        row_count_mismatches=row_mismatches,
        schema_mismatches=schema_mismatches,
        invalid_artifacts=invalid_artifacts,
        manifest_age_hours=age_hours,
        release_status=release_status,
        warnings=warnings,
    )


def write_manifest(dashboard_dir: Path, manifest: dict) -> Path:
    """Atomically publish ``manifest.json`` in the dashboard artifact directory."""
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    destination = dashboard_dir / "manifest.json"
    handle = tempfile.NamedTemporaryFile(
        dir=dashboard_dir,
        prefix=".manifest.json.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def generate_manifest(dashboard_dir: Path) -> dict:
    """Generate a manifest from the current state of files on disk.

    Scans all parquet/npz files, counts rows (parquet), and hashes
    column names. Used by ``update_in_season.py`` to produce a fresh
    manifest after each update run.
    """
    previous = load_manifest(dashboard_dir) or {}
    previous_artifacts = {
        artifact.get("artifact_name"): artifact
        for artifact in previous.get("artifacts", [])
        if artifact.get("artifact_name")
    }
    contract_fields = {
        "schema_version",
        "model_version",
        "classification",
        "consumers",
        "required_upstream_artifacts",
        "required_columns",
        "key_pattern",
    }
    artifacts: list[dict] = []

    for name in _scan_artifact_files(dashboard_dir):
        file_path = dashboard_dir / name
        entry: dict = {
            "artifact_name": name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "validation_status": "validated",
        }
        prior = previous_artifacts.get(name, {})
        entry.update({key: prior[key] for key in contract_fields if key in prior})

        if name.endswith(".parquet"):
            try:
                df = pd.read_parquet(file_path)
                entry["row_count"] = len(df)
                entry["column_hash"] = _column_hash(list(df.columns))
            except Exception as exc:
                raise ValueError(f"Could not read {name} for manifest: {exc}") from exc
            required = set(entry.get("required_columns", []))
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(f"{name} is missing required columns: {missing}")
        elif name.endswith(".npz"):
            entry["row_count"] = None
            entry["column_hash"] = None
            try:
                with np.load(file_path) as archive:
                    entry["array_count"] = len(archive.files)
            except Exception as exc:
                raise ValueError(f"Could not read {name} for manifest: {exc}") from exc

        artifacts.append(entry)

    manifest: dict = {
        "contract_version": previous.get(
            "contract_version", previous.get("schema_version", EXPECTED_SCHEMA_VERSION)
        ),
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "producer_repo": "tdd-dashboard",
        "producer_commit": _get_git_commit(),
        "target_season": RUNTIME["current_season"],
        "release_status": "canonical",
        "validation_status": "validated",
        "artifacts": artifacts,
    }
    return manifest
