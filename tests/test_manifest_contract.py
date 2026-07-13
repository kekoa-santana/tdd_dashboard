"""Contract-validation tests for dashboard artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from services.manifest import (
    _column_hash,
    generate_manifest,
    validate_manifest,
    write_manifest,
)


def _manifest(columns: list[str]) -> dict:
    return {
        "contract_version": 1,
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "producer_repo": "mlb_bayesian_projections",
        "release_status": "canonical",
        "validation_status": "validated",
        "artifacts": [
            {
                "artifact_name": "players.parquet",
                "row_count": 1,
                "column_hash": _column_hash(columns),
                "required_columns": ["player_id", "projection"],
                "validation_status": "validated",
            }
        ],
    }


def test_strict_manifest_accepts_matching_schema(tmp_path: Path) -> None:
    columns = ["player_id", "projection"]
    pd.DataFrame([[1, 0.25]], columns=columns).to_parquet(
        tmp_path / "players.parquet", index=False
    )
    write_manifest(tmp_path, _manifest(columns))

    status = validate_manifest(tmp_path, strict=True)

    assert status.valid
    assert not status.schema_mismatches


def test_strict_manifest_rejects_required_column_drift(tmp_path: Path) -> None:
    pd.DataFrame({"player_id": [1]}).to_parquet(
        tmp_path / "players.parquet", index=False
    )
    write_manifest(tmp_path, _manifest(["player_id", "projection"]))

    status = validate_manifest(tmp_path, strict=True)

    assert not status.valid
    assert any("projection" in issue for issue in status.schema_mismatches)


def test_lenient_manifest_warns_for_experimental_release(tmp_path: Path) -> None:
    columns = ["player_id", "projection"]
    pd.DataFrame([[1, 0.25]], columns=columns).to_parquet(
        tmp_path / "players.parquet", index=False
    )
    manifest = _manifest(columns)
    manifest["release_status"] = "experimental"
    write_manifest(tmp_path, manifest)

    status = validate_manifest(tmp_path)

    assert status.valid
    assert status.release_status == "experimental"
    assert any("experimental" in warning for warning in status.warnings)


def test_write_manifest_replaces_valid_json(tmp_path: Path) -> None:
    destination = write_manifest(tmp_path, {"schema_version": 1})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "schema_version": 1
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_generate_manifest_preserves_and_enforces_required_columns(
    tmp_path: Path,
) -> None:
    columns = ["player_id", "projection"]
    pd.DataFrame([[1, 0.25]], columns=columns).to_parquet(
        tmp_path / "players.parquet", index=False
    )
    write_manifest(tmp_path, _manifest(columns))
    pd.DataFrame({"player_id": [1]}).to_parquet(
        tmp_path / "players.parquet", index=False
    )

    try:
        generate_manifest(tmp_path)
    except ValueError as exc:
        assert "projection" in str(exc)
    else:
        raise AssertionError("Required-column drift should prevent publication")
