"""Publish dashboard artifacts to the R2 bucket the deployed app reads.

Replaces committing ``data/dashboard`` to git. Uploads only what the dashboard
actually reads, skips unchanged objects, and writes an ``index.json`` so the
app can enumerate artifacts (object storage has no directory listing).

Credentials come from the environment and are never read from the repo, which
is public:

    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY

Usage
-----
    python scripts/publish_artifacts.py            # upload changed artifacts
    python scripts/publish_artifacts.py --dry-run  # show what would upload
    python scripts/publish_artifacts.py --prune    # also delete remote orphans
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Running this file directly puts scripts/ on sys.path rather than the project
# root, so repo imports (lib.db) would fail without this.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
BUCKET = os.environ.get("R2_BUCKET", "data-diamond")

# Extensions the dashboard can read.
PUBLISHED_SUFFIXES = {".parquet", ".json", ".npz", ".pkl"}

# Never published: the per-date sample archive is local-only bulk, and salvage
# copies of unreadable files are triage artefacts.
EXCLUDED_DIRS = {"history"}
EXCLUDED_SUFFIXES = {".corrupt_backup"}

# npz posterior samples that no dashboard code path reads. They are inputs to
# the player_profiles sim engine and stay on the local disk only. Verified by
# tracing every np.load / loader call site; the two game-sim files lost their
# last callers in 389cc98c when Layer 2 was removed.
UNREAD_NPZ = {
    "batter_game_sim_samples.npz",
    "pitcher_game_sim_samples.npz",
    "game_run_sim_samples.npz",
    "hitter_hr_samples.npz",
    "hitter_k_samples.npz",
    "hitter_bb_samples.npz",
    "pitcher_bb_samples.npz",
    "pitcher_hr_samples.npz",
    "milb_batter_bb_rate_samples.npz",
    "milb_batter_hr_rate_samples.npz",
    "milb_batter_k_rate_samples.npz",
    "milb_pitcher_bb_rate_samples.npz",
    "milb_pitcher_hr_per_bf_samples.npz",
    "milb_pitcher_k_rate_samples.npz",
}


def should_publish(path: Path) -> bool:
    """True when ``path`` is an artifact the deployed dashboard may read."""
    if path.suffix in EXCLUDED_SUFFIXES or path.suffix not in PUBLISHED_SUFFIXES:
        return False
    relative = path.relative_to(DASHBOARD_DIR)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.name in UNREAD_NPZ:
        return False
    # Preseason sample archives mirror the unread top-level npz.
    if path.suffix == ".npz" and "_preseason" in path.stem:
        return False
    return True


def collect() -> list[Path]:
    if not DASHBOARD_DIR.exists():
        sys.exit(f"No artifact directory at {DASHBOARD_DIR}")
    return sorted(p for p in DASHBOARD_DIR.rglob("*") if p.is_file() and should_publish(p))


def key_for(path: Path) -> str:
    return str(path.relative_to(DASHBOARD_DIR)).replace(os.sep, "/")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _credential(name: str) -> str | None:
    """Read a credential from the process environment, falling back to .env.

    Reuses the repo's existing .env convention (see lib/db.py) so the pipeline
    has one place for secrets. .env is gitignored; nothing here is ever read
    from a committed file.
    """
    value = os.environ.get(name)
    if value:
        return value
    from lib.db import _parse_dotenv

    return _parse_dotenv(PROJECT_ROOT / ".env").get(name) or None


def build_client():
    account = _credential("R2_ACCOUNT_ID")
    access_key = _credential("R2_ACCESS_KEY_ID")
    secret = _credential("R2_SECRET_ACCESS_KEY")
    missing = [
        name
        for name, value in (
            ("R2_ACCOUNT_ID", account),
            ("R2_ACCESS_KEY_ID", access_key),
            ("R2_SECRET_ACCESS_KEY", secret),
        )
        if not value
    ]
    if missing:
        sys.exit(f"Missing credentials in environment: {', '.join(missing)}")

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        sys.exit("boto3 is required: pip install boto3")

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret,
        # R2 ignores regions but botocore requires one.
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def remote_etags(client) -> dict[str, str]:
    """Map object key -> ETag for everything currently in the bucket."""
    etags: dict[str, str] = {}
    token = None
    while True:
        kwargs = {"Bucket": BUCKET}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            etags[item["Key"]] = item["ETag"].strip('"')
        if not response.get("IsTruncated"):
            return etags
        token = response.get("NextContinuationToken")


CONTENT_TYPES = {
    ".json": "application/json",
    ".parquet": "application/octet-stream",
    ".npz": "application/octet-stream",
    ".pkl": "application/octet-stream",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report without uploading")
    parser.add_argument("--prune", action="store_true", help="delete remote objects with no local file")
    args = parser.parse_args()

    files = collect()
    index = {
        "artifacts": [key_for(p) for p in files],
        # Object storage has no cheap stat, so sizes ride along for Data Health.
        "sizes": {key_for(p): p.stat().st_size for p in files},
    }
    total_mb = sum(p.stat().st_size for p in files) / 1048576
    print(f"{len(files)} artifacts selected ({total_mb:.1f} MB)")

    if args.dry_run:
        skipped = [
            p for p in DASHBOARD_DIR.rglob("*")
            if p.is_file() and p.suffix in PUBLISHED_SUFFIXES and not should_publish(p)
        ]
        skipped_mb = sum(p.stat().st_size for p in skipped) / 1048576
        print(f"{len(skipped)} skipped as unread by the dashboard ({skipped_mb:.1f} MB)")
        for p in files[:10]:
            print(f"  would upload {key_for(p)}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
        return 0

    client = build_client()
    existing = remote_etags(client)

    uploaded = skipped = 0
    for path in files:
        key = key_for(path)
        # Multipart ETags contain a dash and are not plain MD5; always re-upload
        # those rather than guessing the part size.
        etag = existing.get(key)
        if etag and "-" not in etag and etag == md5_of(path):
            skipped += 1
            continue
        client.upload_file(
            str(path), BUCKET, key,
            ExtraArgs={"ContentType": CONTENT_TYPES.get(path.suffix, "application/octet-stream")},
        )
        uploaded += 1

    client.put_object(
        Bucket=BUCKET,
        Key="index.json",
        Body=json.dumps(index, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"uploaded {uploaded}, unchanged {skipped}, index.json refreshed")

    if args.prune:
        published = set(index["artifacts"]) | {"index.json"}
        orphans = [key for key in existing if key not in published]
        for key in orphans:
            client.delete_object(Bucket=BUCKET, Key=key)
        print(f"pruned {len(orphans)} remote object(s) with no local file")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
