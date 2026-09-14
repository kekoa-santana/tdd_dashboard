"""Artifact resolution: local files when present, object storage otherwise.

The dashboard reads pre-computed parquet/npz/json artifacts. Locally those sit
in ``data/dashboard`` and are read straight off disk. On the deployed app that
directory is not in the repo, so artifacts are fetched from the public R2
bucket on first use and cached on the container's local disk.

Cached copies expire after ``_CACHE_TTL_SECONDS`` and are then revalidated with
a conditional GET, so a container that stays up for days still picks up each
new publish instead of serving its first download forever.

Local files always win. That keeps the dev loop offline and unchanged, and it
means the pipeline (which writes locally, then uploads) never round-trips
through the network to read its own output.

Object storage has no directory listing, so anything that needs to enumerate
artifacts reads ``index.json``, which the publish step writes alongside them.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import config

_BASE_URL_KEY = "TDD_ARTIFACT_BASE_URL"


def _resolve_base_url() -> str:
    """Read the bucket URL from the environment, then Streamlit secrets.

    Streamlit Cloud only mirrors root-level secrets into os.environ, and
    environment names are case-sensitive on Linux, so a secret saved as
    ``tdd_artifact_base_url`` would silently leave the app local-only. Checking
    st.secrets case-insensitively removes that failure mode.
    """
    for key in (_BASE_URL_KEY, _BASE_URL_KEY.lower()):
        if os.environ.get(key):
            return os.environ[key].strip().rstrip("/")
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            if key.upper() == _BASE_URL_KEY and isinstance(value, str) and value.strip():
                return value.strip().rstrip("/")
    except Exception:  # no secrets file locally, or streamlit unavailable
        pass
    return ""


# Public base URL for the artifact bucket, e.g. "https://pub-xxxx.r2.dev".
# Empty means local-only: every lookup resolves to data/dashboard.
ARTIFACT_BASE_URL: str = _resolve_base_url()

# Downloads land here and persist for the life of the container.
_CACHE_DIR = Path(tempfile.gettempdir()) / "tdd_artifacts"

_INDEX_NAME = "index.json"
_TIMEOUT = 30
_HEADERS = {"User-Agent": "tdd-dashboard/1.0 (+https://thedatadiamond.com)"}

# How long a downloaded artifact is trusted before it is revalidated against
# the bucket. Must not exceed the loaders' cache TTL, or a long-lived container
# would keep serving whatever it downloaded first: st.cache_data would expire,
# re-run the loader, and be handed the same stale file. Revalidation is a
# conditional GET, so an unchanged artifact costs one small 304 rather than a
# re-download.
_CACHE_TTL_SECONDS = 240


class ArtifactUnavailable(RuntimeError):
    """The artifact store could not be reached, or returned an error."""


class ArtifactMissing(LookupError):
    """The artifact store is reachable but does not hold this artifact."""


# Names that failed to fetch for a reason other than "not published", kept so
# the UI can distinguish a genuine outage from an artifact that simply does not
# exist. Callers see a missing path either way.
_fetch_failures: dict[str, str] = {}


def fetch_failures() -> dict[str, str]:
    """Artifacts that failed to download, mapped to the reason."""
    return dict(_fetch_failures)


def remote_enabled() -> bool:
    """True when a remote artifact bucket is configured."""
    return bool(ARTIFACT_BASE_URL)


def _remote_url(name: str) -> str:
    return f"{ARTIFACT_BASE_URL}/{name.replace(os.sep, '/')}"


def _age_seconds(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return float("inf")


def _mark_fresh(path: Path) -> None:
    """Restart the freshness clock on a cached file confirmed still current."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def _etag_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".etag")


def _etag_of(path: Path) -> str | None:
    """The ETag we recorded when this cached artifact was downloaded."""
    if not path.exists():
        return None
    try:
        return _etag_sidecar(path).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _download(name: str, destination: Path) -> None:
    """Fetch ``name`` from the bucket, writing atomically to ``destination``.

    Raises ArtifactUnavailable on transport failure or a non-404 status. A 404
    means the artifact genuinely is not published, which callers treat the same
    as a missing local file, so it is reported as :class:`ArtifactMissing`.
    """
    import requests  # ships with streamlit; imported lazily to keep import cheap

    url = _remote_url(name)
    headers = dict(_HEADERS)
    # Revalidate rather than re-download when we already hold a copy: R2 replies
    # 304 for an unchanged object, which costs a few hundred bytes instead of
    # the whole parquet.
    etag = _etag_of(destination)
    if etag:
        headers["If-None-Match"] = etag

    try:
        # Identify ourselves explicitly: Cloudflare challenges some default
        # client User-Agents (bare urllib gets a 403 on the same object that
        # requests fetches fine), so do not depend on the library default.
        response = requests.get(url, timeout=_TIMEOUT, headers=headers)
    except Exception as exc:  # network error, DNS, TLS, timeout
        raise ArtifactUnavailable(f"Could not reach {url}: {exc}") from exc
    if response.status_code == 304:
        _mark_fresh(destination)  # unchanged; restart the local freshness clock
        return
    if response.status_code == 404:
        raise ArtifactMissing(f"{name} is not published")
    if response.status_code != 200:
        raise ArtifactUnavailable(f"{url} returned HTTP {response.status_code}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file first so a concurrent reader never sees a
    # half-written parquet. os.replace is atomic on the same filesystem.
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    )
    try:
        # Close via the context manager so the handle is released even if the
        # write raises; Windows refuses to unlink a file that is still open.
        with handle:
            handle.write(response.content)
        os.replace(handle.name, destination)
        new_etag = response.headers.get("ETag") if hasattr(response, "headers") else None
        if new_etag:
            _etag_sidecar(destination).write_text(new_etag.strip(), encoding="utf-8")
    except Exception as exc:
        # A body that dies mid-stream, a full disk, a permission error: the
        # destination is untouched because the rename never ran, and the
        # caller sees the same typed failure as any other transport problem.
        raise ArtifactUnavailable(f"Could not write {name}: {exc}") from exc
    finally:
        Path(handle.name).unlink(missing_ok=True)


def artifact_path(name: str) -> Path:
    """Return a readable local path for artifact ``name``.

    ``name`` is bucket-relative and may include subdirectories, e.g.
    ``"snapshots/weekly/hitter_projections_2026-08-24.parquet"``.

    Resolution order: the local artifact directory, then the container cache,
    then a download.

    This never raises. Callers throughout the dashboard follow the pattern
    ``if not path.exists(): return pd.DataFrame()``, so a fetch failure
    resolves to a path that does not exist and the page degrades exactly as it
    would for a missing local file. Non-404 failures are recorded in
    :func:`fetch_failures` so an outage can still be surfaced deliberately
    rather than looking like empty data.
    """
    local = config.DASHBOARD_DIR / name
    if local.exists():
        return local

    if not remote_enabled():
        return local

    cached = _CACHE_DIR / name
    if cached.exists() and _age_seconds(cached) < _CACHE_TTL_SECONDS:
        return cached

    try:
        _download(name, cached)
    except ArtifactMissing:
        _fetch_failures.pop(name, None)
        return cached  # does not exist; caller degrades as for a missing file
    except ArtifactUnavailable as exc:
        _fetch_failures[name] = str(exc)
        # A stale copy beats no data at all: if the bucket is briefly
        # unreachable, keep serving what we already have rather than blanking
        # the page. The failure is recorded either way.
        return cached
    _fetch_failures.pop(name, None)
    return cached


def load_index() -> dict:
    """Return the published artifact index, or an empty index if unavailable."""
    path = artifact_path(_INDEX_NAME)
    if not path.exists():
        return {"artifacts": []}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"artifacts": []}


def list_artifacts(prefix: str = "") -> list[str]:
    """List artifact names under ``prefix``.

    Falls back to walking the local directory when it is populated, so this
    works in development before anything has been published.
    """
    root = config.DASHBOARD_DIR
    local_root = root / prefix if prefix else root
    if local_root.exists():
        return sorted(
            str(p.relative_to(root)).replace(os.sep, "/")
            for p in local_root.rglob("*")
            if p.is_file()
        )

    names = load_index().get("artifacts", [])
    if prefix:
        prefix = prefix.rstrip("/") + "/"
        names = [n for n in names if n.startswith(prefix)]
    return sorted(names)
