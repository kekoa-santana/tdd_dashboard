"""Artifact resolution: local-first, remote fallback, and failure handling."""
from __future__ import annotations

import json
import time

import pytest

import config
from services import artifacts


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point the artifact dir and download cache at a scratch directory."""
    local = tmp_path / "dashboard"
    local.mkdir()
    monkeypatch.setattr(config, "DASHBOARD_DIR", local)
    monkeypatch.setattr(artifacts, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "")
    artifacts._fetch_failures.clear()
    return local


class _Response:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


def _stub_requests(monkeypatch, handler):
    """Install a fake requests module for artifacts._download to import."""
    import sys
    import types

    module = types.ModuleType("requests")
    module.get = handler
    monkeypatch.setitem(sys.modules, "requests", module)


def test_local_file_wins_and_no_network_is_used(isolated, monkeypatch):
    (isolated / "game_props.parquet").write_bytes(b"local")
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://example.invalid")

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("network used despite a local file being present")

    _stub_requests(monkeypatch, explode)

    path = artifacts.artifact_path("game_props.parquet")
    assert path.read_bytes() == b"local"


def test_without_bucket_missing_file_resolves_to_absent_local_path(isolated):
    path = artifacts.artifact_path("absent.parquet")
    assert not path.exists()
    assert path.parent == isolated


def test_remote_download_is_cached_and_fetched_once(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    calls: list[str] = []

    def handler(url, timeout=None, **kw):
        calls.append(url)
        return _Response(200, b"remote-bytes")

    _stub_requests(monkeypatch, handler)

    first = artifacts.artifact_path("game_props.parquet")
    assert first.read_bytes() == b"remote-bytes"
    assert calls == ["https://cdn.example/game_props.parquet"]

    # Second lookup is served from the on-disk cache, not refetched.
    second = artifacts.artifact_path("game_props.parquet")
    assert second == first
    assert len(calls) == 1


def test_nested_names_are_requested_with_forward_slashes(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    seen: list[str] = []
    _stub_requests(monkeypatch, lambda url, timeout=None, **kw: (seen.append(url), _Response(200, b"x"))[1])

    artifacts.artifact_path("snapshots/weekly/hitter_projections_2026-08-24.parquet")
    assert seen == [
        "https://cdn.example/snapshots/weekly/hitter_projections_2026-08-24.parquet"
    ]


def test_404_degrades_quietly_without_recording_an_outage(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    _stub_requests(monkeypatch, lambda url, timeout=None, **kw: _Response(404))

    path = artifacts.artifact_path("never_published.parquet")

    # Callers do `if not path.exists(): return pd.DataFrame()`, so an artifact
    # that was simply never published must not raise.
    assert not path.exists()
    assert artifacts.fetch_failures() == {}


def test_transport_failure_is_recorded_but_still_does_not_raise(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")

    def boom(url, timeout=None, **kw):
        raise OSError("connection reset")

    _stub_requests(monkeypatch, boom)

    path = artifacts.artifact_path("game_props.parquet")
    assert not path.exists()
    failures = artifacts.fetch_failures()
    assert "game_props.parquet" in failures
    assert "connection reset" in failures["game_props.parquet"]


def test_partial_download_never_leaves_a_truncated_file(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")

    class _Exploding(_Response):
        @property
        def content(self):  # type: ignore[override]
            raise OSError("stream died mid-body")

        @content.setter
        def content(self, value):
            pass

    _stub_requests(monkeypatch, lambda url, timeout=None, **kw: _Exploding(200))

    path = artifacts.artifact_path("game_props.parquet")
    assert not path.exists()


def test_expired_cache_picks_up_a_new_publish(isolated, monkeypatch):
    """A long-lived container must not serve its first download forever."""
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    body = {"n": 0}

    def handler(url, timeout=None, **kw):
        body["n"] += 1
        return _Response(200, f"version-{body['n']}".encode())

    _stub_requests(monkeypatch, handler)

    first = artifacts.artifact_path("game_props.parquet")
    assert first.read_bytes() == b"version-1"

    # Age the cached file past its TTL, as happens on a container that has been
    # up longer than one publish cycle.
    import os as _os
    stale = time.time() - (artifacts._CACHE_TTL_SECONDS + 60)
    _os.utime(first, (stale, stale))

    second = artifacts.artifact_path("game_props.parquet")
    assert second.read_bytes() == b"version-2", "expired cache was not refreshed"


def test_unchanged_artifact_revalidates_with_304_and_is_not_redownloaded(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    sent_headers: list[dict] = []

    def handler(url, timeout=None, headers=None, **kw):
        sent_headers.append(headers or {})
        if (headers or {}).get("If-None-Match") == '"abc123"':
            return _Response(304)
        r = _Response(200, b"payload")
        r.headers = {"ETag": '"abc123"'}
        return r

    _stub_requests(monkeypatch, handler)

    path = artifacts.artifact_path("game_props.parquet")
    assert path.read_bytes() == b"payload"

    import os as _os
    stale = time.time() - (artifacts._CACHE_TTL_SECONDS + 60)
    _os.utime(path, (stale, stale))

    again = artifacts.artifact_path("game_props.parquet")
    # 304 leaves the bytes alone and restarts the freshness clock.
    assert again.read_bytes() == b"payload"
    assert sent_headers[-1].get("If-None-Match") == '"abc123"'
    assert artifacts._age_seconds(again) < artifacts._CACHE_TTL_SECONDS


def test_outage_keeps_serving_the_stale_copy(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    state = {"fail": False}

    def handler(url, timeout=None, **kw):
        if state["fail"]:
            raise OSError("bucket unreachable")
        return _Response(200, b"good-data")

    _stub_requests(monkeypatch, handler)
    path = artifacts.artifact_path("game_props.parquet")
    assert path.read_bytes() == b"good-data"

    import os as _os
    stale = time.time() - (artifacts._CACHE_TTL_SECONDS + 60)
    _os.utime(path, (stale, stale))
    state["fail"] = True

    again = artifacts.artifact_path("game_props.parquet")
    # Serving slightly stale data beats blanking the page during an outage.
    assert again.read_bytes() == b"good-data"
    assert "game_props.parquet" in artifacts.fetch_failures()


def test_list_artifacts_walks_local_directory_when_populated(isolated):
    weekly = isolated / "snapshots" / "weekly"
    weekly.mkdir(parents=True)
    (weekly / "hitter_projections_2026-08-24.parquet").write_bytes(b"a")
    (weekly / "pitcher_projections_2026-08-24.parquet").write_bytes(b"b")

    names = artifacts.list_artifacts("snapshots/weekly")
    assert names == [
        "snapshots/weekly/hitter_projections_2026-08-24.parquet",
        "snapshots/weekly/pitcher_projections_2026-08-24.parquet",
    ]


def test_list_artifacts_falls_back_to_published_index(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    index = {
        "artifacts": [
            "game_props.parquet",
            "snapshots/weekly/hitter_projections_2026-08-24.parquet",
        ]
    }
    _stub_requests(
        monkeypatch,
        lambda url, timeout=None, **kw: _Response(200, json.dumps(index).encode()),
    )

    assert artifacts.list_artifacts("snapshots/weekly") == [
        "snapshots/weekly/hitter_projections_2026-08-24.parquet"
    ]


def test_missing_index_yields_empty_listing_rather_than_raising(isolated, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE_URL", "https://cdn.example")
    _stub_requests(monkeypatch, lambda url, timeout=None, **kw: _Response(404))
    assert artifacts.list_artifacts("snapshots/weekly") == []


def test_base_url_is_read_from_lowercase_streamlit_secret(monkeypatch):
    """Streamlit Cloud does not mirror a lowercase secret into os.environ."""
    import sys
    import types

    monkeypatch.delenv("TDD_ARTIFACT_BASE_URL", raising=False)
    monkeypatch.delenv("tdd_artifact_base_url", raising=False)
    fake = types.ModuleType("streamlit")
    fake.secrets = {"tdd_artifact_base_url": "https://pub-x.r2.dev/"}
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    assert artifacts._resolve_base_url() == "https://pub-x.r2.dev"
