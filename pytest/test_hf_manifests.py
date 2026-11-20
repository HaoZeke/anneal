# Copyright (c) anneal developers
"""The pinned Hugging Face tree agrees with both paper-data manifests."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import http.client
import importlib.util
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HF_FETCH_ATTEMPTS = 3
HF_FETCH_TIMEOUT_S = 20.0
HF_RETRY_BACKOFF_S = 0.5
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


class _OfflineError(RuntimeError):
    """Raised when Hugging Face cannot be reached after bounded retries."""


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetch_url(
    url: str,
    *,
    attempts: int = HF_FETCH_ATTEMPTS,
    timeout: float = HF_FETCH_TIMEOUT_S,
    backoff: float = HF_RETRY_BACKOFF_S,
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read(), response.headers.get("Link", "")
        except urllib.error.HTTPError as exc:
            if exc.code not in TRANSIENT_HTTP_CODES:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last_error = exc
        if attempt + 1 < attempts and backoff > 0:
            time.sleep(backoff * (attempt + 1))
    raise _OfflineError(f"{url}: {last_error}") from last_error


def dataset_tree(repository: str, revision: str) -> dict[str, dict]:
    url = (
        f"https://huggingface.co/api/datasets/{repository}/tree/{revision}"
        "?recursive=true&limit=1000"
    )
    entries = []
    while url:
        body, link = _fetch_url(url)
        entries.extend(json.loads(body))
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else ""
    return {entry["path"]: entry for entry in entries if entry["type"] == "file"}


def downloaded_object(repository: str, revision: str, row: dict[str, str]):
    quoted = urllib.parse.quote(row["path"])
    url = (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{revision}/{quoted}"
    )
    body, _ = _fetch_url(url)
    return row["path"], len(body), hashlib.sha256(body).hexdigest()


def test_pinned_hf_tree_matches_every_manifest_row():
    data = load("paper_fetch_data", "scripts/data/fetch_data.py")
    bench = load("paper_fetch_bench_data", "paper/jctc/scripts/fetch_bench_data.py")
    assert data.HF_REVISION_DEFAULT == bench.HF_REVISION_DEFAULT
    assert data.HF_REPO == bench.HF_DATASET

    try:
        tree = dataset_tree(data.HF_REPO, data.HF_REVISION_DEFAULT)
    except _OfflineError as exc:
        pytest.skip(f"Hugging Face unreachable: {exc}")

    rows = data.read_manifest()
    receipt_path = ROOT / "data/hf-tree.csv"
    with receipt_path.open(newline="") as handle:
        receipt = {row["path"]: row for row in csv.DictReader(handle)}
    assert set(receipt) == {row["path"] for row in rows}
    for row in rows:
        entry = tree[row["path"]]
        recorded = receipt[row["path"]]
        assert int(row["size"]) == entry["size"] == int(recorded["size"])
        assert entry["oid"] == recorded["git_oid"]
        lfs_sha256 = entry.get("lfs", {}).get("oid", "")
        assert lfs_sha256 == recorded["lfs_sha256"]
        if lfs_sha256:
            assert lfs_sha256 == row["sha256"]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            downloaded = {
                path: (size, digest)
                for path, size, digest in executor.map(
                    lambda row: downloaded_object(
                        data.HF_REPO, data.HF_REVISION_DEFAULT, row
                    ),
                    rows,
                )
            }
    except _OfflineError as exc:
        pytest.skip(f"Hugging Face unreachable: {exc}")

    assert downloaded == {
        row["path"]: (int(row["size"]), row["sha256"])
        for row in rows
    }

    bench_paths = bench.manifest_paths()
    assert all(path.as_posix() in tree for path in bench_paths)


def test_hf_manifest_updates_match_the_downloaded_objects():
    data = load("paper_fetch_data_updates", "scripts/data/fetch_data.py")
    with data.MANIFEST_UPDATES.open(newline="") as handle:
        updates = [row for row in csv.DictReader(handle) if row["status"] == "active"]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            downloaded = dict(
                (path, (size, digest))
                for path, size, digest in executor.map(
                    lambda row: downloaded_object(
                        data.HF_REPO, data.HF_REVISION_DEFAULT, row
                    ),
                    updates,
                )
            )
    except _OfflineError as exc:
        pytest.skip(f"Hugging Face unreachable: {exc}")

    assert downloaded == {
        row["path"]: (int(row["size"]), row["sha256"])
        for row in updates
    }


def test_hf_fetch_skips_cleanly_when_offline(monkeypatch):
    calls: list[tuple[str, float]] = []

    def offline(url, timeout=None):
        calls.append((url, timeout))
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    with pytest.raises(pytest.skip.Exception, match="Hugging Face unreachable"):
        test_pinned_hf_tree_matches_every_manifest_row()
    assert len(calls) == HF_FETCH_ATTEMPTS
    assert all(timeout == HF_FETCH_TIMEOUT_S for _, timeout in calls)


def test_hf_fetch_does_not_skip_on_404(monkeypatch):
    calls = 0

    def not_found(url, timeout=None):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    with pytest.raises(urllib.error.HTTPError):
        _fetch_url("https://huggingface.co/api/datasets/example/tree/deadbeef")
    assert calls == 1
