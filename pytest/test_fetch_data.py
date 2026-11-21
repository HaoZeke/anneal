# Copyright (c) anneal developers
"""The paper-data fetch checks only the manifest files it is asked for."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fetch_data(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("fetch_data", ROOT / "scripts" / "data" / "fetch_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    present = tmp_path / "data" / "present.txt"
    present.parent.mkdir()
    present.write_bytes(b"kept\n")
    manifest = tmp_path / "hf-files.csv"
    digest = hashlib.sha256(b"kept\n").hexdigest()
    manifest.write_text(
        "path,size,sha256\n"
        f"data/present.txt,5,{digest}\n"
        f"data/absent.txt,5,{digest}\n"
    )
    read = module.read_manifest
    monkeypatch.setattr(module, "read_manifest", lambda: read(manifest))
    monkeypatch.setattr(module, "HF_REVISION", "PINNED_REVISION")
    return module


def test_a_named_file_that_matches_needs_no_download(monkeypatch, tmp_path):
    fetch = _fetch_data(monkeypatch, tmp_path)
    fetch.ensure(root=tmp_path, paths=["data/present.txt"])


def test_every_file_is_checked_without_names(monkeypatch, tmp_path):
    fetch = _fetch_data(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="1 file"):
        fetch.ensure(root=tmp_path)


def test_a_name_outside_the_manifest_is_refused(monkeypatch, tmp_path):
    fetch = _fetch_data(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="not in"):
        fetch.ensure(root=tmp_path, paths=["data/other.txt"])
