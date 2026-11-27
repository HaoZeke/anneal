# Copyright (c) anneal developers
"""The Sutton-Chen harness refuses a GMIN binary or tarball that misses the pin."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sc_h2h", ROOT / "scripts" / "paper_rerun" / "sc_h2h.py")
sc_h2h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sc_h2h)

TARBALL = "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_tarball_digest_is_recorded_beside_the_pin():
    assert sc_h2h.GMIN_PIN == "9cbec1a446352bc9ef064bd2b1a5767194655919"
    assert sc_h2h.GMIN_TARBALL_SHA256 == TARBALL
    assert sc_h2h.GMIN_BINARY_SHA256 == ""


def test_an_unrecorded_binary_digest_refuses_the_arm(tmp_path: Path):
    binary = tmp_path / "GMIN"
    tarball = tmp_path / "svn.tar.bz2"
    source = tmp_path / "source.txt"
    binary.write_bytes(b"binary\n")
    tarball.write_bytes(b"tarball\n")
    source.write_text(f"svnrev={sc_h2h.GMIN_PIN}\n")
    assert sc_h2h.gmin_artifact_error(binary, tarball, source) == (
        "binary sha256 is not recorded beside the pin"
    )


def test_a_tarball_mismatch_is_refused(tmp_path: Path, monkeypatch):
    binary = tmp_path / "GMIN"
    tarball = tmp_path / "svn.tar.bz2"
    source = tmp_path / "source.txt"
    binary.write_bytes(b"the executable\n")
    tarball.write_bytes(b"not the pinned snapshot\n")
    source.write_text(f"svnrev={sc_h2h.GMIN_PIN}\n")
    monkeypatch.setattr(sc_h2h, "GMIN_BINARY_SHA256", _sha(binary))
    reason = sc_h2h.gmin_artifact_error(binary, tarball, source)
    assert reason is not None
    assert reason.startswith("tarball sha256 ")
    assert TARBALL in reason
    assert _sha(tarball) in reason


def test_a_binary_mismatch_is_refused(tmp_path: Path, monkeypatch):
    binary = tmp_path / "GMIN"
    other = tmp_path / "other"
    tarball = tmp_path / "svn.tar.bz2"
    source = tmp_path / "source.txt"
    binary.write_bytes(b"the executable\n")
    other.write_bytes(b"a different executable\n")
    tarball.write_bytes(b"snapshot\n")
    source.write_text(f"svnrev={sc_h2h.GMIN_PIN}\n")
    monkeypatch.setattr(sc_h2h, "GMIN_TARBALL_SHA256", _sha(tarball))
    monkeypatch.setattr(sc_h2h, "GMIN_BINARY_SHA256", _sha(other))
    reason = sc_h2h.gmin_artifact_error(binary, tarball, source)
    assert reason is not None
    assert reason.startswith("binary sha256 ")
    assert _sha(other) in reason
    assert _sha(binary) in reason


def test_matching_artifacts_and_svnrev_are_accepted(tmp_path: Path, monkeypatch):
    binary = tmp_path / "GMIN"
    tarball = tmp_path / "svn.tar.bz2"
    source = tmp_path / "source.txt"
    binary.write_bytes(b"the executable\n")
    tarball.write_bytes(b"snapshot\n")
    source.write_text(f"svnrev={sc_h2h.GMIN_PIN}\n")
    monkeypatch.setattr(sc_h2h, "GMIN_TARBALL_SHA256", _sha(tarball))
    monkeypatch.setattr(sc_h2h, "GMIN_BINARY_SHA256", _sha(binary))
    assert sc_h2h.gmin_artifact_error(binary, tarball, source) is None


def test_a_wrong_svnrev_is_refused_after_the_hashes_match(tmp_path: Path, monkeypatch):
    binary = tmp_path / "GMIN"
    tarball = tmp_path / "svn.tar.bz2"
    source = tmp_path / "source.txt"
    binary.write_bytes(b"the executable\n")
    tarball.write_bytes(b"snapshot\n")
    source.write_text("svnrev=deadbeef\n")
    monkeypatch.setattr(sc_h2h, "GMIN_TARBALL_SHA256", _sha(tarball))
    monkeypatch.setattr(sc_h2h, "GMIN_BINARY_SHA256", _sha(binary))
    assert sc_h2h.gmin_artifact_error(binary, tarball, source) == "svnrev deadbeef is not " + sc_h2h.GMIN_PIN


def test_a_missing_binary_or_tarball_is_refused(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sc_h2h, "GMIN_BINARY_SHA256", "ab" * 32)
    assert sc_h2h.gmin_artifact_error(None, tmp_path / "missing", None) == "binary absent"
    binary = tmp_path / "GMIN"
    binary.write_bytes(b"x\n")
    assert sc_h2h.gmin_artifact_error(binary, None, None) == "tarball absent"


def test_the_rerun_takes_its_machine_digest_from_the_helper():
    source = (ROOT / "scripts" / "paper_rerun" / "sc_h2h.py").read_text()
    assert "def machine_id(" not in source
    assert "/etc/machine-id" not in source
    assert "host = machine_digest()" in source
