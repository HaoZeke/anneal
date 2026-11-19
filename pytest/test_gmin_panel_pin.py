# Copyright (c) anneal developers
"""Every GMIN build and row uses one verified source pin."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "gmin_panel" / "build_gmin.sh"
sys.path.insert(0, str(ROOT / "scripts"))
from gmin_pin import REVISION, TARBALL_SHA256, read_svnrev
from morse_gmin.run_panel import gmin_line


def test_every_gmin_builder_loads_the_shared_pin() -> None:
    builds = tuple((ROOT / "scripts").glob("**/build_gmin*.sh"))
    assert len(builds) == 6
    for build in builds:
        text = build.read_text()
        assert "gmin_pin.py" in text, build
        assert (
            "verify-tarball" in text
            or "build_gmin.sh" in text
            or "build_gmin_darby.sh" in text
        ), build


def test_fake_built_svnrev_flows_into_code_revision(tmp_path: Path) -> None:
    svnrev = tmp_path / "SVNREV"
    svnrev.write_text(f"{REVISION}\n")
    revision = read_svnrev(svnrev)
    row = gmin_line(
        "h2h_row best=-1 wall_s=1 pot_s=0.5 charged=10 evals_to_hit=10",
        "m30r6",
        2000,
        400_000,
        -1.0,
        revision,
    )
    assert row["code_revision"] == REVISION


def test_the_build_normalizes_paths_and_stamps() -> None:
    script = BUILD.read_text()
    assert 'WORK=$(cd "$WORK" && pwd -P)' in script
    assert "SRC=$WORK/gmin-src" in script
    assert "TARBALL=$WORK/svn.tar.bz2" in script
    assert '-ffile-prefix-map=$SRC=/usr/src/gmin' in script
    assert '-fdebug-prefix-map=$SRC=/usr/src/gmin' in script
    assert '-ffile-prefix-map=$WORK=/build/gmin' in script
    assert '-fdebug-prefix-map=$WORK=/build/gmin' in script
    assert 'realpath --relative-to="$PWD"' in script
    assert "SOURCE_DATE_EPOCH=1795057278" in script
    assert "LC_ALL=C exec /usr/bin/date" in script
    assert "export LC_ALL=C" in script
    assert "export USER=builder" in script
    assert "export HOSTNAME=gmin-panel" in script
    assert "-o) echo linux" in script
    assert "-m) echo x86_64" in script


def test_the_build_uses_file_identity_and_cleans_partial_downloads() -> None:
    script = BUILD.read_text()
    assert 'test "$LOCAL" -ef "$TARBALL"' in script
    assert "trap cleanup_partial EXIT" in script


def _mock_fetch(tmp_path: Path, fail_hf: bool = False) -> subprocess.CompletedProcess[str]:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    curl = mock_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MOCK_CURL_LOG"
if [[ ${MOCK_HF_FAIL:-0} == 1 && $* == *huggingface.co* ]]; then
  exit 22
fi
while [[ $# -gt 0 ]]; do
  if [[ $1 == -o ]]; then
    printf 'wrong archive\n' > "$2"
    exit 0
  fi
  shift
done
exit 2
"""
    )
    curl.chmod(0o755)
    work = tmp_path / "work"
    log = tmp_path / "curl.log"
    env = os.environ.copy()
    env["PATH"] = f"{mock_bin}:{env['PATH']}"
    env["MOCK_CURL_LOG"] = str(log)
    env["MOCK_HF_FAIL"] = "1" if fail_hf else "0"
    env["GMIN_PANEL_WORK"] = str(work)
    proc = subprocess.run(
        ["bash", str(BUILD)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    proc.curl_log = log.read_text()  # type: ignore[attr-defined]
    assert not list(work.glob("svn.tar.bz2.partial.*"))
    return proc


def test_fetch_uses_the_pinned_dataset_and_refuses_its_wrong_hash(tmp_path: Path) -> None:
    proc = _mock_fetch(tmp_path)
    assert proc.returncode == 1
    assert (
        "https://huggingface.co/datasets/HaoZeke/anneal-paper-data/resolve/"
        "2d639566b6e9a7243bfd485aa3cb9a904c915451/gmin/svn-20261119.tar.bz2"
        in proc.curl_log  # type: ignore[attr-defined]
    )
    assert "www-wales.ch.cam.ac.uk" not in proc.curl_log  # type: ignore[attr-defined]
    assert "tarball sha256 mismatch:" in proc.stderr


def test_fetch_falls_back_to_wales_after_dataset_failure(tmp_path: Path) -> None:
    proc = _mock_fetch(tmp_path, fail_hf=True)
    assert proc.returncode == 1
    assert "pinned dataset download failed" in proc.stderr
    assert "www-wales.ch.cam.ac.uk/svn.tar.bz2" in proc.curl_log  # type: ignore[attr-defined]
    assert "tarball sha256 mismatch:" in proc.stderr


@pytest.mark.parametrize("how", ["env", "flag"])
def test_a_local_tarball_with_the_wrong_sha256_is_refused(tmp_path: Path, how: str) -> None:
    bad = tmp_path / "svn.tar.bz2"
    bad.write_bytes(b"not the pinned snapshot\n")
    work = tmp_path / "work"
    env = os.environ.copy()
    env["GMIN_PANEL_WORK"] = str(work)
    env.pop("GMIN_TARBALL", None)
    command = ["bash", str(BUILD)]
    if how == "env":
        env["GMIN_TARBALL"] = str(bad)
    else:
        command += ["--tarball", str(bad)]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    assert "tarball sha256 mismatch:" in proc.stderr
    assert TARBALL_SHA256 in proc.stderr
    assert not work.exists()
