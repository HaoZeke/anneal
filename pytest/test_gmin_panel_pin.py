# Copyright (c) anneal developers
"""The GMIN panel build refuses a tarball whose sha256 is not the pin."""

from __future__ import annotations

import os
import re
import subprocess
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "gmin_panel" / "build_gmin.sh"
PANEL = ROOT / "scripts" / "gmin_panel" / "run_panel.py"


def _assign(text: str, name: str) -> str:
    match = re.search(rf"^{name}=(\S+)$", text, re.M)
    if match is None:
        raise AssertionError(f"{name} is not set")
    return match.group(1)


def test_the_runner_pins_the_same_snapshot_as_the_build() -> None:
    script = BUILD.read_text()
    panel = PANEL.read_text()
    revision = _assign(script, "REV")
    digest = _assign(script, "TARBALL_SHA256")
    assert _assign(script, "SNAPSHOT") == "2026-11-19"
    assert f'GMIN_PIN = "{revision}"' in panel
    assert f'GMIN_TARBALL_SHA256 = "{digest}"' in panel
    assert 'GMIN_REV = "aac1548a8aec13b53ad12a8d6904625081685d2e"' in panel


def test_the_build_is_path_and_stamp_reproducible() -> None:
    script = BUILD.read_text()
    assert '-ffile-prefix-map=$SRC=/usr/src/gmin' in script
    assert '-fdebug-prefix-map=$SRC=/usr/src/gmin' in script
    assert '-ffile-prefix-map=$WORK=/build/gmin' in script
    assert '-fdebug-prefix-map=$WORK=/build/gmin' in script
    assert 'realpath --relative-to="$PWD"' in script
    assert "SOURCE_DATE_EPOCH=1795057278" in script
    assert "export USER=builder" in script
    assert "export HOSTNAME=reproducible" in script
    assert "reproducible-os" in script
    assert "reproducible-arch" in script


def test_the_build_uses_file_identity_and_cleans_partial_downloads() -> None:
    script = BUILD.read_text()
    assert 'test "$LOCAL" -ef "$TARBALL"' in script
    assert "trap cleanup_partial EXIT" in script


def test_summaries_select_the_pinned_gmin_revision() -> None:
    spec = importlib.util.spec_from_file_location("gmin_panel_pin_test", PANEL)
    assert spec is not None and spec.loader is not None
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    rows = [
        {
            "system": "au55",
            "code": "gmin",
            "seed": "2000",
            "code_revision": panel.GMIN_REV,
            "harness_revision": "anneal-rev",
        },
        {
            "system": "au55",
            "code": "gmin",
            "seed": "2000",
            "code_revision": panel.GMIN_PIN,
            "harness_revision": "anneal-rev",
        },
    ]
    selected = panel.panel_groups(rows, "anneal-rev")
    assert selected[("au55", "gmin")][0]["code_revision"] == panel.GMIN_PIN
    selected = panel.panel_groups(rows, "latest")
    assert selected[("au55", "gmin")][0]["code_revision"] == panel.GMIN_PIN


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
    assert "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776" in proc.stderr
    assert not work.exists()
