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
