# Copyright (c) anneal developers
"""The GMIN panel build refuses a tarball whose sha256 is not the pin."""

from __future__ import annotations

import os
import re
import subprocess
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


def test_the_runner_pins_the_same_dated_tarball_as_the_build() -> None:
    script = BUILD.read_text()
    panel = PANEL.read_text()
    date = _assign(script, "TARBALL_DATE")
    digest = _assign(script, "TARBALL_SHA256")
    assert date == "2026-11-19"
    assert f'GMIN_TARBALL_DATE = "{date}"' in panel
    assert f'GMIN_TARBALL_SHA256 = "{digest}"' in panel
    assert "SVNREV" not in script
    assert "\nREV=" not in script
    assert 'GMIN_REV = "aac1548a8aec13b53ad12a8d6904625081685d2e"' in panel


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
