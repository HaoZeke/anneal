# Copyright 2023--present rgpot developers
"""The version-drift check notices a bundle lock that lost its cog markers.

cogapp --check exits 0 when the markers are absent. Cargo without --locked
on rgpot-bundle rewrites the lock and removes them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci" / "version"))
import check_drift

LOCK = ROOT / "rgpot-bundle" / "Cargo.lock"


def _strip(text: str) -> str:
    kept = [
        line
        for line in text.splitlines()
        if line not in (check_drift.COG_BEGIN, check_drift.COG_END)
    ]
    return "\n".join(kept) + "\n"


def test_real_lock_keeps_the_markers():
    check_drift.require_bundle_markers(LOCK.read_text(encoding="utf-8"))


def test_missing_markers_fail():
    text = LOCK.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="missing the cog markers"):
        check_drift.require_bundle_markers(_strip(text))


def test_a_marker_away_from_the_version_line_fails():
    text = LOCK.read_text(encoding="utf-8")
    moved = text.replace(check_drift.COG_END + "\n", "", 1)
    moved += "\n" + check_drift.COG_END + "\n"
    with pytest.raises(SystemExit, match="missing the cog markers"):
        check_drift.require_bundle_markers(moved)


def test_cog_check_misses_a_stripped_lock(tmp_path):
    pytest.importorskip("cogapp")
    lock = tmp_path / "Cargo.lock"
    lock.write_text(_strip(LOCK.read_text(encoding="utf-8")), encoding="utf-8")
    listing = tmp_path / "files.txt"
    listing.write_text(f"{lock}\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cogapp",
            "-I",
            str(ROOT / "ci" / "version"),
            "--check",
            f"@{listing}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_version_drift_accepts_the_tree():
    pytest.importorskip("cogapp")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "ci" / "version" / "check_drift.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
