#!/usr/bin/env python3
"""Fail when a version copy drifts, including a bundle lock with no markers.

cogapp ``--check`` compares the text between ``[[[cog`` and ``[[[end`` with
what the generator would write. A lock that has lost those two lines has
nothing to compare, so ``--check`` succeeds and the next release leaves
``rgpot-core`` at the old version. ``cargo`` without ``--locked`` on
``rgpot-bundle`` rewrites the lock and drops the lines.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_LOCK = ROOT / "rgpot-bundle" / "Cargo.lock"
COG_BEGIN = (
    "# [[[cog import rgpot_version as v; "
    "cog.outl(f'version = \"{v.VERSION}\"')]]]"
)
COG_END = "# [[[end]]]"


def require_bundle_markers(text: str) -> None:
    """The rgpot-core package entry keeps its cog markers around version."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line != COG_BEGIN or index + 2 >= len(lines):
            continue
        version = lines[index + 1]
        if version.startswith('version = "') and lines[index + 2] == COG_END:
            return
    raise SystemExit(
        "rgpot-bundle/Cargo.lock is missing the cog markers around the "
        "rgpot-core version; cargo without --locked strips them"
    )


def main() -> None:
    listing = ROOT / "ci" / "version" / "files.txt"
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
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    require_bundle_markers(BUNDLE_LOCK.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
