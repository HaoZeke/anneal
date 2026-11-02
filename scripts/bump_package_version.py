#!/usr/bin/env python3
"""Set the workspace package version and restamp the generated copies.

Cocogitto calls this from ``pre_bump_hooks`` with the new triple.
https://docs.cocogitto.io/
https://nedbatchelder.com/code/cog/
https://towncrier.readthedocs.io/en/stable/
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIPLE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def main() -> int:
    if len(sys.argv) != 2 or not TRIPLE.fullmatch(sys.argv[1]):
        raise SystemExit("usage: bump_package_version.py X.Y.Z")
    version = sys.argv[1]
    cargo = ROOT / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^version = "[0-9]+\.[0-9]+\.[0-9]+"',
        'version = "%s"' % version,
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("Cargo.toml package version line was not found")
    # The first column-0 `version = "` line is [package].version.
    # The [workspace] table does not set a version, so this replacement
    # and the CMake LIMIT_COUNT 1 read stay on that line.
    cargo.write_text(updated, encoding="utf-8")
    # https://doc.rust-lang.org/cargo/commands/cargo-update.html
    subprocess.run(
        ["cargo", "update", "--workspace", "--offline"],
        cwd=ROOT,
        check=True,
    )
    cog = [sys.executable, str(ROOT / "scripts" / "version_cog.py")]
    subprocess.run([*cog, "-r", "@scripts/cogfiles"], cwd=ROOT, check=True)
    subprocess.run(
        [*cog, "-d", "-o", ".zenodo.json", "packaging/zenodo.cog"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "towncrier", "build", "--yes"],
        cwd=ROOT,
        check=True,
    )
    # Drift checks run here, before cocogitto records the bump commit.
    # https://docs.cocogitto.io/
    subprocess.run([*cog, "--check", "@scripts/cogfiles"], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_version_sot.py")],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
