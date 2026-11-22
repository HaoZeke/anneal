#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One verified source pin for every GMIN build and benchmark row."""

from __future__ import annotations

import argparse
import hashlib
import shlex
from pathlib import Path

SNAPSHOT = "2026-11-19"
URL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
REVISION = "9cbec1a446352bc9ef064bd2b1a5767194655919"
TARBALL_SHA256 = "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776"
# SVNREV of the provisional rows already committed. No builder uses it.
HISTORICAL_REVISION = "aac1548a8aec13b53ad12a8d6904625081685d2e"
# Every SVNREV a stored GMIN row may carry.
KNOWN_REVISIONS = (HISTORICAL_REVISION, REVISION)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tarball(path: Path) -> str:
    """Refuse any tarball other than the pinned snapshot."""
    actual = sha256(path)
    if actual != TARBALL_SHA256:
        raise ValueError(
            f"tarball sha256 mismatch: got {actual}, expected {TARBALL_SHA256}"
        )
    return actual


def read_svnrev(path: Path) -> str:
    """Read and verify the revision embedded in a built source tree."""
    actual = path.read_text().strip()
    if actual != REVISION:
        raise ValueError(f"GMIN SVNREV is {actual}, expected {REVISION}")
    return actual


def read_built_revision(svnrev: Path, metadata: Path | None = None) -> str:
    """Read a built SVNREV and optionally verify its generated source note."""
    revision = read_svnrev(svnrev)
    if metadata is not None:
        source = dict(
            line.split("=", 1)
            for line in metadata.read_text().splitlines()
            if "=" in line
        )
        if source.get("svnrev") != revision:
            raise ValueError(f"{metadata} does not describe {svnrev}")
        if source.get("tarball_sha256") != TARBALL_SHA256:
            raise ValueError(f"{metadata} has another GMIN tarball")
    return revision


def shell_assignments() -> str:
    """Return shell-safe assignments consumed by build scripts."""
    values = {
        "GMIN_PIN_SNAPSHOT": SNAPSHOT,
        "GMIN_PIN_URL": URL,
        "GMIN_PIN_REVISION": REVISION,
        "GMIN_PIN_TARBALL_SHA256": TARBALL_SHA256,
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("shell")
    tarball = sub.add_parser("verify-tarball")
    tarball.add_argument("path", type=Path)
    svnrev = sub.add_parser("read-svnrev")
    svnrev.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "shell":
            print(shell_assignments())
        elif args.command == "verify-tarball":
            print(verify_tarball(args.path))
        else:
            print(read_svnrev(args.path))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
