#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Check each archived file against a git ls-tree listing.

Usage: lj75_source_verify.py ARCHIVE LISTING WORK
WORK is a directory used for the extract. The script writes nothing else.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


def blob_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", "--", str(path)], text=True
    ).strip()


TREE = "37535ec1a6e99d47a2afe0752072a33af64dccb3"


def listing_matches_tree(lines: list[str]) -> None:
    try:
        full = subprocess.check_output(
            ["git", "ls-tree", "-r", TREE],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return
    have = set(full.splitlines())
    missing = [line for line in lines if line not in have]
    if missing:
        raise SystemExit(f"listing line is absent from {TREE}: {missing[0]}")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: lj75_source_verify.py ARCHIVE LISTING WORK")
    archive, listing, work = map(Path, sys.argv[1:])
    lines = [line for line in listing.read_text().splitlines() if line]
    listing_matches_tree(lines)
    expected = {}
    for line in lines:
        mode, kind, sha, name = line.split(maxsplit=3)
        if kind != "blob":
            raise SystemExit(f"not a blob: {line}")
        expected[name] = sha
    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:xz") as packed:
        names = [item.name for item in packed.getmembers() if item.isfile()]
        packed.extractall(work)
    archived = set(names)
    if archived != set(expected):
        missing = sorted(set(expected) - archived)
        extra = sorted(archived - set(expected))
        raise SystemExit(f"member mismatch missing={missing[:5]} extra={extra[:5]}")
    for name, sha in expected.items():
        got = blob_sha(work / name)
        if got != sha:
            raise SystemExit(f"blob mismatch {name} {got} {sha}")
    print(f"verified {len(expected)} blobs")


if __name__ == "__main__":
    main()
