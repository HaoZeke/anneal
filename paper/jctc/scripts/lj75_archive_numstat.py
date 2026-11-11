#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Diff the LJ75 source archive against commit da65cc6d.

That commit is the one whose tree matches the insertion and deletion
counts printed in the Supporting Information. The script reads the
archive and `git show` of that commit. It does not type the counts.
"""
from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "paper/jctc/logs/lj75-source-37535ec1.tar.xz"
COMMIT = "da65cc6d"
FILES = (
    "src/methods/cluster_hopping.rs",
    "src/potentials/tip4p.rs",
    "src/rigid_body.rs",
    "examples/molecular_cluster.rs",
    "Cargo.toml",
    "Cargo.lock",
)


def counts() -> list[tuple[str, int, int]]:
    rows = []
    with tarfile.open(ARCHIVE, "r:xz") as tar, tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        for name in FILES:
            member = tar.extractfile(name)
            if member is None:
                raise SystemExit(f"archive lacks {name}")
            archived = dest / "old"
            archived.write_bytes(member.read())
            shown = subprocess.check_output(
                ["git", "show", f"{COMMIT}:{name}"], cwd=ROOT
            )
            current = dest / "new"
            current.write_bytes(shown)
            diff = subprocess.run(
                ["git", "diff", "--no-index", "--numstat", str(archived), str(current)],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            if diff.returncode not in (0, 1):
                raise SystemExit(diff.stderr.decode())
            added, deleted, _path = diff.stdout.decode().split("\t", 2)
            rows.append((name, int(added), int(deleted)))
    return rows


def main() -> int:
    for name, added, deleted in counts():
        print(f"{name} {added} {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
