#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Recompute the copper log blob ids at the source commit.

The expected blob and sha256 of each log are the lines in PROVENANCE.
This checks those bytes, then the blob of the same path at
311d9bcc28b5401780a69898a16fff25707901ad.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

COMMIT = "311d9bcc28b5401780a69898a16fff25707901ad"
PREFIX = "rewrite_2026/data/hq_sci_cuh2"
URL = "https://github.com/HaoZeke/simmAnnealTeX.git"


def records(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    name = blob = ""
    for raw in path.read_text().splitlines():
        line = raw[:-1] if raw.endswith(".") else raw
        key, _, value = line.partition(" ")
        if key == "Log":
            name, blob = value, ""
        elif key == "Blob":
            blob = value
        elif key == "Hash":
            rows.append((name, blob, value))
    if len(rows) != 8:
        raise SystemExit(f"PROVENANCE lists {len(rows)} logs")
    return rows


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
    )


def fetch(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    fetched = git(repo, "fetch", "--depth", "1", URL, COMMIT)
    if fetched.returncode != 0:
        detail = (fetched.stderr or fetched.stdout).strip()
        raise SystemExit(f"commit {COMMIT} is not available\n{detail}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path("paper/jctc/data/hq_sci_cuh2")
    repo = Path(args[1]) if len(args) > 1 else None
    for name, blob, sha in records(root / "PROVENANCE.txt"):
        data = (root / name).read_bytes()
        got_sha = hashlib.sha256(data).hexdigest()
        if got_sha != sha:
            raise SystemExit(f"{name} sha256 {got_sha}")
        hashed = subprocess.run(
            ["git", "hash-object", str(root / name)],
            text=True,
            capture_output=True,
            check=True,
        )
        if hashed.stdout.strip() != blob:
            raise SystemExit(f"{name} blob {hashed.stdout.strip()}")
    if repo is None:
        with tempfile.TemporaryDirectory() as tmp:
            fetched = Path(tmp)
            fetch(fetched)
            return compare(fetched, root)
    return compare(repo, root)


def compare(repo: Path, root: Path) -> int:
    for name, blob, _sha in records(root / "PROVENANCE.txt"):
        shown = git(repo, "rev-parse", f"{COMMIT}:{PREFIX}/{name}")
        if shown.returncode != 0:
            detail = (shown.stderr or shown.stdout).strip()
            raise SystemExit(f"{name} at {COMMIT}\n{detail}")
        if shown.stdout.strip() != blob:
            raise SystemExit(f"{name} commit blob {shown.stdout.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
