#!/usr/bin/env python3
# Copyright (c) anneal developers
# SPDX-License-Identifier: MIT
"""Check logs/sha-map.tsv, or rewrite it when asked.

--check reads the existing map and does not write.
It compares every row with the file, the SHA256SUMS line, and the tip blob.
A mismatch is named and the process exits non-zero.
--refresh is the only mode that rewrites the map.
A git error is fatal. A path that is absent from the tip is not a git error.
"""

import hashlib
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_count(path: Path) -> int:
    return path.read_bytes().count(b"\n")


def read_map(path: Path):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        raw, red, raw_n, red_n, name = line.split("\t")
        rows.append([raw, red, raw_n, red_n, name])
    return header, rows


def write_map(path: Path, header, rows) -> None:
    body = ["\t".join(header)]
    for row in rows:
        body.append("\t".join(row))
    path.write_text("\n".join(body) + "\n")


def read_sums(path: Path):
    found = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        value, name = line.split(None, 1)
        if name.startswith("./"):
            name = name[2:]
        found[name] = value
    return found


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = ["git", "-c", f"safe.directory={repo.resolve()}", *args]
    return subprocess.run(
        cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_fatal(result: subprocess.CompletedProcess, what: str) -> None:
    if result.returncode == 0:
        return
    sys.stderr.buffer.write(result.stderr)
    print(f"git {what} failed", file=sys.stderr)
    print(
        "If the repo is reached through a store mount, "
        "run git with -c safe.directory=<repo> for that call.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def require_head(repo: Path) -> None:
    git_fatal(git(repo, "rev-parse", "--verify", "HEAD"), "rev-parse")


def tip_digest(repo: Path, name: str):
    listed = git(repo, "ls-tree", "HEAD", "--", name)
    git_fatal(listed, "ls-tree")
    if not listed.stdout.strip():
        return None
    shown = git(repo, "show", f"HEAD:{name}")
    git_fatal(shown, "show")
    return hashlib.sha256(shown.stdout).hexdigest()


def refresh(kit: Path) -> None:
    map_path = kit / "logs" / "sha-map.tsv"
    header, rows = read_map(map_path)
    for row in rows:
        name = row[4]
        file_path = kit / name
        file_hash = digest(file_path)
        file_lines = str(line_count(file_path))
        if row[0] == row[1]:
            row[0] = file_hash
            row[2] = file_lines
        row[1] = file_hash
        row[3] = file_lines
    write_map(map_path, header, rows)


def check(kit: Path, repo: Path) -> int:
    require_head(repo)
    map_path = kit / "logs" / "sha-map.tsv"
    sums = read_sums(kit / "SHA256SUMS")
    _header, rows = read_map(map_path)
    failures = 0
    for _raw, red, _raw_n, red_n, name in rows:
        file_path = kit / name
        file_hash = digest(file_path)
        file_lines = str(line_count(file_path))
        if file_hash != red or file_lines != red_n:
            print(f"file mismatch {name}", file=sys.stderr)
            failures += 1
        if sums.get(name) != red:
            print(f"sums mismatch {name}", file=sys.stderr)
            failures += 1
        blob = tip_digest(repo, name)
        if blob is not None and blob != red:
            print(f"tip mismatch {name}", file=sys.stderr)
            failures += 1
    if failures:
        print(f"sha-map check failed: {failures}", file=sys.stderr)
        return 1
    print(f"sha-map check ok: {len(rows)}")
    return 0


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("--check", "--refresh"):
        print("usage: lj104_sha_map.py --check <kit> <repo>", file=sys.stderr)
        print("       lj104_sha_map.py --refresh <kit> <repo>", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    kit = Path(sys.argv[2])
    repo = Path(sys.argv[3])
    if mode == "--check":
        return check(kit, repo)
    refresh(kit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
