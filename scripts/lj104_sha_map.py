#!/usr/bin/env python3
"""Rewrite logs/sha-map.tsv from the kit files, then check every row.

A row whose two digests are equal is refreshed from the file.
A row whose digests differ keeps the earlier raw digest.
The check fails unless each redacted digest matches the file,
the SUMS line, and the tip blob when the path is in the tip.
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


def tip_digest(repo: Path, name: str):
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{name}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        return None
    show = subprocess.run(
        ["git", "show", f"HEAD:{name}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return hashlib.sha256(show.stdout).hexdigest()


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
    if len(sys.argv) != 3:
        print("usage: lj104_sha_map.py <kit> <repo>", file=sys.stderr)
        return 2
    kit = Path(sys.argv[1])
    repo = Path(sys.argv[2])
    refresh(kit)
    return check(kit, repo)


if __name__ == "__main__":
    sys.exit(main())
