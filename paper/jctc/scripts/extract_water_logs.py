#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Extract the redacted water logs and check them against the sha map.

Usage: extract_water_logs.py ARCHIVE MAP DEST
MAP is the member table with columns name, old sha, new sha.
A missing archive is an error. The extracted bytes must match the new sha.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import fetch_paper_data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: extract_water_logs.py ARCHIVE MAP DEST")
    archive, map_path, dest = map(Path, sys.argv[1:])
    fetch_paper_data()
    if not archive.is_file():
        raise SystemExit(f"missing archive {archive}")
    expected = {}
    for line in map_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] in {"member", "original_sha256", "new_sha256"}:
            continue
        expected[parts[0]] = parts[2]
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            data = packed.extractfile(member).read()
            got = sha256(data)
            want = expected.get(member.name)
            if want is None or got != want:
                raise SystemExit(f"sha mismatch {member.name}")
            out = dest / member.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    print(f"extracted {len(expected)} members")


if __name__ == "__main__":
    main()
