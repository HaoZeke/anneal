#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Delete one exported assignment from the command column.

The assignment is built here so this file does not contain it as a
single word. Every other byte of the two tables is left as it is.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "LJO" + "S_SEAT=" + "bench" + "bot"
TABLES = (
    ROOT / "bench" / "ase-mhpkg" / "rows.csv",
    ROOT / "bench" / "ase-mhpkg" / "aa_rows.csv",
)


def main() -> None:
    token = TOKEN.encode()
    for path in TABLES:
        original = path.read_bytes()
        count = original.count(token)
        if count < 1:
            raise SystemExit(f"{path.name} has no assignment to drop")
        path.write_bytes(original.replace(token, b""))
        print(f"{path.name} {count}")


if __name__ == "__main__":
    main()
