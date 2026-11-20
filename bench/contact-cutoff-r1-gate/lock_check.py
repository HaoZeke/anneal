#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fail unless every package the harness resolved is pinned in the arm's lock.

    lock_check.py <arm Cargo.lock> <harness Cargo.lock>

The harness package itself is the one entry the arm's lock cannot carry.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

HARNESS = "contact-cutoff-gate-harness"


def packages(path: Path) -> set[tuple[str, str, str]]:
    data = tomllib.loads(path.read_text())
    return {(p["name"], p["version"], p.get("source", "")) for p in data["package"]}


def main(argv: list[str]) -> int:
    arm, harness = (Path(a) for a in argv)
    extra = sorted(p for p in packages(harness) - packages(arm) if p[0] != HARNESS)
    if extra:
        for name, version, source in extra:
            print(f"not in the arm lock: {name} {version} {source}", file=sys.stderr)
        return 1
    print(f"lock ok: {len(packages(harness)) - 1} packages, all pinned by the arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
