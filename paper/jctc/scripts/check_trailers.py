#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fail when a commit after integration v1.31 carries a trailer.

Usage: check_trailers.py [base]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "6c048e30869cdc5527886d726faa5c832bbb2881"


def trailers(base: str) -> list[str]:
    listed = subprocess.check_output(
        ["git", "log", "--format=%H%x1e%B%x1e", f"{base}..HEAD"],
        cwd=ROOT,
    )
    bad = []
    chunks = listed.split(b"\x1e")
    # triples of empty, hash, message because of the leading separator pattern
    i = 0
    while i + 1 < len(chunks):
        sha = chunks[i].strip()
        message = chunks[i + 1]
        i += 2
        if not sha:
            continue
        parsed = subprocess.run(
            ["git", "interpret-trailers", "--parse", "--no-divider"],
            input=message,
            capture_output=True,
            check=False,
        )
        if parsed.stdout.strip():
            bad.append(f"{sha.decode()}: {parsed.stdout.decode().strip()}")
    return bad


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    found = trailers(base)
    if found:
        print("\n".join(found), file=sys.stderr)
        return 1
    print("no trailers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
