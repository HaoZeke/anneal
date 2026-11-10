#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Hash a GMIN_BEST executable and compare it to the registered digest.

The executable is not stored in this tree. Pass a relative path.
The script writes nothing except its report on stdout.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

DIGEST = (
    "b8eeddfa6d18a697a76d0714c0a6b356c98c1d40022bd639ac55581e119baf95"
)


def main() -> None:
    print("expected digest follows")
    print(DIGEST)
    if len(sys.argv) != 2:
        print("usage: verify_gmin_best.py RELATIVE_PATH")
        raise SystemExit(2)
    rel = Path(sys.argv[1])
    if rel.is_absolute() or ".." in rel.parts:
        print("path must be relative and must not contain ..")
        raise SystemExit(2)
    if not rel.is_file():
        print("executable bytes are absent")
        raise SystemExit(2)
    digest = hashlib.sha256(rel.read_bytes()).hexdigest()
    print(f"sha256 {digest}")
    if digest != DIGEST:
        print("mismatch")
        raise SystemExit(1)
    print("match")


if __name__ == "__main__":
    main()
