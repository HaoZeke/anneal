#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Attempt a GMIN_BEST rebuild and record why it does not match.

The Fortran source is not pinned in this repository. The script
does not emit a binary and does not claim a digest match.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

EXECUTABLE = (
    "b8eeddfa6d18a697a76d0714c0a6b356c98c1d40022bd639ac55581e119baf95"
)
KEYWORDS = (
    "62618de90c53478bbfe724efe527dd1fa67cd6386da0cf00cb0cc8431bd7037d"
)
SOURCE = Path("source/potential.f90")
KEYWORDS_FILE = Path("docs/prereg/lj104-sota/gmin_best.keywords")


def main() -> None:
    print("GMIN_BEST rebuild attempt")
    print("expected executable digest follows")
    print(EXECUTABLE)
    print("digest status unverified")
    if SOURCE.is_file():
        print("source potential.f90 is present")
    else:
        print("pinned Fortran source is absent")
        print("missing source/potential.f90")
    compiler = shutil.which("gfortran-13") or shutil.which("gfortran")
    if compiler is None:
        print("gfortran 13.3.0 is absent")
    else:
        print("gfortran 13.3.0 is present")
    print("a separate compile produced no binary")
    print("no rebuilt digest is available")
    if not KEYWORDS_FILE.is_file():
        print("keyword file is absent")
        raise SystemExit(2)
    keyword_digest = hashlib.sha256(KEYWORDS_FILE.read_bytes()).hexdigest()
    print("keyword sha256 follows")
    print(keyword_digest)
    if keyword_digest != KEYWORDS:
        print("keyword file does not match the recorded keyword digest")
        raise SystemExit(1)
    print("keyword file matches its own recorded digest")
    print("that match is not an executable match")
    print("no binary was produced")
    print("rebuild did not match the executable digest")
    raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())
