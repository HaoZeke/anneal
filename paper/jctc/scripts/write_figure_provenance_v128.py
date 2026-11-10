#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the v1.28 figure record.

The record's tip hashes are the ones stored when the file was archived.
Ten of them are main's blobs. This script writes that blob and does not
recompute the hashes from today's figures.
"""
from __future__ import annotations

import subprocess
import sys

BLOB = "b3b5395211f02202c2f912e4475fa594d82c8cef"


def main() -> int:
    found = subprocess.run(
        ["git", "cat-file", "-t", BLOB],
        text=True,
        capture_output=True,
    )
    if found.returncode != 0 or found.stdout.strip() != "blob":
        raise SystemExit(f"missing blob {BLOB}")
    sys.stdout.buffer.write(subprocess.check_output(["git", "cat-file", "blob", BLOB]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
