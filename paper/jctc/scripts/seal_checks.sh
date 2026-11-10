#!/bin/sh
# Copyright (c) anneal developers
# Check commit messages and, when a kit directory is given, the seal order.
# Usage: seal_checks.sh [KIT]
set -eu
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
sh "$here/check_message_newlines.sh"
if [ $# -ge 1 ]; then
  python3 - "$1" << 'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sums = (root / "SHA256SUMS").stat().st_mtime
outer = (root / "OUTER").stat().st_mtime
ready = (root / "STORE_READY").stat().st_mtime
if not (sums < outer < ready):
    raise SystemExit(f"seal order sums={sums} outer={outer} ready={ready}")
later = {"series.bundle", "clone.txt", "bundle-verify.txt", "fsck-strict.txt"}
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.name in {"OUTER", "STORE_READY"}:
        continue
    mtime = path.stat().st_mtime
    if path.name in later:
        if not (outer < mtime < ready):
            raise SystemExit(f"{path.name} is outside the bundle step")
        continue
    if mtime > sums:
        raise SystemExit(f"{path.name} is newer than SHA256SUMS")
print("seal order ok")
PY
fi
