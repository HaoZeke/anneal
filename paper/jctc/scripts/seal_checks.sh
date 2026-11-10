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
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.name in {"OUTER", "STORE_READY"}:
        continue
    if path.stat().st_mtime > sums:
        raise SystemExit(f"{path.name} is newer than SHA256SUMS")
print("seal order ok")
PY
fi
