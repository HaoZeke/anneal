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
bundle = root / "series.bundle"
log_names = ("bundle-verify.txt", "clone.txt", "fsck-strict.txt")
logs = [root / "logs" / name for name in log_names]
sums = root / "SHA256SUMS"
outer = root / "OUTER"
ready = root / "STORE_READY"
named = [bundle, *logs, sums, outer, ready]
for path in named:
    if not path.is_file():
        raise SystemExit(f"missing {path.name}")
named_ids = {path.resolve() for path in named}
content = [
    path for path in root.rglob("*")
    if path.is_file() and path.resolve() not in named_ids
]
if not content:
    raise SystemExit("no content files")

def sec(path):
    return int(path.stat().st_mtime)

content_sec = max(sec(path) for path in content)
bundle_sec = sec(bundle)
log_secs = [sec(path) for path in logs]
sums_sec = sec(sums)
outer_sec = sec(outer)
ready_sec = sec(ready)
order = (
    content_sec < bundle_sec < min(log_secs)
    and max(log_secs) < sums_sec < outer_sec < ready_sec
)
if not order:
    raise SystemExit(
        "seal order "
        f"content={content_sec} bundle={bundle_sec} "
        f"logs={log_secs} sums={sums_sec} "
        f"outer={outer_sec} ready={ready_sec}"
    )
print("seal order ok")
PY
fi
