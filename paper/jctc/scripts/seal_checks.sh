#!/bin/sh
# Copyright (c) anneal developers
# Check commit messages and, when a kit directory is given, the seal.
# Usage: seal_checks.sh [KIT]
set -eu
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
sh "$here/check_message_newlines.sh"
if [ $# -ge 1 ]; then
  python3 - "$1" << 'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
bundle = root / "series.bundle"
sums = root / "SHA256SUMS"
outer = root / "OUTER"
ready = root / "STORE_READY"
named = [bundle, sums, outer, ready]
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

def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hashed.update(chunk)
    return hashed.hexdigest()

content_sec = max(sec(path) for path in content)
bundle_sec = sec(bundle)
sums_sec = sec(sums)
outer_sec = sec(outer)
ready_sec = sec(ready)
order = content_sec < bundle_sec < sums_sec < outer_sec < ready_sec
if not order:
    raise SystemExit(
        "seal order "
        f"content={content_sec} bundle={bundle_sec} "
        f"sums={sums_sec} outer={outer_sec} ready={ready_sec}"
    )

listed = {}
for line in sums.read_text().splitlines():
    if not line.strip():
        continue
    digest_text, sep, rel = line.partition("  ")
    if sep != "  " or len(digest_text) != 64:
        raise SystemExit(f"bad SHA256SUMS line {line}")
    listed[rel] = digest_text
covered = {path.relative_to(root).as_posix() for path in content}
covered.add("series.bundle")
missing = sorted(covered - set(listed))
if missing:
    raise SystemExit(f"SHA256SUMS omits {missing[0]}")
extra = sorted(set(listed) - covered)
if extra:
    raise SystemExit(f"SHA256SUMS lists {extra[0]}")
for rel, expect in listed.items():
    got = digest(root / rel)
    if got != expect:
        raise SystemExit(f"SHA256SUMS mismatch {rel}")

bundle_sha = digest(bundle)
sums_sha = digest(sums)
outer_text = outer.read_text().strip()
if outer_text == bundle_sha:
    raise SystemExit("OUTER equals the bundle sha")
if outer_text != sums_sha:
    raise SystemExit("OUTER is not sha256 of SHA256SUMS")

fields = {}
for line in ready.read_text().splitlines():
    key, sep, value = line.partition(" ")
    if sep != " ":
        raise SystemExit(f"bad STORE_READY line {line}")
    fields[key] = value
for key in ("OUTER", "TIP", "TREE", "BUNDLE_SHA256"):
    if key not in fields:
        raise SystemExit(f"STORE_READY lacks {key}")
if fields["OUTER"] != sums_sha:
    raise SystemExit("STORE_READY OUTER is not sha256 of SHA256SUMS")
if fields["BUNDLE_SHA256"] != bundle_sha:
    raise SystemExit("STORE_READY BUNDLE_SHA256 is not the bundle sha")
if len(fields["TIP"]) != 40 or len(fields["TREE"]) != 40:
    raise SystemExit("STORE_READY tip or tree is not a sha")
print("seal order ok")
PY
fi
