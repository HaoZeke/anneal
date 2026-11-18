#!/usr/bin/env bash
# Fail when rgpot-core/include/rgpot.h is not what cbindgen generates from
# the Rust source.
#
# cbindgen copies doc comments into the header, so editing a doc comment on
# an extern "C" function changes it too. Build.rs rewrites the tracked file
# whenever the gen-header feature is on, so a build alone never fails: this
# script builds, then fails if git sees the rewrite.
# Fix: bash tools/check_header_drift.sh, then commit the header it wrote.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

header=rgpot-core/include/rgpot.h

cargo build --locked -p rgpot-core --features gen-header
if ! git diff --exit-code -- "$header"; then
  echo "error: $header is out of date with the Rust source (diff above)." >&2
  echo "fix: pixi run gen-header, then commit $header" >&2
  exit 1
fi
echo "OK: $header matches the cbindgen output"
