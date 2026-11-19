#!/usr/bin/env bash
# Fail when the committed rgpot-core/include/rgpot.h differs from the header
# the crate regenerates with cbindgen.
# Entry points: pixi r header-drift   OR   bash scripts/check_header_drift.sh
# Regenerates with `pixi r gen-header` (gen-header only) and again with every
# cargo feature, so a constant behind a feature cannot leave the header stale.
# The committed header is restored afterwards; fix a failure with
# `pixi r gen-header` and commit the result.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

header=rgpot-core/include/rgpot.h
committed="$(mktemp)"
trap 'cp "$committed" "$header"; rm -f "$committed"' EXIT
cp "$header" "$committed"

status=0
for features in "--features gen-header" "--all-features"; do
  # build.rs reruns when this variable changes, so each pass regenerates.
  # shellcheck disable=SC2086
  RGPOT_SOURCE_REVISION="header-drift-$$-${features//[^a-z]/}" \
    cargo build --manifest-path rgpot-core/Cargo.toml $features >&2
  if ! diff -u "$committed" "$header" >&2; then
    echo "error: $header differs from the header cbindgen regenerates ($features)" >&2
    status=1
  fi
  cp "$committed" "$header"
done

if [ "$status" -ne 0 ]; then
  echo "fix: pixi r gen-header, review the diff and commit $header" >&2
  exit 1
fi
echo "OK: $header matches the regenerated header"
