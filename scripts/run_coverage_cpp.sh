#!/usr/bin/env bash
# Instrument CppCore (C++ and the Fortran kernels) via Meson b_coverage + lcov.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-cpp_lcov.info}"
BUILD="${RGPOT_COV_BUILDDIR:-bbdir-cov}"

if ! command -v meson >/dev/null 2>&1; then
  echo "ERROR: meson not on PATH (use pixi -e devbld)" >&2
  exit 1
fi
if ! command -v lcov >/dev/null 2>&1; then
  echo "ERROR: lcov not on PATH" >&2
  exit 1
fi

echo "==> meson setup ${BUILD} (coverage, tests, rpc)"
rm -rf "$BUILD"
meson setup "$BUILD" \
  -Dwith_tests=true \
  -Dwith_examples=false \
  -Dwith_xtensor=false \
  -Dwith_eigen=false \
  -Dwith_rpc=true \
  -Dwith_cache=false \
  -Dpure_lib=false \
  -Db_coverage=true \
  --buildtype=debug

meson compile -C "$BUILD"
meson test -C "$BUILD" --print-errorlogs \
  --timeout-multiplier "${RGPOT_COV_TIMEOUT_MULTIPLIER:-1}"

# lcov 2 renamed the branch-coverage rc option and rejects the old name. It
# also checks gcov data against source lines and reports template instances in
# toolchain headers as inconsistent; that one class is tolerated, every other
# lcov error still fails the script.
LCOV_MAJOR=$(lcov --version | sed -n 's/.*LCOV version \([0-9][0-9]*\)\..*/\1/p')
LCOV_ARGS=()
if [[ "${LCOV_MAJOR:-0}" -ge 2 ]]; then
  LCOV_ARGS+=(--rc branch_coverage=1 --ignore-errors inconsistent)
else
  LCOV_ARGS+=(--rc lcov_branch_coverage=1)
fi

echo "==> lcov capture"
lcov --directory "$BUILD" --capture --output-file "$OUT" "${LCOV_ARGS[@]}"

# Keep library sources: CppCore/rgpot (C++ and Fortran kernels)
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
lcov --extract "$OUT" \
  '*/CppCore/rgpot/*' '*/include/*' \
  --output-file "$TMP" "${LCOV_ARGS[@]}"
mv "$TMP" "$OUT"

test -s "$OUT"
python3 - "$OUT" <<'PY'
import sys
hits = tot = 0
sfs = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if line.startswith("SF:"):
        sfs.append(line.strip())
    if line.startswith("DA:"):
        h = int(line.split(":")[1].split(",")[1])
        tot += 1
        hits += h > 0
print(f"cpp lcov {100*hits/tot:.1f}% {hits}/{tot}" if tot else "empty lcov")
for s in sfs[:8]:
    print(" ", s)
if tot == 0:
    raise SystemExit("no DA records in cpp LCOV")
PY
echo "OK wrote $OUT"
