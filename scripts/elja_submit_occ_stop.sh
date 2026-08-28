#!/usr/bin/env bash
# Submit occupancy hop jobs against anneal-stop after BUILD_OK.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-stop}
BIN=$ROOT/target/release/examples/lj_cluster_search
SERVER=$ROOT/target/release/examples/catalog_server
test -x "$BIN"
test -x "$SERVER"
test -s "$ROOT/SOURCE_COMMIT"
if ldd "$BIN" | grep -F "not found" >/dev/null; then
  echo "unresolved libraries in $BIN" >&2
  ldd "$BIN" >&2
  exit 2
fi
if ! grep -q personal "$BIN" 2>/dev/null; then
  # release binary may strip strings; identity is SOURCE_COMMIT
  :
fi
echo "SOURCE=$(cat "$ROOT/SOURCE_COMMIT")"
echo "BIN=$BIN"
sbatch "$ROOT/scripts/elja_lj38_occ_hops.sbatch"
sbatch "$ROOT/scripts/elja_lj75_occ_hops.sbatch"
sbatch "$ROOT/scripts/elja_lj98_occ_hops.sbatch"
