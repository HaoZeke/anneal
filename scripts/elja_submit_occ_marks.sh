#!/usr/bin/env bash
# Submit LJ75 Marks occupancy on packing-gt-stop 7fe8d5a after BUILD_OK.
# Isolated tree. Does not write sealed lj75-shared-0002 or anneal-stop.
set -euo pipefail

ROOT=${LJ_ROOT:-$HOME/anneal-occ-7fe8d5a}
WANT=7fe8d5a431a4ce227f8e3d424bd299f3eae51103
BIN=$ROOT/target/release/examples/lj_cluster_search
SERVER=$ROOT/target/release/examples/catalog_server
SRC_FILE=$ROOT/SOURCE_COMMIT

test -s "$SRC_FILE"
IFS= read -r SRC <"$SRC_FILE"
if [[ $SRC != "$WANT" ]]; then
  echo "refusing submit: SOURCE_COMMIT=$SRC want=$WANT" >&2
  exit 2
fi
test -x "$BIN"
test -x "$SERVER"
OUT75=$HOME/ljwork/jcc/lj75-occ-gt/lj75/shared/lj75-shared-0000
if [[ -e $OUT75 ]]; then
  echo "refusing submit: $OUT75 already exists" >&2
  exit 2
fi

cd "$ROOT"
if squeue -u "$USER" -h -n lj75-occ-marks 2>/dev/null | grep -q .; then
  j75=$(squeue -u "$USER" -h -n lj75-occ-marks -o '%i' | head -1)
  echo "already queued lj75-occ-marks $j75"
else
  j75=$(sbatch --parsable scripts/elja_lj75_occ_marks.sbatch)
fi
echo "SUBMIT_OK 75=$j75 SOURCE=$SRC"
