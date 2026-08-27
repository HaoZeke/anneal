#!/usr/bin/env bash
# Submit LJ75 Marks occupancy on packing-gt-stop 9dde60d after BUILD_OK.
# Isolated tree. Does not write sealed ensembles 0002 or 0003, or anneal-stop.
set -euo pipefail

ROOT=${LJ_ROOT:-$HOME/anneal-occ-9dde60d}
NEED=9dde60d5130623c7862e29181697508af4dc2ba8
BIN=$ROOT/target/release/examples/lj_cluster_search
SERVER=$ROOT/target/release/examples/catalog_server
SRC_FILE=$ROOT/SOURCE_COMMIT

test -s "$SRC_FILE"
IFS= read -r SRC <"$SRC_FILE"
HEAD=$(git -C "$ROOT" rev-parse HEAD)
if [[ $SRC != "$HEAD" ]]; then
  echo "refusing submit: SOURCE_COMMIT=$SRC does not match HEAD=$HEAD" >&2
  exit 2
fi
if ! git -C "$ROOT" merge-base --is-ancestor "$NEED" "$SRC"; then
  echo "refusing submit: SOURCE_COMMIT=$SRC is not a descendant of $NEED" >&2
  exit 2
fi
test -x "$BIN"
test -x "$SERVER"
if [[ ! -s $ROOT/BUILD_SHA256SUMS ]]; then
  echo "refusing submit: missing $ROOT/BUILD_SHA256SUMS" >&2
  exit 2
fi
(cd "$ROOT" && sha256sum -c BUILD_SHA256SUMS)
OUT75=$HOME/ljwork/jcc/lj75-occ-dwell/lj75/shared/lj75-shared-0021
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
