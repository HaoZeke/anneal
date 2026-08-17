#!/usr/bin/env bash
# Submit LJ75 ensemble 0007 and LJ98 ensemble 0005 on the complete
# occupancy binary (Leave only into a new DECAF family; no invented
# n_occupied_families). Refuses any other SOURCE_COMMIT.
set -euo pipefail

ROOT=${LJ_ROOT:-$HOME/anneal-stop}
WANT=41ad564ef5a69fef99445d03c3064464705025e1
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
if ! grep -a -F -q different_decaf_family "$BIN"; then
  echo "refusing submit: $BIN missing different_decaf_family" >&2
  exit 2
fi

OUT75=$HOME/ljwork/jcc/lj75-occ-sb/lj75/shared/lj75-shared-0007
OUT98=$HOME/ljwork/jcc/lj98-occ-sb/lj98/shared/lj98-shared-0005
if [[ -e $OUT75 ]]; then
  echo "refusing submit: $OUT75 already exists" >&2
  exit 2
fi
if [[ -e $OUT98 ]]; then
  echo "refusing submit: $OUT98 already exists" >&2
  exit 2
fi

cd "$ROOT"
j75=""
j98=""
if squeue -u "$USER" -h -n lj75-occ-hops 2>/dev/null | grep -q .; then
  j75=$(squeue -u "$USER" -h -n lj75-occ-hops -o '%i' | head -1)
  echo "already queued lj75-occ-hops $j75"
else
  j75=$(sbatch --parsable scripts/elja_lj75_occ_hops.sbatch)
fi
if squeue -u "$USER" -h -n lj98-occ-hops 2>/dev/null | grep -q .; then
  j98=$(squeue -u "$USER" -h -n lj98-occ-hops -o '%i' | head -1)
  echo "already queued lj98-occ-hops $j98"
else
  j98=$(sbatch --parsable scripts/elja_lj98_occ_hops.sbatch)
fi
echo "SUBMIT_OK 75=$j75 98=$j98 SOURCE=$SRC"
