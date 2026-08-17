#!/usr/bin/env bash
# Submit LJ38 occupancy ensemble 0003 on the complete occupancy
# binary. Refuses any other SOURCE_COMMIT.
set -euo pipefail

ROOT=${LJ_ROOT:-$HOME/anneal-stop}
WANT=b9c8cf378ff0d364f91523e737c9dfd8ec2a4112
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

OUT=$HOME/ljwork/jcc/lj38-occ-sb/lj38/shared/lj38-shared-0004
if [[ -e $OUT ]]; then
  echo "refusing submit: $OUT already exists" >&2
  exit 2
fi

cd "$ROOT"
if squeue -u "$USER" -h -n lj38-occ-hops 2>/dev/null | grep -q .; then
  j38=$(squeue -u "$USER" -h -n lj38-occ-hops -o '%i' | head -1)
  echo "already queued lj38-occ-hops $j38"
else
  j38=$(sbatch --parsable scripts/elja_lj38_occ_hops.sbatch)
fi
echo "SUBMIT_OK 38=$j38 SOURCE=$SRC"
