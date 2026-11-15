#!/usr/bin/env bash
# Copyright (c) anneal developers
# The water_gmin panel for a develop revision and this checkout, back to
# back in one session, then the branch-against-develop comparison and the
# trace identity of the two builds on the other potentials.
# (H2O)8/12/20 take twelve seeds, (H2O)16/21 thirty-six; GMIN runs on the
# first twelve. Both anneal arms append to bench/water-gmin/rows.csv; GMIN
# rows are reused wherever the start sha1 matches.
#
#   enter_water_fix.sh <develop revision> [panel|identity|all]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVELOP=${1:?develop revision}
STEP=${2:-all}
OUT=$ROOT/bench/water-fix
WORK=/work/water-fix
TREE=$WORK/develop-tree
WIDE=36

mkdir -p "$OUT" "$WORK"
develop_rev=$(git -C "$ROOT" rev-parse "$DEVELOP^{commit}")
if [[ ! -d $TREE ]]; then
  git -C "$ROOT" worktree add --detach "$TREE" "$develop_rev"
fi
git -C "$TREE" checkout --quiet --detach "$develop_rev"

arm() {
  "$@" bash "$HERE/enter_water_gmin.sh" --sizes 8 12 20
  "$@" bash "$HERE/enter_water_gmin.sh" --sizes 16 21 --seeds "$WIDE" --gmin-seeds 12
}
if [[ $STEP == panel || $STEP == all ]]; then
  arm env ANNEAL_ROOT="$TREE" WATER_GMIN_WORK="$WORK/develop" \
    WATER_GMIN_SUMMARY="$OUT/summary-develop.csv"
  arm env WATER_GMIN_WORK="$WORK/branch" WATER_GMIN_SUMMARY="$OUT/summary-branch.csv"
  python3 "$HERE/compare_water_arms.py" \
    --rows "$ROOT/bench/water-gmin/rows.csv" \
    --develop "$develop_rev" \
    --branch "$(cat "$WORK/branch/anneal-head")" \
    --gmin "gmin-$(sed -n 's/^revision=//p' /work/gmin-h2h/gmin-source.txt)" \
    --seeds 16="$WIDE" 21="$WIDE" \
    --output "$OUT/compare.csv"
fi
[[ $STEP == identity || $STEP == all ]] || exit 0

for side in develop branch; do
  tree=$ROOT
  [[ $side == develop ]] && tree=$TREE
  (cd "$tree" && CARGO_TARGET_DIR=$WORK/ti-$side cargo +1.88.0 build --release --locked \
    --example h2h_seed --example h2h_morse --example h2h_metal)
done
cat > "$WORK/identity.sh" << EOF
#!/bin/bash
set -eu
cd /work
JOBS=1 taskset -c 0 bash $ROOT/scripts/generic_alloy/trace_identity.sh \\
  $WORK/ti-develop/release/examples $develop_rev \\
  $WORK/ti-branch/release/examples $(git -C "$ROOT" rev-parse HEAD) > $OUT/identity.txt
EOF
chmod +x "$WORK/identity.sh"
sudo -n unshare --uts sh -c \
  'inside=$1; hostname bench; exec sudo -u ubuntu "$inside"' _ "$WORK/identity.sh"
