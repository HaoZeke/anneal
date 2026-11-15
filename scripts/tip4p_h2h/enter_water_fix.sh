#!/usr/bin/env bash
# Copyright (c) anneal developers
# The water_gmin panel for a develop revision and this checkout, back to
# back in one session, then the branch-against-develop comparison.
# Both anneal arms append to bench/water-gmin/rows.csv; GMIN rows are reused
# wherever the start sha1 matches.
#
#   enter_water_fix.sh <develop revision>
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVELOP=${1:?develop revision}
OUT=$ROOT/bench/water-fix
TREE=/work/water-fix/develop-tree

mkdir -p "$OUT" /work/water-fix
develop_rev=$(git -C "$ROOT" rev-parse "$DEVELOP^{commit}")
if [[ ! -d $TREE ]]; then
  git -C "$ROOT" worktree add --detach "$TREE" "$develop_rev"
fi
git -C "$TREE" checkout --quiet --detach "$develop_rev"

ANNEAL_ROOT=$TREE WATER_GMIN_WORK=/work/water-fix/develop \
  WATER_GMIN_SUMMARY=$OUT/summary-develop.csv bash "$HERE/enter_water_gmin.sh"
WATER_GMIN_WORK=/work/water-fix/branch \
  WATER_GMIN_SUMMARY=$OUT/summary-branch.csv bash "$HERE/enter_water_gmin.sh"

python3 "$HERE/compare_water_arms.py" \
  --rows "$ROOT/bench/water-gmin/rows.csv" \
  --develop "$develop_rev" \
  --branch "$(cat /work/water-fix/branch/anneal-head)" \
  --gmin "gmin-$(sed -n 's/^revision=//p' /work/gmin-h2h/gmin-source.txt)" \
  --output "$OUT/compare.csv"
