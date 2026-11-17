#!/usr/bin/env bash
# Copyright (c) anneal developers
# Hop traces for bench/openconf-h2h/conf-fix9 under hostname bench: every
# pair of the paired run at its budget, and seeds 3000-3119 of 915,
# amprenavir and lopinavir, with elite_return off (develop) and on (anneal).
# OPENCONF_H2H_SKIP_BUILD=1 keeps the installed wheel.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DIR=$ROOT/bench/openconf-h2h/conf-fix9
WORK=/work/openconf-h2h/conf-fix9-traces
PREFIX=${OPENCONF_H2H_ENV:-/work/env}
TRACE="$PREFIX/bin/python $HERE/elite_trace.py"

if [[ ${1:-} != inside ]]; then
  if [[ ${OPENCONF_H2H_SKIP_BUILD:-0} != 1 ]]; then
    OPENCONF_H2H_CREST=0 bash "$HERE/build.sh"
  fi
  mkdir -p "$WORK" /work/openconf-h2h/home /work/openconf-h2h/tmp
  sudo -n unshare --uts sh -c \
    'hostname bench; user=$1; shift; exec sudo -u "$user" bash "$@"' _ "$(id -un)" "$0" inside
  for set in gate dev; do
    for arm in develop anneal; do
      $TRACE score "$WORK/$set-$arm" --references "$DIR/references.xyz" --arm "$arm" \
        --out "$WORK/$set-$arm.csv" --jobs 7
    done
    { cat "$WORK/$set-develop.csv"; tail -n +2 "$WORK/$set-anneal.csv"; } > "$DIR/traces-$set.csv"
  done
  exit 0
fi

export HOME=/work/openconf-h2h/home TMPDIR=/work/openconf-h2h/tmp PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
for arm in develop anneal; do
  if [[ $arm == develop ]]; then kw='{"elite_return": false}'; else kw='{"elite_return": true}'; fi
  rm -rf "$WORK/gate-$arm" "$WORK/dev-$arm"
  $TRACE run --molecules 915,amprenavir,fluvastatin,lopinavir,lorlatinib,rq3,xld,z34 \
    --seeds 2000 2024 --budgets "$DIR/seeds.csv" --kw "$kw" --out "$WORK/gate-$arm" --jobs 7
  $TRACE run --molecules 915,amprenavir,lopinavir --seeds 3000 3120 --kw "$kw" \
    --out "$WORK/dev-$arm" --jobs 7
done
