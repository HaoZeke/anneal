#!/bin/bash
# Copyright (c) anneal developers
# One GMIN seed at the published temperature-1.5 keywords.
#
# Usage:
#   gmin_run.sh BINARY WRITER WORK SEED BUDGET
#
# WRITER is called as: WRITER coords lj104 SEED STARTFILE
# The head-to-head example is that writer.
# Keywords match GMIN_BEST: temperature 1.5, step 0.40 0.51,
# NEWRESTART 40, AVOID 4.0 20 F. The cap is H2H_BUDGET.
set -eu
binary=$1
writer=$2
work=$3
seed=$4
budget=$5
if [[ -e ${work}/row.txt || -e ${work}/stdout.txt ]]; then
  echo "refusing to overwrite ${work}" >&2
  exit 1
fi
mkdir -p "$work"
"$writer" coords lj104 "$seed" "${work}/start.xyz"
cp "${work}/start.xyz" "${work}/coords"
cat > "${work}/data" << EOF
SLOPPYCONV 5.0D-3
TIGHTCONV 1.0D-3
EDIFF 0.003
MAXIT 600 500
SAVE 2
CENTRE
SORT
TEMPERATURE 1.5
STEPS 90000000 1.0
STEP 0.40 0.51
NEWRESTART 40
AVOID 4.0 20 F
RANSEED ${seed}
EOF
(
  cd "$work"
  H2H_BUDGET="$budget" \
  H2H_TARGET=-582.086642 \
  H2H_TOL=0.001 \
  H2H_SEED="$seed" \
  H2H_SYSTEM=lj104 \
  H2H_ROW="${work}/row.txt" \
    "$binary" > stdout.txt 2> stderr.txt
)
