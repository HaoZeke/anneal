#!/bin/bash
# GFN2 water hexamer board: water_hexamer gfn2 good_turing, seeds 0 to 31,
# 4000 calls, through the deposited tblite 0.7.0 ASE calculator.
# Usage: [BRIDGE=ase_objective.py] [APART=1 [JOBS=n]] experiments/water_board.sh BINARY OUTDIR
# A binary runs the examples/ase_objective.py of the tree it was built in;
# BRIDGE runs another one instead, so builds of different trees can share it.
# APART=1 runs each seed in its own process (WATER_SEED), JOBS at a time, so
# no seed inherits the SCF history of the seeds before it; the binary must
# read WATER_SEED.
set -euo pipefail
BIN=$1 OUT=$2
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
export ASE_PYTHON=${ASE_PYTHON:-/tmp/tb07/bin/python}
export LD_LIBRARY_PATH=${IRA_LIB_DIR:-/tmp/h2h/ira/lib}:/tmp/fort/lib
export WATER_ISOMER_DIR=${WATER_ISOMER_DIR:-/tmp/v4-stage/isomers}
export PAPER_OUT=$OUT/board.txt
unset PAPER_SMOKE WATER_GOFEE_DIR WATER_SEED
work=$(mktemp -d)
if [ -n "${BRIDGE:-}" ]; then
  printf '#!/bin/bash\nexec %q %q\n' "$ASE_PYTHON" "$(realpath "$BRIDGE")" >"$work/python"
  chmod +x "$work/python"
  ASE_PYTHON=$work/python
fi
cd "$work"
status=0
if [ -n "${APART:-}" ]; then
  mkdir -p "$OUT/seeds"
  export BIN OUT work
  seq 0 31 | xargs -P "${JOBS:-1}" -I{} bash -c \
    'mkdir -p "$work/{}" && cd "$work/{}" && PAPER_OUT=$OUT/seeds/{}.txt WATER_SEED={} \
      "$BIN" gfn2 good_turing >"$OUT/seeds/{}.log" 2>"$OUT/seeds/{}.err"' || status=$?
  {
    cat "$OUT/seeds/0.txt"
    for seed in $(seq 1 31); do sed -n '/^seed: /,$p' "$OUT/seeds/$seed.txt"; done
  } >"$OUT/board.txt"
else
  "$BIN" gfn2 good_turing >"$OUT/board.log" 2>"$OUT/board.err" || status=$?
fi
echo $status >"$OUT/exit"
cd / && rm -rf "$work"
