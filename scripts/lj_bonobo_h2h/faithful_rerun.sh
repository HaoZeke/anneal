#!/bin/bash
# LJ38 at the paper's population and iteration cap, then LJ75 at 20000
# relaxations. One process list, so both arms share the bench namespace.
set -eu
PY=${PY:-python3}
CAMP=${CAMP:-/work/lj-bonobo-h2h/bin/run_campaign.py}
FIX=${FIX:-/agent/repos/anneal/bench/lj-bonobo-h2h/cambridge_lj.tsv}
BIN=${BIN:-/work/lj-bonobo-h2h/bin/lj_lean_relax_budget}
UP=${UP:-/work/lj-bonobo-h2h/upstream/bonobo}
RUNS=${RUNS:-/work/lj-bonobo-h2h/runs/faithful}
OUT=${OUT:-/agent/repos/anneal/bench/lj-bonobo-h2h}

common=(
  --fixture "$FIX"
  --binary "$BIN"
  --upstream "$UP"
  --runs "$RUNS"
)

echo "stage lj38 population 20 max_iter 1500 relaxations 30020" 
"$PY" -u "$CAMP" "${common[@]}" \
  --size 38 --budget 30020 --population 20 --max-iter 1500 \
  --summary "$OUT/lj38_paper.tsv" --seeds-out "$OUT/lj38_paper_seeds.tsv"

echo "stage lj75 population 20 max_iter 999 relaxations 20000"
"$PY" -u "$CAMP" "${common[@]}" \
  --size 75 --budget 20000 --population 20 --max-iter 999 \
  --summary "$OUT/lj75_20k.tsv" --seeds-out "$OUT/lj75_20k_seeds.tsv"
