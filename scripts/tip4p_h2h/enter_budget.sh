#!/usr/bin/env bash
# (H2O)21 budget sweep, anneal against GMIN, twelve seeds, one core.
# One UTS namespace, hostname bench, working directory /work. Same build and
# arm as enter_h2h.sh, but for one size at 2M, 4M and 8M charged evaluations.
# GMIN at a (budget, seed) pair is deterministic; prior GMIN rows are reused
# through --gmin-rows and GMIN only runs for the pairs they are missing.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/tip4p-h2h

bash "$HERE/build_anneal.sh"

GMIN_SRC=/work/gmin-h2h/src/wales/GMIN/source/newtip.f90
if [[ -f $GMIN_SRC ]] && ! grep -q H2H_POT_ENTER "$GMIN_SRC"; then
  patch -p1 --forward --batch -d /work/gmin-h2h/src/wales/GMIN < "$HERE/newtip-timer.patch"
  rm -f /work/gmin-h2h/bin/GMIN
fi
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs-budget"
cp -a "$HERE/run_budget.py" "$WORK/bin/run_budget.py"
cp -a "$ROOT/examples/h2h_gmin_tip4p.data" "$WORK/bin/h2h_gmin_tip4p.data"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
chmod +x "$WORK/bin/GMIN" "$WORK/bin/h2h_seed" "$WORK/bin/run_budget.py"

REUSE=${H2H_GMIN_ROWS:-}

# The full sweep writes the committed records. A single-pair timing call (asv
# passes --budget/--seed) writes to scratch so it never clobbers them.
if [[ $# -eq 0 ]]; then
  ROWS="$ROOT/bench/tip4p-h2h/rows-budget21.csv"
  SUMMARY="$ROOT/bench/tip4p-h2h/summary-budget21.csv"
else
  ROWS="$WORK/runs-budget/rows.csv"
  SUMMARY="$WORK/runs-budget/summary.csv"
fi

cat > "$WORK/bin/inside_budget.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:\$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd /work
exec /usr/bin/python3 -u $WORK/bin/run_budget.py \\
  --binary $WORK/bin/h2h_seed \\
  --gmin $WORK/bin/GMIN \\
  --data $WORK/bin/h2h_gmin_tip4p.data \\
  --work $WORK/runs-budget \\
  --rows $ROWS \\
  --summary $SUMMARY \\
  ${REUSE:+--gmin-rows $REUSE} \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside_budget.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/tip4p-h2h/bin/inside_budget.sh "$@"' _ "$@"
