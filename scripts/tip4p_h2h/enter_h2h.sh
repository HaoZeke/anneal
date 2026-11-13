#!/usr/bin/env bash
# TIP4P head-to-head, anneal against GMIN, twelve seeds, one core.
# One UTS namespace, hostname bench, working directory /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/tip4p-h2h

bash "$HERE/build_anneal.sh"

# Time NEWTIP the way the Lennard-Jones kernel is timed. The public GMIN
# build only brackets LJ, so pot_s would be empty for the water arm.
GMIN_SRC=/work/gmin-h2h/src/wales/GMIN/source/newtip.f90
if [[ -f $GMIN_SRC ]] && ! grep -q H2H_POT_ENTER "$GMIN_SRC"; then
  patch -p1 --forward --batch -d /work/gmin-h2h/src/wales/GMIN < "$HERE/newtip-timer.patch"
  rm -f /work/gmin-h2h/bin/GMIN
fi
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$WORK/gm"
cp -a "$HERE/run_h2h.py" "$WORK/bin/run_h2h.py"
cp -a "$ROOT/examples/h2h_gmin_tip4p.data" "$WORK/bin/h2h_gmin_tip4p.data"
# A single-point data file for the reproduction check: no basin hops, and
# CHECKD prints the energy at the input geometry.
{ sed 's/^STEPS .*/STEPS 0 1.0/' "$ROOT/examples/h2h_gmin_tip4p.data"; echo "CHECKD 0"; } \
  > "$WORK/bin/h2h_gmin_tip4p_verify.data"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
cp -a /work/gmin-h2h/gmin-source.txt "$WORK/gmin-source.txt"
chmod +x "$WORK/bin/GMIN" "$WORK/bin/h2h_seed" "$WORK/bin/run_h2h.py"

# Published minima as NEWTIP centre/angle-axis blocks for the check.
python3 "$HERE/gm_radius.py" --gmin "$WORK/gm" >/dev/null

cat > "$WORK/bin/inside.sh" << EOF
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
exec /usr/bin/python3 -u $WORK/bin/run_h2h.py \\
  --binary $WORK/bin/h2h_seed \\
  --point $WORK/bin/h2h_tip4p_point \\
  --gmin $WORK/bin/GMIN \\
  --data $WORK/bin/h2h_gmin_tip4p.data \\
  --verify-data $WORK/bin/h2h_gmin_tip4p_verify.data \\
  --gm-dir $WORK/gm \\
  --verify-out $WORK/verify.csv \\
  --work $WORK/runs \\
  --rows $WORK/rows.csv \\
  --summary $ROOT/bench/tip4p-h2h/h2h-summary.csv \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/tip4p-h2h/bin/inside.sh "$@"' _ "$@"
