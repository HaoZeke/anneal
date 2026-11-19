#!/usr/bin/env bash
# Copyright (c) anneal developers
# (H2O)8/12/16/20/21 head-to-head, anneal's default preset against GMIN's
# rigid-body TIP4P basin-hopping, twelve seeds, one core.
# One UTS namespace, hostname bench, working directory /work.
# With --size/--seed (the asv suite) rows go to scratch files under WORK.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/water-gmin
OUT=${WATER_GMIN_OUT:-/work/water-gmin/output}

if [[ -n $(git -C "$ROOT" status --porcelain -- src examples Cargo.toml Cargo.lock build.rs) ]]; then
  echo "anneal sources differ from HEAD; code_revision would not name the binary" >&2
  exit 1
fi
H2H_WORK=$WORK bash "$HERE/build_anneal.sh"

GMIN_SRC=/work/gmin-h2h/src/wales/GMIN/source/newtip.f90
if [[ -f $GMIN_SRC ]] && ! grep -q H2H_POT_ENTER "$GMIN_SRC"; then
  patch -p1 --forward --batch -d /work/gmin-h2h/src/wales/GMIN < "$HERE/newtip-timer.patch"
  rm -f /work/gmin-h2h/bin/GMIN
fi
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"
GMIN_REV=gmin-$(sed -n 's/^revision=//p' /work/gmin-h2h/gmin-source.txt)

mkdir -p "$WORK/bin" "$WORK/user" "$WORK/scratch" "$WORK/runs" "$WORK/gm" "$OUT"
cp -a "$HERE/run_water_gmin.py" "$HERE/run_h2h.py" "$HERE/gm_radius.py" "$ROOT/scripts/target_tol.py" "$WORK/bin/"
cp -a "$ROOT/examples/h2h_gmin_tip4p.data" "$WORK/bin/h2h_gmin_tip4p.data"
{ sed 's/^STEPS .*/STEPS 0 1.0/' "$ROOT/examples/h2h_gmin_tip4p.data"; echo "CHECKD 0"; } \
  > "$WORK/bin/h2h_gmin_tip4p_verify.data"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
cp -a /work/gmin-h2h/gmin-source.txt "$WORK/gmin-source.txt"
chmod +x "$WORK/bin/GMIN" "$WORK/bin/h2h_seed" "$WORK/bin/h2h_tip4p_point"

if [[ $# -gt 0 ]]; then
  ROWS=$WORK/scratch-rows.csv
  SUMMARY=$WORK/scratch-summary.csv
  VERIFY=$WORK/scratch-verify.csv
else
  ROWS=$OUT/rows.csv
  SUMMARY=$OUT/summary.csv
  VERIFY=$OUT/verify.csv
fi

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/user
export TMPDIR=$WORK/scratch
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
exec /usr/bin/python3 -u $WORK/bin/run_water_gmin.py \\
  --repo $ROOT \\
  --binary $WORK/bin/h2h_seed \\
  --point $WORK/bin/h2h_tip4p_point \\
  --anneal-revision $(cat "$WORK/anneal-head") \\
  --gmin $WORK/bin/GMIN \\
  --gmin-revision $GMIN_REV \\
  --data $WORK/bin/h2h_gmin_tip4p.data \\
  --verify-data $WORK/bin/h2h_gmin_tip4p_verify.data \\
  --gm-dir $WORK/gm \\
  --verify-out $VERIFY \\
  --work $WORK/runs \\
  --rows $ROWS \\
  --summary $SUMMARY \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/water-gmin/bin/inside.sh "$@"' _ "$@"
