#!/usr/bin/env bash
# TIP4P15 and TIP4P20 against GMIN. One UTS namespace, hostname bench, one core.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/tip4p-h2h

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs"
cp -a "$HERE/run_campaign.py" "$WORK/bin/run_campaign.py"
cp -a "$ROOT/examples/h2h_gmin_tip4p.data" "$WORK/bin/h2h_gmin_tip4p.data"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
cp -a /work/tip4p-fix/target/release/examples/h2h_seed "$WORK/bin/h2h_seed"
chmod +x "$WORK/bin/GMIN" "$WORK/bin/h2h_seed" "$WORK/bin/run_campaign.py"

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
exec /usr/bin/python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/h2h_seed \\
  --gmin $WORK/bin/GMIN \\
  --data $WORK/bin/h2h_gmin_tip4p.data \\
  --work $WORK/runs \\
  --rows $ROOT/bench/tip4p-h2h/rows.csv \\
  --summary $ROOT/bench/tip4p-h2h/summary.csv \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/tip4p-h2h/bin/inside.sh "$@"' _ "$@"
