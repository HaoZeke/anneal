#!/usr/bin/env bash
# Copyright (c) anneal developers
# GMIN on every ablation system, after the anneal arms.
# One UTS namespace, hostname bench, working directory /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${ABL_WORK:-/work/abl}

GMIN_SRC=/work/gmin-h2h/src/wales/GMIN/source/morsegh.f90
if [[ -f $GMIN_SRC ]] && ! grep -q H2H_POT_ENTER "$GMIN_SRC"; then
  patch -p1 --forward --batch -d /work/gmin-h2h/src/wales/GMIN < "$ROOT/scripts/morse_h2h/morse-timer.patch"
  rm -f /work/gmin-h2h/bin/GMIN
fi
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
bash "$ROOT/scripts/metal_h2h/build_gmin_darby.sh"
cp -a /work/metal-darby/bin/GMIN "$WORK/bin/GMIN-gupta"
cp -a "$HERE/run_gmin.py" "$WORK/bin/run_gmin.py"

cat > "$WORK/bin/inside-gmin.sh" << INNER
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
cd /work
exec /usr/bin/python3 -u $WORK/bin/run_gmin.py \\
  --root $ROOT --gmin $WORK/bin/GMIN --gmin-gupta $WORK/bin/GMIN-gupta --bins $WORK/bin \\
  --ablation-runs $WORK/runs --work $WORK/gmin-runs \\
  --rows $WORK/gmin-rows.csv "\$@"
INNER
chmod +x "$WORK/bin/inside-gmin.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu "$0" "$@"' "$WORK/bin/inside-gmin.sh" "$@"
