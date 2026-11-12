#!/usr/bin/env bash
# Anneal against GMIN on Darby Cu-Au. One UTS namespace, hostname bench,
# working directory /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-darby
LOGS=$ROOT/bench/metal-cluster-h2h/darby-logs

bash "$HERE/build_anneal.sh"
bash "$HERE/build_gmin_darby.sh"
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$LOGS" /work
cp -a "$HERE/run_darby.py" "$WORK/bin/"
cp -a /work/metal-h2h/bin/h2h_metal "$WORK/bin/h2h_metal"
chmod +x "$WORK/bin/h2h_metal"

cat > "$WORK/bin/inside.sh" << EOF
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
export RAYON_NUM_THREADS=1
cd /work
exec /usr/bin/python3 -u $WORK/bin/run_darby.py \\
  --binary $WORK/bin/h2h_metal \\
  --gmin $WORK/bin/GMIN \\
  --fixtures $ROOT/tests/fixtures/ccd \\
  --runs $WORK/runs \\
  --logs $LOGS \\
  --summary $ROOT/bench/metal-cluster-h2h/darby.csv \\
  --source $WORK/gmin-source.txt \\
  --head-file /work/metal-h2h/anneal-head \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-darby/bin/inside.sh "$@"' _ "$@"
