#!/usr/bin/env bash
# Both arms, one UTS namespace, hostname bench, working directory under /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/gmin-h2h

bash "$HERE/build_anneal.sh"
bash "$HERE/build_gmin.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$WORK/starts"
cp -a "$HERE/run_campaign.py" "$WORK/bin/run_campaign.py"
cp -a "$ROOT/examples/h2h_gmin_lj.data" "$WORK/bin/h2h_gmin_lj.data"
chmod +x "$WORK/bin/h2h_gmin_proof" "$WORK/bin/GMIN"

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
cd $WORK
if [ "\$#" -eq 0 ]; then
  set -- --summary $ROOT/bench/gmin-h2h/results.csv
fi
exec /usr/bin/python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/h2h_gmin_proof \\
  --gmin $WORK/bin/GMIN \\
  --data $WORK/bin/h2h_gmin_lj.data \\
  --source $WORK/gmin-source.txt \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs \\
  --starts $WORK/starts \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/gmin-h2h/bin/inside.sh "$@"' _ "$@"
