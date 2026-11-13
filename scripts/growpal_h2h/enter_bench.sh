#!/usr/bin/env bash
# Both arms, one UTS namespace, hostname bench, working directory under /work.
# The timed process is pinned to core 0. GrowPAL's documented nproc is
# cpu_count-2; this session uses one core, which is the setting their
# notes allow on a shared machine. The search parameters stay the published
# defaults.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/growpal-h2h

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK"

bash "$HERE/build_anneal.sh"
bash "$HERE/install_growpal.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" \
  "$ROOT/bench/growpal-h2h/rows"
cp -a "$HERE/run_campaign.py" "$HERE/growpal_arm.py" "$WORK/bin/"
chmod +x "$WORK/bin/h2h_gmin_proof"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:$WORK/venv/bin:\$PATH
export PYTHONPATH=$WORK/bin
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
export PYTHONHASHSEED=0
cd $WORK
if [ "\$#" -eq 0 ]; then
  set -- --summary $ROOT/bench/growpal-h2h/results.csv \\
    --seeds-out $ROOT/bench/growpal-h2h/seeds.csv
fi
exec taskset -c 0 $WORK/venv/bin/python -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/h2h_gmin_proof \\
  --source $WORK/growpal-source.txt \\
  --seeds-xyz $WORK/src/GrowPAL/examples/LJ/LJ010.xyz \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs \\
  --rows $ROOT/bench/growpal-h2h/rows \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/growpal-h2h/bin/inside.sh "$@"' _ "$@"
