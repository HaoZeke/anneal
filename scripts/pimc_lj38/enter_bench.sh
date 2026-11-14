#!/usr/bin/env bash
# Both arms, one UTS namespace, hostname bench, working directory under /work.
# The timed process is pinned to core 0. RPQA uses one force client.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/pimc-lj38

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK"

bash "$HERE/build_anneal.sh"
bash "$HERE/install_rpqa.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" \
  "$ROOT/bench/pimc-lj38/rows"
cp -a "$HERE/run_campaign.py" "$HERE/rpqa_driver.py" "$HERE/rpqa_ipi.py" \
  "$HERE/lj_potential.py" "$HERE/meter.py" "$WORK/bin/"
chmod +x "$WORK/bin/h2h_gmin_proof"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:\$PATH
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
  set -- --summary $ROOT/bench/pimc-lj38/results.csv \\
    --seeds-out $ROOT/bench/pimc-lj38/seeds.csv
fi
exec taskset -c 0 python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/h2h_gmin_proof \\
  --source $WORK/rpqa-source.txt \\
  --ipi-root $WORK/i-pi \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs \\
  --rows $ROOT/bench/pimc-lj38/rows \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
if [[ "${1:-}" == "paper" ]]; then
  shift
  set -- --arm rpqa_paper \
    --summary "$ROOT/bench/pimc-lj38/results.csv" \
    --seeds-out "$ROOT/bench/pimc-lj38/seeds.csv" \
    "$@"
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/pimc-lj38/bin/inside.sh "$@"' _ "$@"
