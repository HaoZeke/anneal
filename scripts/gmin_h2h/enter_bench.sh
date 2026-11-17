#!/usr/bin/env bash
# Copyright (c) anneal developers
# Both arms, one UTS namespace, hostname bench, working directory under /work.
# H2H_BRANCH=1 turns on Config::branch_on_stall in the anneal arm; those runs
# are kept under runs-branch, apart from the default ones.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/gmin-h2h
RUNS=$WORK/runs
branch_env=
if [ "${H2H_BRANCH:-}" = 1 ]; then
  RUNS=$WORK/runs-branch
  branch_env="export H2H_BRANCH=1"
fi

bash "$HERE/build_anneal.sh"
bash "$HERE/build_gmin.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$RUNS" "$WORK/starts"
cp -a "$HERE/run_campaign.py" "$WORK/bin/run_campaign.py"
cp -a "$ROOT/examples/h2h_gmin_lj.data" "$WORK/bin/h2h_gmin_lj.data"
chmod +x "$WORK/bin/h2h_gmin_proof" "$WORK/bin/GMIN"
# Captured before HOME moves, so --stats can import scipy.
user_site=$(python3 -c 'import site; print(site.getusersitepackages())')

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PYTHONPATH=$user_site\${PYTHONPATH:+:\$PYTHONPATH}
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
$branch_env
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
  --runs $RUNS \\
  --starts $WORK/starts \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/gmin-h2h/bin/inside.sh "$@"' _ "$@"
