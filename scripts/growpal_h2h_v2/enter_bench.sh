#!/usr/bin/env bash
# Both arms, one UTS namespace, hostname bench, working directory under /work.
# The campaign and every timed child are pinned to core 0. GrowPAL runs
# with one worker and its published search parameters.
#
# Extra arguments go to run_campaign.py after the defaults, so --out and
# --runs can point a single timed seed (the asv benchmark) at scratch.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/growpal-h2h-v2

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK"

bash "$HERE/build_anneal.sh"
bash "$HERE/install_growpal.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$ROOT/bench/growpal-h2h-v2"
cp -a "$HERE/run_campaign.py" "$HERE/growpal_arm.py" "$WORK/bin/"
user=$(id -un)

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
exec taskset -c 0 $WORK/venv/bin/python -u -W ignore $WORK/bin/run_campaign.py \\
  --bin-dir $WORK/bin \\
  --python $WORK/venv/bin/python \\
  --source $WORK/growpal-source.txt \\
  --seeds-xyz $WORK/src/GrowPAL/examples/LJ/LJ010.xyz \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs \\
  --out $ROOT/bench/growpal-h2h-v2 \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u "$0" '"$WORK"'/bin/inside.sh "$@"' "$user" "$@"
