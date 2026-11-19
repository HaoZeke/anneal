#!/usr/bin/env bash
# Copyright (c) anneal developers
# GAD rerun, anneal, and GMIN, plus the deposited GAD run. One UTS namespace,
# hostname bench, one core (GAD_BENCH_CORE, default 0).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/gad-published
ZIP=${GAD_ZIP:-$WORK/supplementary_data.zip}
CORE=${GAD_BENCH_CORE:-0}

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK"

bash "$HERE/build_anneal.sh"
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"
bash "$HERE/build_gad.sh"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
cp -a /work/gmin-h2h/gmin-source.txt "$WORK/gmin-source.txt"
cp -a "$ROOT/examples/h2h_gmin_lj.data" "$WORK/bin/h2h_gmin_lj.data"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs"
cp -a "$HERE/run_campaign.py" "$HERE/read_deposit.py" "$HERE/gad_rerun.py" "$ROOT/scripts/target_tol.py" "$WORK/bin/"
chmod +x "$WORK/bin/gad_published" "$WORK/bin/GMIN"

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
export TQDM_DISABLE=1
cd $WORK
if [ "\$#" -eq 0 ]; then
  set -- --summary $ROOT/bench/gad-published/rows.csv
fi
HEAD=\$(cat $WORK/anneal-head)
exec taskset -c $CORE $WORK/venv/bin/python -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/gad_published \\
  --gmin $WORK/bin/GMIN \\
  --data $WORK/bin/h2h_gmin_lj.data \\
  --zip $ZIP \\
  --source $WORK/gmin-source.txt \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs/\$HEAD \\
  --core $CORE \\
  --gad-python $WORK/venv/bin/python \\
  --gad-script $WORK/bin/gad_rerun.py \\
  --condevo $WORK/src/condevo \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/gad-published/bin/inside.sh "$@"' _ "$@"
