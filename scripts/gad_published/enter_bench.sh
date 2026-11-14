#!/usr/bin/env bash
# Deposited GAD figures, anneal, and GMIN. One UTS namespace, hostname bench.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/gad-published
ZIP=${GAD_ZIP:-$WORK/supplementary_data.zip}

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK"

bash "$HERE/build_anneal.sh"
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
cp -a /work/gmin-h2h/gmin-source.txt "$WORK/gmin-source.txt"
cp -a "$ROOT/examples/h2h_gmin_lj.data" "$WORK/bin/h2h_gmin_lj.data"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs"
cp -a "$HERE/run_campaign.py" "$HERE/read_deposit.py" "$WORK/bin/"
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
cd $WORK
if [ "\$#" -eq 0 ]; then
  set -- --summary $ROOT/bench/gad-published/summary.csv --zip $ZIP
fi
exec taskset -c 0 /usr/bin/python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/gad_published \\
  --gmin $WORK/bin/GMIN \\
  --data $WORK/bin/h2h_gmin_lj.data \\
  --zip $ZIP \\
  --source $WORK/gmin-source.txt \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/gad-published/bin/inside.sh "$@"' _ "$@"
