#!/usr/bin/env bash
# Both arms, one UTS namespace, hostname bench, working directory under /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-h2h
LOGS=$ROOT/bench/metal-cluster-h2h/logs

bash "$HERE/build_anneal.sh"
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$LOGS"
cp -a "$HERE/run_campaign.py" "$HERE/ase_ga.py" "$HERE/gupta_calc.py" "$HERE/check_match.py" "$WORK/bin/"
chmod +x "$WORK/bin/h2h_metal"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/home/ubuntu/.local/bin:\$PATH
export PYTHONPATH=$WORK/bin:/home/ubuntu/.local/lib/python3.12/site-packages
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
export METAL_FIXTURES=$ROOT/tests/fixtures/ccd
cd $WORK
python3 -u $WORK/bin/check_match.py $WORK/bin/h2h_metal
if [ "\$#" -eq 0 ]; then
  set -- --summary $ROOT/bench/metal-cluster-h2h/results.csv
fi
exec /usr/bin/python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/h2h_metal \\
  --ga $WORK/bin/ase_ga.py \\
  --runs $WORK/runs \\
  --logs $LOGS \\
  --head-file $WORK/anneal-head \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-h2h/bin/inside.sh "$@"' _ "$@"
