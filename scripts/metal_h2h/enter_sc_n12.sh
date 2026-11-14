#!/usr/bin/env bash
# Twelve-seed Sutton–Chen head-to-head. One UTS namespace, hostname bench.
# Finished seeds already in the table are kept. The six-seed table is not
# opened.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
LOGS=$ROOT/bench/metal-cluster-h2h/sc-n12/logs
RUNS=$WORK/runs-n12
LIB=${RGPOT_SC_LIB:-/work/rgpot/bdir-xtb/CppCore/librgpot.so}

bash "$HERE/build_sc.sh"
bash "$HERE/build_gmin_sc.sh"
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$RUNS" "$LOGS" /work \
  "$ROOT/bench/metal-cluster-h2h/sc-n12"
cp -a "$HERE/run_sc.py" "$HERE/run_sc_n12.py" "$WORK/bin/"
chmod +x "$WORK/bin/h2h_sc" "$WORK/bin/GMIN" "$WORK/bin/run_sc.py" "$WORK/bin/run_sc_n12.py"

cat > "$WORK/bin/inside-n12.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/usr/bin:/bin
export RGPOT_SC_LIB=$LIB
export LD_LIBRARY_PATH=$(dirname "$LIB")\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd /work
exec /usr/bin/python3 -u $WORK/bin/run_sc_n12.py \\
  --binary $WORK/bin/h2h_sc \\
  --gmin $WORK/bin/GMIN \\
  --runs $RUNS \\
  --logs $LOGS \\
  --summary $ROOT/bench/metal-cluster-h2h/sc-n12/results.csv \\
  --seeds-out $ROOT/bench/metal-cluster-h2h/sc-n12/seeds.csv \\
  --source $WORK/gmin-source.txt \\
  --head-file $WORK/anneal-head \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside-n12.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-n12.sh "$@"' _ "$@"
