#!/usr/bin/env bash
# Anneal against GMIN on reduced Sutton–Chen. One UTS namespace, hostname
# bench, working directory /work. The potential call is rgpot.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
LOGS=$ROOT/bench/metal-cluster-h2h/sc-8e076b7f/logs
LIB=${RGPOT_SC_LIB:-/work/rgpot/bdir-xtb/CppCore/librgpot.so}

bash "$HERE/build_sc.sh"
bash "$HERE/build_gmin_sc.sh"
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$LOGS" /work
cp -a "$HERE/run_sc.py" "$WORK/bin/"
chmod +x "$WORK/bin/h2h_sc" "$WORK/bin/GMIN" "$WORK/bin/run_sc.py"

cat > "$WORK/bin/inside.sh" << EOF
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
exec /usr/bin/python3 -u $WORK/bin/run_sc.py \\
  --binary $WORK/bin/h2h_sc \\
  --gmin $WORK/bin/GMIN \\
  --runs $WORK/runs \\
  --logs $LOGS \\
  --summary $ROOT/bench/metal-cluster-h2h/sc-8e076b7f/results.csv \\
  --seeds-out $ROOT/bench/metal-cluster-h2h/sc-8e076b7f/seeds.csv \\
  --source $WORK/gmin-source.txt \\
  --head-file $WORK/anneal-head \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside.sh "$@"' _ "$@"
