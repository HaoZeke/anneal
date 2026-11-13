#!/usr/bin/env bash
# Gradients per quench on GFN2-xTB: anneal L-BFGS, anneal TRIC, geomeTRIC
# TRIC and Cartesian. One UTS namespace, hostname bench, working directory
# /work, one core.
#   enter_bench.sh [run options]   replay the quenches, then summarise
#   enter_bench.sh search          whole searches with each anneal quench
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/tric-quench
BASE=${OPENCONF_H2H_ENV:-/work/env}
PREFIX=${TRIC_QUENCH_ENV:-/work/env-dev}
CORE=${TRIC_QUENCH_CORE:-0}

# The head-to-head campaign imports anneal from the base environment; this
# bench installs the checkout into a venv layered over it.
if [[ ! -x $PREFIX/bin/python ]]; then
  "$BASE/bin/python" -m venv --system-site-packages "$PREFIX"
fi
if [[ ${TRIC_QUENCH_SKIP_BUILD:-0} != 1 ]]; then
  (cd "$ROOT" && PATH=$BASE/bin:$PATH CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/work/target-py} \
    maturin build --release --locked -i "$BASE/bin/python" -o "$WORK/wheels")
  "$PREFIX/bin/pip" install -q --force-reinstall --no-deps "$WORK"/wheels/anneal-*.whl
fi
mkdir -p "$WORK/home" "$WORK/tmp"

cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
[ "\$(hostname)" = bench ] || { echo "hostname is \$(hostname), not bench" >&2; exit 1; }
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$PREFIX/bin:$BASE/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export PYTHONHASHSEED=0
cd /work
run="taskset -c $CORE $PREFIX/bin/python -u $HERE/quench_compare.py"
if [ "\${1:-}" = search ]; then
  shift
  \$run search "\$@"
  cp $WORK/searches.csv $ROOT/bench/tric-quench/searches.csv
  exit 0
fi
[ -f $WORK/starts.npz ] || \$run starts
\$run run "\$@"
\$run summary
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; user=$1; shift; exec sudo -u "$user" '"$WORK"'/inside.sh "$@"' _ "$(id -un)" "$@"
