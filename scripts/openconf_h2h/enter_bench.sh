#!/usr/bin/env bash
# Copyright (c) anneal developers
# anneal, openconf and CREST on drug-like molecules. One UTS namespace,
# hostname bench, working directory /work, every job on core 0.
#   enter_bench.sh pilot | campaign [--only MOLECULE SEED ARM] | score
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/openconf-h2h
PREFIX=${OPENCONF_H2H_ENV:-/work/env}

if [[ ${OPENCONF_H2H_SKIP_BUILD:-0} != 1 ]]; then
  bash "$HERE/build.sh"
fi
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" /work

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$PREFIX/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export PYTHONHASHSEED=0
cd /work
exec $PREFIX/bin/python -u $HERE/run_bench.py "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; user=$1; shift; exec sudo -u "$user" '"$WORK"'/bin/inside.sh "$@"' _ "$(id -un)" "$@"
