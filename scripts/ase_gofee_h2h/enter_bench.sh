#!/bin/bash
# Copyright (c) anneal developers
# Timed Lennard-Jones arms, one UTS namespace, hostname bench, paths under /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/ase-gofee-h2h
BINARY=${H2H_BINARY:-/work/target/release/examples/h2h_gmin_proof}
VENV=${H2H_VENV:-$WORK/venv}

mkdir -p "$WORK/runs" "$WORK/home" "$WORK/tmp" "$WORK/mpl" "$WORK/numba" "$WORK/cache" \
  /work/scratch /work/target

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing venv $VENV" >&2
  exit 1
fi

if [[ ! -x "$BINARY" ]]; then
  export CARGO_TARGET_DIR=/work/target
  export RUSTFLAGS="--remap-path-prefix=/work/anneal=/src --remap-path-prefix=/agent/repos/anneal=/src --remap-path-prefix=/work/target=/src --remap-path-prefix=/work/scratch=/build"
  (cd /work/anneal && cargo +1.88 build --release --example h2h_gmin_proof)
fi

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu env \
    HOME=/work/ase-gofee-h2h/home \
    TMPDIR=/work/ase-gofee-h2h/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/work/ase-gofee-h2h/mpl \
    NUMBA_CACHE_DIR=/work/ase-gofee-h2h/numba \
    XDG_CACHE_HOME=/work/ase-gofee-h2h/cache \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/work/anneal/scripts/ase_gofee_h2h \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    OMP_PROC_BIND=TRUE \
    OMP_PLACES=cores \
    RAYON_NUM_THREADS=1 \
    RAY_AUTH_MODE=disabled \
    PATH=/work/ase-gofee-h2h/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    taskset -c 0 \
    /work/ase-gofee-h2h/venv/bin/python -u /work/anneal/scripts/ase_gofee_h2h/campaign.py \
      --binary /work/target/release/examples/h2h_gmin_proof \
      --repo /work/anneal \
      --runs /work/ase-gofee-h2h/runs \
      "$@"' _ "$@"
