#!/bin/sh
# One AGOX seed. The caller already set hostname bench via unshare --uts.
set -eu
cd /work/anneal
seed=$1
echo "wrapper_header hostname bench was set via unshare --uts"
echo "uts_hostname=$(hostname)"
echo "seed=$seed arm=agox"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export PYTHONUNBUFFERED=1
export LJ_BOARD_ROOT=results/lj-board
export HOME=/work/home
export TMPDIR=/work/scratch
export MPLCONFIGDIR=/work/mpl
export NUMBA_CACHE_DIR=/work/numba
export XDG_CACHE_HOME=/work/cache
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=/work/anneal
out=results/lj-board/agox
mkdir -p "$out" results/lj-board/provenance
exec /work/anneal/.venv/bin/py-spy record \
  --nonblocking \
  -r 50 \
  -f speedscope \
  -o "$out/lj38-seed${seed}.speedscope.json" \
  -- /work/anneal/.venv/bin/python3 -u /work/run/agox_one.py "$seed"
