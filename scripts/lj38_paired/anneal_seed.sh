#!/bin/sh
# One anneal seed. The caller already set hostname bench via unshare --uts.
set -eu
cd /work/anneal
seed=$1
echo "wrapper_header hostname bench was set via unshare --uts"
echo "uts_hostname=$(hostname)"
echo "seed=$seed arm=anneal"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export HOME=/work/home
export TMPDIR=/work/scratch
out=results/lj-board/anneal
mkdir -p "$out" results/lj-board/provenance
exec /usr/lib/linux-tools/6.8.0-149-generic/perf record \
  -F 499 \
  -e cpu-clock \
  -o "$out/lj38-seed${seed}.perf.data" \
  -- /work/bin/lj38_board_anneal "$seed"
