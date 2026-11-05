#!/usr/bin/env bash
# Time one weak-scaling cell and write a raw log.
# Usage: weak_scale_cell.sh <family> <workers> <seed> <log>
set -euo pipefail
family=${1:?family}
workers=${2:?workers}
seed=${3:?seed}
log=${4:?log}
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
target_dir=${CARGO_TARGET_DIR:-$root/target}
bin=$target_dir/release/examples/weak_scale
mkdir -p "$(dirname -- "$log")"
if [[ ! -x $bin ]]; then
  echo "missing binary $bin" >&2
  exit 2
fi
cmd="target/release/examples/weak_scale $family $workers $seed"
tmp=$(mktemp)
set +e
"$bin" "$family" "$workers" "$seed" >"$tmp"
ec=$?
set -e
cpu=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1 | sed 's/(R)//g; s/(r)//g')
# Snakemake exports OMP_NUM_THREADS, and nproc honors that export.
cores=$(env -u OMP_NUM_THREADS -u GOTO_NUM_THREADS -u OPENBLAS_NUM_THREADS -u MKL_NUM_THREADS -u VECLIB_MAXIMUM_THREADS -u NUMEXPR_NUM_THREADS nproc)
{
  echo "HEAD: $(git -C "$root" rev-parse HEAD)"
  echo "tree: $(git -C "$root" rev-parse 'HEAD^{tree}')"
  echo "command: $cmd"
  echo "cpu_model: $cpu"
  echo "cores: $cores"
  echo "gpu: none"
  echo "cuda: none"
  cat "$tmp"
  echo "exit: $ec"
} >"$log"
rm -f "$tmp"
exit "$ec"
