#!/usr/bin/env bash
# Measure occupancy-general Leave starts on the sealed LJ75 ico well.
# SHS, farthest-point cover, SC-AFIR push/peel, AV-restricted rungs.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export PATH="/usr/bin:${HOME}/.cargo/bin:${PATH}"
ROOT="${1:-${HOME}/Git/Github/Rust/anneal}"
cd "$ROOT"
echo "commit=$(git rev-parse HEAD)"
echo "pwd=$(pwd)"
srun --partition=cpu --time=00:40:00 --mem=8G --cpus-per-task=8 \
  --job-name=leave-av-test \
  cargo test --lib afir_ -- --nocapture
srun --partition=cpu --time=00:40:00 --mem=8G --cpus-per-task=8 \
  --job-name=leave-av-ex \
  cargo build --example leave_packing_probe --release
mkdir -p "${ROOT}/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="${ROOT}/logs/leave_av_probe_8_600_${STAMP}.log"
echo "log=${LOG}"
srun --partition=cpu --time=02:00:00 --mem=8G --cpus-per-task=4 \
  --job-name=leave-av-probe \
  ./target/release/examples/leave_packing_probe 8 600 av | tee "${LOG}"
