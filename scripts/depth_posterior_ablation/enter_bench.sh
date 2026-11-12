#!/bin/bash
# Depth-posterior ablation from /work, hostname bench.
# The release binary is built with --remap-path-prefix. The search itself
# runs inside one UTS namespace, one seed at a time.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/depth-posterior-ablation
BIN="$WORK/depth_ablation"
INNER="$WORK/campaign.py"
RESULTS="$ROOT/bench/depth-posterior-ablation/results.csv"
LOGS="$WORK/logs"

build_binary() {
  local cargo_home="${CARGO_HOME:-$HOME/.cargo}"
  sudo -n mkdir -p /work/target /work/rustc-tmp "$WORK" "$LOGS"
  sudo -n chown -R ubuntu:ubuntu /work
  (
    cd "$ROOT"
    CARGO_TARGET_DIR=/work/target \
      TMPDIR=/work/rustc-tmp \
      RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo" \
      cargo +1.88.0 build --release --locked --example depth_ablation -j 4
  )
  cp -a /work/target/release/examples/depth_ablation "$BIN"
  cp -a "$HERE/campaign.py" "$INNER"
  chmod +x "$BIN" "$INNER"
}

if [[ "${1:-}" == "--build" ]]; then
  build_binary
  exit 0
fi

if [[ "$(hostname)" != "bench" ]]; then
  build_binary
  if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
    echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
  fi
  exec sudo -n unshare --uts sh -c \
    'hostname bench; exec sudo -u ubuntu env HOME=/work/depth-posterior-ablation/home TMPDIR=/work/depth-posterior-ablation/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_PROC_BIND=TRUE OMP_PLACES=cores RAYON_NUM_THREADS=1 "$@"' \
    _ bash "$HERE/enter_bench.sh" "$@"
fi

mkdir -p "$WORK/home" "$WORK/tmp" "$LOGS"
cd "$WORK"
exec python3 -u "$INNER" \
  --binary "$BIN" \
  --log-dir "$LOGS" \
  --results "$RESULTS" \
  "$@"
