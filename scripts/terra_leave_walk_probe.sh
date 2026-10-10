#!/usr/bin/env bash
# Shell+two-phase and a 32-hop packing walk from the sealed LJ75 ico well.
set -euo pipefail
export PATH="/usr/bin:${HOME}/.cargo/bin:${PATH}"
ROOT="${1:-${HOME}/Git/Github/Rust/anneal}"
cd "$ROOT"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/leave_walk_probe_wrapper.log"
exec > >(tee -a "${LOG}") 2>&1
echo "commit=$(git rev-parse HEAD)"
echo "pwd=$(pwd)"
cargo test --lib leave_av_walk -- --nocapture
cargo build --example leave_packing_probe --release
./target/release/examples/leave_packing_probe 8 600 walk
echo "EXIT:$?"
