#!/usr/bin/env bash
# Raw-quench leftover hollow / fill / surface / shell from the sealed ico well.
set -euo pipefail
export PATH="/usr/bin:${HOME}/.cargo/bin:${PATH}"
ROOT="${1:-${HOME}/Git/Github/Rust/anneal}"
cd "$ROOT"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/leave_pack_probe_wrapper.log"
exec > >(tee -a "${LOG}") 2>&1
echo "commit=$(git rev-parse HEAD)"
echo "pwd=$(pwd)"
cargo test --lib leave_av_surface -- --nocapture
cargo build --example leave_packing_probe --release
./target/release/examples/leave_packing_probe 8 600 pack
echo "EXIT:$?"
