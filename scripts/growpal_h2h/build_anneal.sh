#!/usr/bin/env bash
# Release build of the GMIN head-to-head example.
# Its paper_budget map is the shared evaluation ceiling. Paths in the
# binary are remapped so the timed run is not labelled with this checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/growpal-h2h
if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
cargo_home=${CARGO_HOME:-$HOME/.cargo}
mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$WORK/bin"
cd "$ROOT"
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --example h2h_gmin_proof -j 4
cp -a "$WORK/target/release/examples/h2h_gmin_proof" "$WORK/bin/h2h_gmin_proof"
chmod +x "$WORK/bin/h2h_gmin_proof"
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_gmin_proof head=$(cat "$WORK/anneal-head")"
