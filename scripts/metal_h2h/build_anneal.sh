#!/usr/bin/env bash
# Release build of the Gupta head-to-head example.
# Paths in the binary are remapped so the timed run is not labelled
# with this checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/metal-h2h
cargo_home=${CARGO_HOME:-$HOME/.cargo}
mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$WORK/bin"
cd "$ROOT"
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --example h2h_metal -j 4
cp -a "$WORK/target/release/examples/h2h_metal" "$WORK/bin/h2h_metal"
chmod +x "$WORK/bin/h2h_metal"
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_metal head=$(cat "$WORK/anneal-head")"
