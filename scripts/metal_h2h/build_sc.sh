#!/usr/bin/env bash
# Release build of the Sutton-Chen head-to-head example.
# The potential is rgpot. This build only enables the loader.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/metal-sc
cargo_home=${CARGO_HOME:-$HOME/.cargo}
mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$WORK/bin"
cd "$ROOT"
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --features rgpot-ex --example h2h_sc -j 4
cp -a "$WORK/target/release/examples/h2h_sc" "$WORK/bin/h2h_sc"
chmod +x "$WORK/bin/h2h_sc"
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_sc head=$(cat "$WORK/anneal-head")"
