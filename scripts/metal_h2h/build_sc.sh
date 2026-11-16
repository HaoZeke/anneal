#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release builds of the Sutton-Chen head-to-head examples, h2h_sc and sc_h2h.
# The h2h_sc potential is rgpot. Its build only enables the loader.
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
cargo +1.88.0 build --release --locked --example sc_h2h -j 4
for example in h2h_sc sc_h2h; do
  cp -a "$WORK/target/release/examples/$example" "$WORK/bin/$example"
  chmod +x "$WORK/bin/$example"
done
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_sc head=$(cat "$WORK/anneal-head")"
