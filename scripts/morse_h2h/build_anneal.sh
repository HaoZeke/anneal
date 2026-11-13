#!/usr/bin/env bash
# Release build of the Morse head-to-head example.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/morse-h2h
cargo_home=${CARGO_HOME:-$HOME/.cargo}
mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$WORK/bin"
cd "$ROOT"
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --offline --example h2h_morse -j 4
cp -a "$WORK/target/release/examples/h2h_morse" "$WORK/bin/h2h_morse"
chmod +x "$WORK/bin/h2h_morse"
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_morse head=$(cat "$WORK/anneal-head")"
