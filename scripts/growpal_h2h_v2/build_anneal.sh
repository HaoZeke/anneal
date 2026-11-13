#!/usr/bin/env bash
# Release builds of the Lennard-Jones and Morse head-to-head examples.
# Their paper_budget maps are the shared call ceilings. Paths in the
# binaries are remapped so the timed run is not labelled with this checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/growpal-h2h-v2
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
cargo +1.88.0 build --release --locked --example h2h_gmin_proof --example h2h_morse -j 4
for name in h2h_gmin_proof h2h_morse; do
  cp -a "$WORK/target/release/examples/$name" "$WORK/bin/$name"
  chmod +x "$WORK/bin/$name"
done
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binaries=$WORK/bin/{h2h_gmin_proof,h2h_morse} head=$(cat "$WORK/anneal-head")"
