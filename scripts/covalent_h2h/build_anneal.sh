#!/usr/bin/env bash
# Release build of the covalent head-to-head example and its rgpot library.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/covalent-h2h
RGPOT=${RGPOT_ROOT:-/work/rgpot}
cargo_home=${CARGO_HOME:-$HOME/.cargo}
mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$WORK/bin" "$WORK/lib"
bash "$HERE/build_rgpot_covalent.sh" "$RGPOT" "$WORK/lib/librgpot_covalent.so"
cd "$ROOT"
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --offline --features rgpot-ex --example h2h_covalent -j 4
cp -a "$WORK/target/release/examples/h2h_covalent" "$WORK/bin/h2h_covalent"
chmod +x "$WORK/bin/h2h_covalent"
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_covalent head=$(cat "$WORK/anneal-head")"
