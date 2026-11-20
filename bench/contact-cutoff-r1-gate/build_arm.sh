#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release build of the gate harness against one arm's checkout.
#
#   build_arm.sh <arm checkout> <build directory>
#
# The harness compiles the campaign drivers from <arm checkout>/examples
# against the anneal-core of that checkout. Its lock file starts as the
# checkout's Cargo.lock, so every shared package keeps the checkout's pin.
set -euo pipefail

TREE=$(realpath "${1:?arm checkout}")
WORK=$(realpath -m "${2:?build directory}")
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ -e "$WORK" ]]; then
  echo "$WORK exists; a build is never reused" >&2
  exit 2
fi
mkdir -p "$WORK"
cp -a "$HERE/harness" "$WORK/harness"
ln -s "$TREE" "$WORK/anneal"
cp "$TREE/Cargo.lock" "$WORK/harness/Cargo.lock"
cp "$TREE/Cargo.lock" "$WORK/arm.lock"
cargo_home=${CARGO_HOME:-$HOME/.cargo}
export CARGO_TARGET_DIR=$WORK/target
export RUSTFLAGS="--remap-path-prefix=${TREE}=/src --remap-path-prefix=${WORK}=/build --remap-path-prefix=${cargo_home}=/cargo"
cd "$WORK/harness"
cargo +1.88.0 build --release --bins -j 4
mkdir -p "$WORK/bin"
for driver in h2h_gmin_proof h2h_sc h2h_metal h2h_morse; do
  cp -a "$CARGO_TARGET_DIR/release/$driver" "$WORK/bin/$driver"
done
python3 "$HERE/lock_check.py" "$WORK/arm.lock" "$WORK/harness/Cargo.lock"
git -C "$TREE" rev-parse 'HEAD^{tree}' > "$WORK/tree"
(cd "$WORK/bin" && sha256sum h2h_gmin_proof h2h_sc h2h_metal h2h_morse) > "$WORK/bin-sha256.txt"
echo "tree=$(cat "$WORK/tree")"
cat "$WORK/bin-sha256.txt"
