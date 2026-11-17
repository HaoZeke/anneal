#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release build of the h2h_abcluster example. Paths in the binary are
# remapped so the timed run is not labelled with this checkout. The code
# the rows name is the code that was built, so the build refuses a tree
# whose source or runner differs from HEAD.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/abcluster-h2h
if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
cargo_home=${CARGO_HOME:-$HOME/.cargo}
mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$WORK/bin"
cd "$ROOT"
if [[ -n "$(git status --porcelain -- src examples scripts/abcluster_h2h Cargo.toml Cargo.lock)" ]]; then
  echo "the source or the runner differs from HEAD; commit it first" >&2
  exit 1
fi
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --example h2h_abcluster -j 4
cp -a "$WORK/target/release/examples/h2h_abcluster" "$WORK/bin/h2h_abcluster"
chmod +x "$WORK/bin/h2h_abcluster"
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_abcluster head=$(cat "$WORK/anneal-head")"
