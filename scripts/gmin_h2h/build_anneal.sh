#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release build of the existing head-to-head example.
# Paths in the binary are remapped so the timed run is not labelled
# with this checkout. ANNEAL_REV, when set, builds that revision from a
# detached worktree under /work instead of this checkout's HEAD.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/gmin-h2h
cargo_home=${CARGO_HOME:-$HOME/.cargo}
SRC=$ROOT
if [ -n "${ANNEAL_REV:-}" ]; then
  SRC=$WORK/anneal-src
  if [ ! -d "$SRC" ]; then
    git -C "$ROOT" worktree add --detach "$SRC" "$ANNEAL_REV"
  fi
  git -C "$SRC" checkout --detach -q "$ANNEAL_REV"
  if [ -n "$(git -C "$SRC" status --porcelain)" ]; then
    echo "$SRC differs from $ANNEAL_REV" >&2
    exit 1
  fi
fi
mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$WORK/bin"
cd "$SRC"
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${SRC}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --example h2h_gmin_proof -j 4
cp -a "$WORK/target/release/examples/h2h_gmin_proof" "$WORK/bin/h2h_gmin_proof"
chmod +x "$WORK/bin/h2h_gmin_proof"
printf '%s\n' "$(git rev-parse HEAD)" > "$WORK/anneal-head"
echo "anneal binary=$WORK/bin/h2h_gmin_proof head=$(cat "$WORK/anneal-head")"
