#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build the two arms of the panel from one set of sources and one set of
# flags.
#
#   PANEL_A_REV=<commit> PANEL_B_REV=<commit> build_arms.sh [A] [B]
#
# Each arm is a detached worktree of this repository at the named commit.
# The binary Lennard-Jones driver in this directory is copied into the arm's
# `examples` directory, untracked, so one source drives both revisions. The
# other drivers are the ones each revision carries. Every arm is built with
# the same toolchain, the same flags and the same feature sets, and writes a
# manifest of revision, tree, compiler and binary hashes beside its binaries.
#
#   PANEL_WORK       scratch directory, default .bench/blj-panel in the repo
#   PANEL_TOOLCHAIN  rustup toolchain, default 1.88.0
#   PANEL_JOBS       cargo -j, default 4
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${PANEL_WORK:-$ROOT/.bench/blj-panel}
TOOLCHAIN=${PANEL_TOOLCHAIN:-1.88.0}
JOBS=${PANEL_JOBS:-4}
cargo_home=${CARGO_HOME:-$HOME/.cargo}

plain_examples="--example h2h_blj --example h2h_gmin_proof --example h2h_morse"
plain_bins="h2h_blj h2h_gmin_proof h2h_morse"
sc_bins="h2h_sc"

build_arm() {
  local arm=$1 lower rev_var rev src target bin
  lower=$(printf '%s' "$arm" | tr '[:upper:]' '[:lower:]')
  rev_var=PANEL_${arm}_REV
  rev=${!rev_var:?set $rev_var to the commit of arm $arm}
  rev=$(git -C "$ROOT" rev-parse --verify "${rev}^{commit}")
  src=$WORK/arm-$lower
  target=$WORK/target-$lower
  bin=$WORK/bin-$lower
  mkdir -p "$WORK" "$target" "$bin" "$WORK/rustc-tmp"

  if [ ! -d "$src" ]; then
    git -C "$ROOT" worktree add --detach "$src" "$rev"
  fi
  git -C "$src" checkout --detach -q "$rev"
  if [ -n "$(git -C "$src" status --porcelain --untracked-files=no)" ]; then
    echo "arm $arm differs from $rev" >&2
    exit 1
  fi
  cp -f "$HERE/h2h_blj.rs" "$src/examples/h2h_blj.rs"

  (
    cd "$src"
    export CARGO_TARGET_DIR=$target
    export TMPDIR=$WORK/rustc-tmp
    export RUSTFLAGS="--remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo --remap-path-prefix=${src}=/src"
    # shellcheck disable=SC2086
    cargo "+$TOOLCHAIN" build --release --locked -j "$JOBS" $plain_examples
    for name in $plain_bins; do
      cp -f "$target/release/examples/$name" "$bin/$name"
    done
    cargo "+$TOOLCHAIN" build --release --locked -j "$JOBS" --features rgpot-ex --example h2h_sc
    for name in $sc_bins; do
      cp -f "$target/release/examples/$name" "$bin/$name"
    done
  )

  {
    echo "arm=$arm"
    echo "revision=$rev"
    echo "tree=$(git -C "$src" rev-parse "${rev}^{tree}")"
    echo "toolchain=$TOOLCHAIN"
    echo "rustc=$(rustc "+$TOOLCHAIN" -V)"
    echo "cargo=$(cargo "+$TOOLCHAIN" -V)"
    echo "profile=release locked"
    echo "driver_h2h_blj=$(sha256sum "$HERE/h2h_blj.rs" | cut -d' ' -f1)"
  } > "$bin/MANIFEST"
  (cd "$bin" && sha256sum $plain_bins $sc_bins > SHA256SUMS)
  echo "arm $arm revision=$rev built"
}

arms=("$@")
if [ "${#arms[@]}" -eq 0 ]; then
  arms=(A B)
fi
for arm in "${arms[@]}"; do
  build_arm "$arm"
done
