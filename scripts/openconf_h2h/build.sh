#!/usr/bin/env bash
# Copyright (c) anneal developers
# The anneal wheel, the RDKit call counter, and CREST 3.0.2 with a timer
# around its potential call. Everything lands in /work/openconf-h2h/bin.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PREFIX=${OPENCONF_H2H_ENV:-/work/env}
BIN=/work/openconf-h2h/bin
CREST_SRC=${CREST_SRC:-/work/crest-src}
mkdir -p "$BIN"

bash "$HERE/setup_env.sh"
export PATH=$PREFIX/bin:$PATH

(
  cd "$ROOT"
  CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/work/target-py} RUSTUP_TOOLCHAIN=1.88.0 \
    maturin build --release --locked -i "$PREFIX/bin/python" -j 4
  "$PREFIX/bin/pip" install -q --force-reinstall --no-deps \
    "${CARGO_TARGET_DIR:-/work/target-py}"/wheels/anneal-*.whl
  git rev-parse HEAD > "$BIN/anneal-head"
)

gcc -O2 -shared -fPIC -o "$BIN/librdff_count.so" "$HERE/rdff_count.c" -ldl

if [[ ! -x $CREST_SRC/_build/crest ]]; then
  rm -rf "$CREST_SRC"
  git clone -q --depth 1 --branch v3.0.2 https://github.com/crest-lab/crest.git "$CREST_SRC"
  git -C "$CREST_SRC" submodule update --init --depth 1
  git -C "$CREST_SRC" apply "$HERE/crest-engrad-time.patch"
  # GCC 15 defaults to C23, where symmetry_i.c's true() is a keyword.
  CC=$PREFIX/bin/x86_64-conda-linux-gnu-cc FC=$PREFIX/bin/gfortran CMAKE_PREFIX_PATH=$PREFIX \
    cmake -S "$CREST_SRC" -B "$CREST_SRC/_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS=-std=gnu17
  ninja -C "$CREST_SRC/_build" -j 4
fi
cp "$CREST_SRC/_build/crest" "$BIN/crest"
