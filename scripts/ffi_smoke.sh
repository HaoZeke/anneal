#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build the C ABI, generate its header, compile tests/ffi_smoke.c against
# both and run it. The ffi job in .github/workflows/ci.yml does the same
# against the cargo-c install. Needs cargo, cbindgen, a C compiler and
# python3.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cargo build --features capi --lib
target=$(cargo metadata --format-version 1 --no-deps |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')

mkdir -p "$work/include/dlpack"
cp include/dlpack/dlpack.h "$work/include/dlpack/"
cbindgen --config cbindgen.toml --crate anneal-core \
    --output "$work/include/anneal-core.h" 2>/dev/null

"${CC:-cc}" -std=c99 -Wall -Wextra -Werror -I"$work/include" \
    tests/ffi_smoke.c \
    -L"$target/debug" -Wl,-rpath,"$target/debug" -lanneal_core -lm \
    -o "$work/ffi_smoke"
"$work/ffi_smoke"
