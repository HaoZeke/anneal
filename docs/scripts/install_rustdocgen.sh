#!/bin/sh
# Install sphinx-rustdocgen 1.3.0 with the module-path fix.
# `#[path]` is resolved from the directory of the file that contains it.
# The published 1.3.0 binary joins that attribute onto the file itself.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
prefix=${1:-"$root/.pixi/envs/docs"}
stamp="$prefix/share/sphinx-rustdocgen-module-path-1.3.0"
if [ -x "$prefix/bin/sphinx-rustdocgen" ] && [ -f "$stamp" ]; then
    exit 0
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -fsSL -o "$work/crate.crate" \
    https://static.crates.io/crates/sphinx-rustdocgen/sphinx-rustdocgen-1.3.0.crate
tar -xzf "$work/crate.crate" -C "$work"
patch -d "$work/sphinx-rustdocgen-1.3.0" -p1 \
    < "$root/docs/patches/sphinx-rustdocgen-1.3.0-module-path.patch"
cargo install --locked --path "$work/sphinx-rustdocgen-1.3.0" --root "$prefix" --force
mkdir -p "$(dirname "$stamp")"
: > "$stamp"
