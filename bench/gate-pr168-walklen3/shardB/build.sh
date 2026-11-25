#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: build.sh STAGE_DIRECTORY" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../../.." && pwd)"
stage="$(mkdir -p "$1" && cd "$1" && pwd)"

a_rev=7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec
a_tree=e5ebda63f9f0117226d50c13abf33243ccf452cc
m_rev=8824564b2479f0917c1e4927cd5ac938a86c2ac0
m_tree=991d5afc6d7b7f7e6c392cb68aa351af1fc50179

examples=(
    --example ase_panel
    --example h2h_metal
    --example h2h_morse
    --example h2h_seed
)

make_tree() {
    local name=$1 revision=$2
    if [ ! -e "$stage/$name-src/.git" ]; then
        git -C "$root" worktree add --detach "$stage/$name-src" "$revision"
    fi
}

verify_tree() {
    local name=$1 revision=$2 tree=$3
    test "$(git -C "$stage/$name-src" rev-parse HEAD)" = "$revision"
    test "$(git -C "$stage/$name-src" rev-parse "HEAD^{tree}")" = "$tree"
    test -z "$(git -C "$stage/$name-src" status --porcelain)"
}

restore_arm() {
    local name=$1
    local source="$stage/$name-src"
    local backup="$stage/$name-backup"
    if [ -f "$backup/lib.rs" ]; then
        cp "$backup/lib.rs" "$source/src/lib.rs"
        cp "$backup/bias.rs" "$source/src/bias.rs"
        cp "$backup/cluster_hopping.rs" \
            "$source/src/methods/cluster_hopping.rs"
    fi
}

restore_all() {
    restore_arm a
    restore_arm a2
    restore_arm m
}

build_arm() {
    local name=$1 revision=$2 tree=$3
    local source="$stage/$name-src"
    local backup="$stage/$name-backup"
    local target="$stage/$name-target"
    mkdir -p "$backup"
    git -C "$source" show HEAD:src/lib.rs > "$backup/lib.rs"
    git -C "$source" show HEAD:src/bias.rs > "$backup/bias.rs"
    git -C "$source" show HEAD:src/methods/cluster_hopping.rs \
        > "$backup/cluster_hopping.rs"
    restore_arm "$name"
    python3 "$script_dir/instrument.py" "$source"
    CARGO_TARGET_DIR="$target" \
        RUSTFLAGS="--remap-path-prefix=$source=/src" \
        cargo +1.88.0 build --manifest-path "$source/Cargo.toml" \
        --release --locked "${examples[@]}"
    restore_arm "$name"
    verify_tree "$name" "$revision" "$tree"
    if ! grep -q \
        'h2h_seed <lj|morse|tip4p> <n> <rho> <budget> <seed> <arm> <stop> <prefix>' \
        "$source/examples/h2h_seed.rs"; then
        echo "$name h2h_seed usage line is missing" >&2
        exit 1
    fi
}

make_tree a "$a_rev"
make_tree a2 "$a_rev"
make_tree m "$m_rev"
trap restore_all EXIT
restore_all
verify_tree a "$a_rev" "$a_tree"
verify_tree a2 "$a_rev" "$a_tree"
verify_tree m "$m_rev" "$m_tree"
build_arm a "$a_rev" "$a_tree"
build_arm a2 "$a_rev" "$a_tree"
build_arm m "$m_rev" "$m_tree"
trap - EXIT

sum_file="$stage/binary-sha256.txt"
: > "$sum_file.tmp"
for arm in a a2 m; do
    for binary in ase_panel h2h_metal h2h_morse h2h_seed; do
        sha256sum "$stage/$arm-target/release/examples/$binary" \
            >> "$sum_file.tmp"
    done
done
mv "$sum_file.tmp" "$sum_file"
rustc +1.88.0 --version > "$stage/toolchain.txt"
