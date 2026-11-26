#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: build.sh STAGE_DIRECTORY [arm...]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../../.." && pwd)"
stage="$(mkdir -p "$1" && cd "$1" && pwd)"
shift

a_rev=7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec
a_tree=e5ebda63f9f0117226d50c13abf33243ccf452cc
m_rev=8824564b2479f0917c1e4927cd5ac938a86c2ac0
m_tree=991d5afc6d7b7f7e6c392cb68aa351af1fc50179
o_rev=c96807c54e8cc602322255c75be61425c7bf722a
o_tree=7f2625597a167d3c7841b2266eec844997e53266

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
    local name
    for name in "${wanted[@]}"; do
        restore_arm "$name"
    done
}

revision_of() {
    case "$1" in
        a|a2) printf '%s\n' "$a_rev" ;;
        m) printf '%s\n' "$m_rev" ;;
        o) printf '%s\n' "$o_rev" ;;
        *) echo "unknown arm $1" >&2; exit 2 ;;
    esac
}

tree_of() {
    case "$1" in
        a|a2) printf '%s\n' "$a_tree" ;;
        m) printf '%s\n' "$m_tree" ;;
        o) printf '%s\n' "$o_tree" ;;
        *) echo "unknown arm $1" >&2; exit 2 ;;
    esac
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

wanted=("$@")
if [ "${#wanted[@]}" -eq 0 ]; then
    wanted=(a a2 m o)
fi
for name in "${wanted[@]}"; do
    make_tree "$name" "$(revision_of "$name")"
done
trap restore_all EXIT
restore_all
for name in "${wanted[@]}"; do
    verify_tree "$name" "$(revision_of "$name")" "$(tree_of "$name")"
    build_arm "$name" "$(revision_of "$name")" "$(tree_of "$name")"
done
trap - EXIT

sum_file="$stage/binary-sha256.txt"
: > "$sum_file.tmp"
for arm in a a2 m o; do
    for binary in ase_panel h2h_metal h2h_morse h2h_seed; do
        path="$stage/$arm-target/release/examples/$binary"
        if [ -f "$path" ]; then
            sha256sum "$path" >> "$sum_file.tmp"
        fi
    done
done
mv "$sum_file.tmp" "$sum_file"
rustc +1.88.0 --version > "$stage/toolchain.txt"
