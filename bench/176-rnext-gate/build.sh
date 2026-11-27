#!/usr/bin/env bash
# Copyright (c) anneal developers
# Round-2 release builds for the 176 r-next gate.
# Required: STAGE_DIR. When the B worktree is created, PATCH_MBOX.
# Optional: SEED_STAGE_DIR, a previous stage whose target dirs can be copied.
set -euo pipefail

if [ -z "${STAGE_DIR:-}" ]; then
    echo "STAGE_DIR is required" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_COMMIT="e25221f3a0bf4f78a67d4b7877d38f813b002ef5"
BASE_TREE="cc5c9c3c4062db5e4afaaf4c007cabc8965851cc"
EXPECTED_B_TREE="c9f9914d243a9b362513f49ffae749f4ef980e79"

mkdir -p "$STAGE_DIR"
WT_A="$STAGE_DIR/wt-a"
WT_B="$STAGE_DIR/wt-b"
TARGET_A="$STAGE_DIR/target-a"
TARGET_AP="$STAGE_DIR/target-ap"
TARGET_B="$STAGE_DIR/target-b"
TARGET_PAD="$STAGE_DIR/target-pad"
PAD_ROOT="$STAGE_DIR/pads"

COMMON_GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --git-common-dir)"
mkdir -p "$COMMON_GIT_DIR/info"
if ! grep -qxF "/examples/h2h_blj.rs" "$COMMON_GIT_DIR/info/exclude" 2>/dev/null; then
    printf '\n/examples/h2h_blj.rs\n' >> "$COMMON_GIT_DIR/info/exclude"
fi

if [ ! -e "$WT_A/.git" ]; then
    git -C "$REPO_ROOT" worktree add --detach "$WT_A" "$BASE_COMMIT"
fi
if [ ! -e "$WT_B/.git" ]; then
    if [ -z "${PATCH_MBOX:-}" ]; then
        echo "PATCH_MBOX is required to create the B worktree" >&2
        exit 2
    fi
    git -C "$REPO_ROOT" worktree add --detach "$WT_B" "$BASE_COMMIT"
    git -C "$WT_B" am "$PATCH_MBOX"
fi

restore_sources() {
    git -C "$WT_A" checkout -- src/lib.rs src/bias.rs src/methods/cluster_hopping.rs || true
    git -C "$WT_B" checkout -- src/lib.rs src/bias.rs src/lattice.rs src/methods/cluster_hopping.rs || true
}
trap restore_sources EXIT

restore_sources
if [ "$(git -C "$WT_A" rev-parse 'HEAD^{tree}')" != "$BASE_TREE" ]; then
    echo "A tree mismatch" >&2
    exit 1
fi
if [ "$(git -C "$WT_B" rev-parse 'HEAD^{tree}')" != "$EXPECTED_B_TREE" ]; then
    echo "B tree mismatch" >&2
    exit 1
fi
if [ -n "$(git -C "$WT_A" status --porcelain)" ] || [ -n "$(git -C "$WT_B" status --porcelain)" ]; then
    echo "worktree is not clean before build" >&2
    exit 1
fi

install_driver() {
    local dst="$1/examples/h2h_blj.rs"
    cp "$SCRIPT_DIR/h2h_blj.rs" "${dst}.tmp"
    sync "${dst}.tmp"
    mv "${dst}.tmp" "$dst"
}
install_driver "$WT_A"
install_driver "$WT_B"

seed_target() {
    local name="$1"
    local dest="$STAGE_DIR/$2"
    if [ -n "${SEED_STAGE_DIR:-}" ] && [ ! -d "$dest/release/deps" ] && [ -d "$SEED_STAGE_DIR/$name/release/deps" ]; then
        mkdir -p "$dest"
        cp -a "$SEED_STAGE_DIR/$name/." "$dest/"
    fi
}
seed_target target-a target-a
seed_target target-ap target-ap
seed_target target-b target-b

EXAMPLES=(
    --example h2h_gmin_proof
    --example h2h_blj
    --example h2h_morse
    --example h2h_sc
    --example h2h_metal
    --example h2h_seed
)

cargo_build() {
    local manifest="$1"
    local target="$2"
    local -a cargo_cmd=(cargo)
    if command -v rustup >/dev/null 2>&1 && rustup toolchain list | grep -q '^1.88.0'; then
        cargo_cmd=(cargo "+1.88.0")
    fi
    "${cargo_cmd[@]}" build --manifest-path "$manifest" --release --locked \
        --features rgpot-ex --target-dir "$target" "${EXAMPLES[@]}"
}

publish_bins() {
    local src="$1"
    local dest="$2"
    mkdir -p "$dest"
    local ex
    for ex in h2h_gmin_proof h2h_blj h2h_morse h2h_sc h2h_metal h2h_seed; do
        cp "$src/$ex" "$dest/$ex.tmp"
        sync "$dest/$ex.tmp"
        mv "$dest/$ex.tmp" "$dest/$ex"
    done
}

build_arm() {
    local wt="$1"
    local mode="$2"
    local pad="$3"
    local target="$4"
    local files="$5"
    # shellcheck disable=SC2086
    git -C "$wt" checkout -- $files
    python3 "$SCRIPT_DIR/instrument_arms.py" "$wt" "$mode" "$pad"
    cargo_build "$wt/Cargo.toml" "$target"
    # shellcheck disable=SC2086
    git -C "$wt" checkout -- $files
}

build_arm "$WT_A" A 1 "$TARGET_A" "src/lib.rs src/bias.rs src/methods/cluster_hopping.rs"
build_arm "$WT_A" A 1 "$TARGET_AP" "src/lib.rs src/bias.rs src/methods/cluster_hopping.rs"
build_arm "$WT_B" B 1 "$TARGET_B" "src/lib.rs src/bias.rs src/lattice.rs src/methods/cluster_hopping.rs"

if [ ! -d "$TARGET_PAD/release/deps" ]; then
    mkdir -p "$TARGET_PAD"
    cp -a "$TARGET_A/." "$TARGET_PAD/"
fi
PADS=(4096 8192 16384 32768 65536)
NAMES=(P1 P2 P3 P4 P5)
for idx in 0 1 2 3 4; do
    build_arm "$WT_A" A "${PADS[$idx]}" "$TARGET_PAD" "src/lib.rs src/bias.rs src/methods/cluster_hopping.rs"
    publish_bins "$TARGET_PAD/release/examples" "$PAD_ROOT/${NAMES[$idx]}"
done

if [ -n "$(git -C "$WT_A" status --porcelain)" ] || [ -n "$(git -C "$WT_B" status --porcelain)" ]; then
    echo "worktree is not clean after build" >&2
    exit 1
fi

SUMS="$STAGE_DIR/SHA256SUMS"
: > "${SUMS}.tmp"
for dir in "$TARGET_A/release/examples" "$TARGET_AP/release/examples" "$TARGET_B/release/examples" \
    "$PAD_ROOT/P1" "$PAD_ROOT/P2" "$PAD_ROOT/P3" "$PAD_ROOT/P4" "$PAD_ROOT/P5"; do
    for ex in h2h_gmin_proof h2h_blj h2h_morse h2h_sc h2h_metal h2h_seed; do
        sha256sum "$dir/$ex" >> "${SUMS}.tmp"
    done
done
sync "${SUMS}.tmp"
mv "${SUMS}.tmp" "$SUMS"
sha256sum "$SUMS"
