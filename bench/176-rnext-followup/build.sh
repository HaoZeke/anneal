#!/usr/bin/env bash
# Copyright (c) anneal developers
set -euo pipefail

: "${STAGE_DIR:?STAGE_DIR is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_COMMIT="e25221f3a0bf4f78a67d4b7877d38f813b002ef5"
PR_HEAD="6b12b9aab1e78589f0634413aae8c12611104607"
BASE_TREE="cc5c9c3c4062db5e4afaaf4c007cabc8965851cc"
B_TREE="c9f9914d243a9b362513f49ffae749f4ef980e79"
WT_A="$STAGE_DIR/wt-a"
WT_B="$STAGE_DIR/wt-b"
TARGET_A="$STAGE_DIR/target-a"
TARGET_A2="$STAGE_DIR/target-a2"
TARGET_B="$STAGE_DIR/target-b"

mkdir -p "$STAGE_DIR"
if [ ! -e "$WT_A/.git" ]; then
    git -C "$REPO_ROOT" worktree add --detach "$WT_A" "$BASE_COMMIT"
fi
if [ ! -e "$WT_B/.git" ]; then
    git -C "$REPO_ROOT" worktree add --detach "$WT_B" "$BASE_COMMIT"
    git -C "$WT_B" merge --no-ff --no-edit "$PR_HEAD"
fi

restore_sources() {
    git -C "$WT_A" checkout -- src/lib.rs src/bias.rs \
        src/methods/cluster_hopping.rs || true
    git -C "$WT_B" checkout -- src/lib.rs src/bias.rs src/lattice.rs \
        src/methods/cluster_hopping.rs || true
}
trap restore_sources EXIT
restore_sources

[ "$(git -C "$WT_A" rev-parse 'HEAD^{tree}')" = "$BASE_TREE" ]
[ "$(git -C "$WT_B" rev-parse 'HEAD^{tree}')" = "$B_TREE" ]
[ -z "$(git -C "$WT_A" status --porcelain)" ]
[ -z "$(git -C "$WT_B" status --porcelain)" ]

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1 GOTO_NUM_THREADS=1 TBB_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1 POLARS_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cargo_build() {
    local wt="$1"
    local target="$2"
    local -a cargo_cmd=(cargo)
    if command -v rustup >/dev/null 2>&1 && \
            rustup toolchain list | grep -q '^1.88.0'; then
        cargo_cmd=(cargo "+1.88.0")
    fi
    "${cargo_cmd[@]}" build --manifest-path "$wt/Cargo.toml" \
        --release --locked --features rgpot-ex --target-dir "$target" \
        --example h2h_gmin_proof --example h2h_seed
}

build_base() {
    python3 "$SCRIPT_DIR/instrument_arms.py" "$WT_A" A 1
    cargo_build "$WT_A" "$TARGET_A"
    cargo_build "$WT_A" "$TARGET_A2"
    restore_sources
}

build_change() {
    python3 "$SCRIPT_DIR/instrument_arms.py" "$WT_B" B 1
    cargo_build "$WT_B" "$TARGET_B"
    restore_sources
}

build_base
build_change
[ -z "$(git -C "$WT_A" status --porcelain)" ]
[ -z "$(git -C "$WT_B" status --porcelain)" ]

SUMS="$STAGE_DIR/SHA256SUMS"
: > "$SUMS.tmp"
for target in "$TARGET_A" "$TARGET_A2" "$TARGET_B"; do
    for driver in h2h_gmin_proof h2h_seed; do
        sha256sum "$target/release/examples/$driver" >> "$SUMS.tmp"
    done
done
sync "$SUMS.tmp"
mv "$SUMS.tmp" "$SUMS"
sha256sum "$SUMS"
