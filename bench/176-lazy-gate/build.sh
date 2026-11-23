#!/usr/bin/env bash
# Release-build the four arms into separate target directories.
set -euo pipefail

: "${STAGE_DIR:?STAGE_DIR is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_COMMIT="364679f060386949f805a72dea9d7d63b7e9ea4f"
BASE_TREE="79610c980d39f19a2d383ffe3c53e4505ddb9b6c"
PR_HEAD="6b12b9aab1e78589f0634413aae8c12611104607"
LAZY_COMMIT="5bf74404ebc21751e95ac3d5efe1d74f3f58da04"
LAZY_TREE="0467d399182a809d06661f1ae1903553833b1e8e"
B_TREE="9d40e137e4f20cd0a8401b40b0b2f06f7521716e"
B0_TREE="be6101476d3999e9232e71016b7871c17ca52a4d"

WT_A="$STAGE_DIR/wt-a"
WT_B="$STAGE_DIR/wt-b"
WT_B0="$STAGE_DIR/wt-b0"
TARGET_A="$STAGE_DIR/target-a"
TARGET_A2="$STAGE_DIR/target-a2"
TARGET_B="$STAGE_DIR/target-b"
TARGET_B0="$STAGE_DIR/target-b0"
DRIVERS=(h2h_gmin_proof h2h_morse h2h_seed)

mkdir -p "$STAGE_DIR"

if [ ! -e "$WT_A/.git" ]; then
    git -C "$REPO_ROOT" worktree add --detach "$WT_A" "$BASE_COMMIT"
fi
if [ ! -e "$WT_B/.git" ]; then
    git -C "$REPO_ROOT" worktree add --detach "$WT_B" "$BASE_COMMIT"
    git -C "$WT_B" merge --no-ff --no-edit "$LAZY_COMMIT"
fi
if [ ! -e "$WT_B0/.git" ]; then
    git -C "$REPO_ROOT" worktree add --detach "$WT_B0" "$BASE_COMMIT"
    git -C "$WT_B0" merge --no-ff --no-edit "$PR_HEAD"
fi

[ "$(git -C "$WT_A" rev-parse 'HEAD^{tree}')" = "$BASE_TREE" ]
[ "$(git -C "$WT_B" rev-parse 'HEAD^{tree}')" = "$B_TREE" ]
[ "$(git -C "$WT_B0" rev-parse 'HEAD^{tree}')" = "$B0_TREE" ]
[ "$(git -C "$REPO_ROOT" rev-parse "${LAZY_COMMIT}^{tree}")" = "$LAZY_TREE" ]

restore_one() {
    local wt="$1"
    git -C "$wt" checkout -- \
        src/lib.rs src/bias.rs src/lattice.rs \
        src/methods/cluster_hopping.rs \
        examples/h2h_gmin_proof.rs examples/h2h_morse.rs examples/h2h_seed.rs
    find "$wt/src" "$wt/examples" -name '*.tmp' -print0 |
        while IFS= read -r -d '' leftover; do
            mv "$leftover" "$STAGE_DIR/$(basename "$leftover").aside"
        done
}

restore_all() {
    restore_one "$WT_A" || true
    restore_one "$WT_B" || true
    restore_one "$WT_B0" || true
}
trap restore_all EXIT
restore_all

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1 GOTO_NUM_THREADS=1 TBB_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1 POLARS_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE

binaries_ready() {
    local target="$1"
    local driver
    for driver in "${DRIVERS[@]}"; do
        [ -x "$target/release/examples/$driver" ] || return 1
    done
}

build_arm() {
    local _name="$1"
    local wt="$2"
    local target="$3"
    local mode="$4"
    local expect="$5"
    [ "$(git -C "$wt" rev-parse 'HEAD^{tree}')" = "$expect" ]
    [ -z "$(git -C "$wt" status --porcelain)" ]
    if binaries_ready "$target"; then
        return 0
    fi
    python3 "$SCRIPT_DIR/instrument_arms.py" "$wt" "$mode" 1
    cargo +1.88.0 build --manifest-path "$wt/Cargo.toml" \
        --release --locked --features rgpot-ex \
        --target-dir "$target" \
        --example h2h_gmin_proof \
        --example h2h_morse \
        --example h2h_seed
    restore_one "$wt"
    [ -z "$(git -C "$wt" status --porcelain)" ]
    [ "$(git -C "$wt" rev-parse 'HEAD^{tree}')" = "$expect" ]
    binaries_ready "$target"
}

build_arm A "$WT_A" "$TARGET_A" A "$BASE_TREE"
build_arm A2 "$WT_A" "$TARGET_A2" A "$BASE_TREE"
build_arm B "$WT_B" "$TARGET_B" B "$B_TREE"
build_arm B0 "$WT_B0" "$TARGET_B0" B "$B0_TREE"

SUMS="$STAGE_DIR/SHA256SUMS"
: > "$SUMS.tmp"
for target in "$TARGET_A" "$TARGET_A2" "$TARGET_B" "$TARGET_B0"; do
    for driver in "${DRIVERS[@]}"; do
        sha256sum "$target/release/examples/$driver" >> "$SUMS.tmp"
    done
done
sync "$SUMS.tmp"
mv "$SUMS.tmp" "$SUMS"
sha256sum "$SUMS"
