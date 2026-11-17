#!/usr/bin/env bash
# Copyright (c) anneal developers

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
base=0ae5bfe0efb874a718f8e9632d18ff5fbc29f782
head=$(git -C "$root" rev-parse HEAD)
work=${OUTSIDE_PERF_WORK:-/work/outside-perf}
base_tree=$work/base-tree
base_target=$work/base-target
new_target=$work/new-target
rows=$root/bench/outside-perf/rows.csv

mkdir -p "$work"
if [[ ! -e $base_tree/.git ]]; then
    rm -rf "$base_tree"
    git -C "$root" worktree add --detach "$base_tree" "$base"
fi
git -C "$base_tree" rev-parse HEAD | grep -qx "$base"

CARGO_TARGET_DIR=$base_target cargo build --manifest-path "$base_tree/Cargo.toml" \
    --release --locked --example h2h_gmin_proof --example h2h_morse --example h2h_metal
CARGO_TARGET_DIR=$new_target cargo build --manifest-path "$root/Cargo.toml" \
    --release --locked --example h2h_gmin_proof --example h2h_morse --example h2h_metal

sudo unshare --uts bash -c "
set -euo pipefail
hostname bench
cd '$root'
python3 scripts/outside_perf/run_interleaved.py \
  --base-bin '$base_target/release/examples' \
  --new-bin '$new_target/release/examples' \
  --base-revision '$base' \
  --new-revision '$head' \
  --rows '$rows'
"
python3 "$root/scripts/outside_perf/summarize.py" \
    --rows "$rows" \
    --summary "$root/bench/outside-perf/summary.csv" \
    --trajectory-diff "$root/bench/outside-perf/trajectory-diff.csv"
