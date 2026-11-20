#!/usr/bin/env bash
# Copyright (c) anneal developers

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
develop=61fa67cf380e46876b569da45e7cc091647a775b
develop_tree_id=cff796c507a8fa1d0bfb28888c3570b61988055e
head=$(git -C "$root" rev-parse HEAD)
work=${OUTSIDE_PERF_WORK:-$root/target/outside-perf-r2}
develop_tree=$work/develop-tree
develop_target=$work/develop-target
develop_prime_target=$work/develop-prime-target
new_target=$work/new-target
rows=$root/bench/outside-perf/rows.csv
examples=(h2h_gmin_proof h2h_morse h2h_metal h2h_blj h2h_seed)

mkdir -p "$work"
if [[ ! -e $develop_tree/.git ]]; then
    rm -rf "$develop_tree"
    git -C "$root" worktree add --detach "$develop_tree" "$develop"
fi
git -C "$develop_tree" rev-parse HEAD | grep -qx "$develop"
git -C "$develop_tree" rev-parse 'HEAD^{tree}' | grep -qx "$develop_tree_id"
cp "$root/examples/h2h_blj.rs" "$develop_tree/examples/h2h_blj.rs"

example_args=()
for example in "${examples[@]}"; do
    example_args+=(--example "$example")
done
CARGO_TARGET_DIR=$develop_target cargo build --manifest-path "$develop_tree/Cargo.toml" \
    --release --locked "${example_args[@]}"
CARGO_TARGET_DIR=$develop_prime_target cargo build --manifest-path "$develop_tree/Cargo.toml" \
    --release --locked "${example_args[@]}"
CARGO_TARGET_DIR=$new_target cargo build --manifest-path "$root/Cargo.toml" \
    --release --locked "${example_args[@]}"

taskset -c 0 python3 "$root/scripts/outside_perf/run_interleaved.py" \
    --develop-bin "$develop_target/release/examples" \
    --develop-prime-bin "$develop_prime_target/release/examples" \
    --new-bin "$new_target/release/examples" \
    --develop-revision "$develop" \
    --new-revision "$head" \
    --rows "$rows"
python3 "$root/scripts/outside_perf/summarize.py" \
    --rows "$rows" \
    --summary "$root/bench/outside-perf/summary.csv" \
    --trajectory-diff "$root/bench/outside-perf/trajectory-diff.csv"
