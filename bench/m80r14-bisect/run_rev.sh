#!/usr/bin/env bash
# Copyright (c) anneal developers
# Usage: run_rev.sh <rev> <system> [patch]
#
# Builds the paper driver at <rev> in a detached worktree, with [patch]
# applied when given, and runs seeds 2000-2011 of <system> at the
# driver's own paper budget. Each seed is one single-threaded process;
# JOBS of them run at once (default 8). The call counts and energies do
# not depend on JOBS, the wall times do. Rows go to $OUT (default
# ./m80r14-bisect-out), one file per seed and one sorted CSV per run.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
rev=$(git -C "$repo" rev-parse --short=8 "$1")
sys=$2
patch=${3:-}
jobs=${JOBS:-8}
out=${OUT:-$PWD/m80r14-bisect-out}
case $sys in
lj*) example=h2h_gmin_proof ;;
*) example=h2h_morse ;;
esac
label=$rev
if [ -n "$patch" ]; then
	patch=$(cd "$(dirname "$patch")" && pwd)/$(basename "$patch")
	label=$rev-$(basename "$patch" .patch)
fi
mkdir -p "$out/bin"
bin=$out/bin/$example-$label
if [ ! -x "$bin" ]; then
	wt=$(mktemp -d)
	git -C "$repo" worktree add --detach "$wt" "$rev" >/dev/null
	if [ -n "$patch" ]; then git -C "$wt" apply "$patch"; fi
	(cd "$wt" && CARGO_TARGET_DIR=$out/target cargo build --release --example "$example") \
		>"$out/build-$label.log" 2>&1
	cp "$out/target/release/examples/$example" "$bin"
	git -C "$repo" worktree remove --force "$wt"
fi
seq 2000 2011 | RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 xargs -P "$jobs" -I{} \
	sh -c "'$bin' run $sys {} | sed -n 2p > '$out/$sys-$label-{}.row'"
sort -t, -k3 -n "$out/$sys-$label"-20??.row >"$out/$sys-$label.csv"
echo "$label $sys hits=$(awk -F, '$5=="true"' "$out/$sys-$label.csv" | wc -l)/12"
