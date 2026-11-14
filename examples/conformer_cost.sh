#!/usr/bin/env bash
# conformer_cost.sh [CPU]: rerun the two peptide cost runs of the
# conformer_search benchmark from examples/fixtures/ac-ala3-nhme-s9743.xyz.
#
# It builds the example in release and runs it twice, one run after the
# other, pinned to CPU (1 unless given): with the communicating chains, and
# with v1's arguments, one chain and a private history.  Both runs have the
# benchmark's budget of 13750 calls, its seed 9743 and its environment: one
# OpenMP thread, the tblite GFN2-xTB engine and, when it is installed, the
# serial OpenBLAS preload.  xtb must be on PATH, and python3 must have
# tblite 0.7.0 and ASE 3.29.0.  Outputs go under $WORK, target/conformer-cost
# unless set.  What it prints names the repository, scratch, cargo and home
# directories by short names, and the machine only by uname -m, nproc and the
# CPU model.
set -u
export LC_ALL=C
cd "$(dirname "$0")/.." || exit 2
root=$PWD
cpu=${1:-1}
budget=13750
seed=9743
start=examples/fixtures/ac-ala3-nhme-s9743.xyz
start_sha256=9cb8e0ff5481827146bb0cf5626d1bc1d75257dc77bf1189a063aaa909ced698
work=${WORK:-target/conformer-cost}
blas=/usr/lib/$(uname -m)-linux-gnu/openblas-serial
communicating=(--replicas 4 --history shared --arm communicating --superbasin on)
v1=(--replicas 1 --history private --precondition on --polish 0 --settle 0.1)

for tool in cargo xtb python3 taskset; do
    command -v "$tool" > /dev/null || { echo "$tool is not on PATH" >&2; exit 2; }
done
[ "$(sha256sum < "$start" | cut -d' ' -f1)" = "$start_sha256" ] ||
    { echo "$start is not the benchmark's start" >&2; exit 2; }
preload=
if [ -e "$blas/libblas.so.3" ] && [ -e "$blas/liblapack.so.3" ]; then
    preload=$blas/libblas.so.3:$blas/liblapack.so.3
fi

neutral() {
    sed -e "s|${CARGO_TARGET_DIR:-/nonexistent}|\$CARGO_TARGET_DIR|g" -e "s|$root|.|g" \
        -e "s|${CARGO_HOME:-/nonexistent}|\$CARGO_HOME|g" -e "s|${RUSTUP_HOME:-/nonexistent}|\$RUSTUP_HOME|g" \
        -e "s|${TMPDIR:-/nonexistent}|\$TMPDIR|g" -e "s|${HOME:-/nonexistent}|~|g"
}

run() {
    local name=$1 dir=$work/$1 t0 t1 code
    shift
    rm -rf "$dir" && mkdir -p "$dir" && cp "$start" "$dir/start.xyz"
    echo "## $name: taskset -c $cpu conformer_search start.xyz $budget $seed anneal $*"
    t0=$EPOCHREALTIME
    (cd "$dir" && env OMP_NUM_THREADS=1 ASE_ENGINE=tblite-gfn2 ${preload:+LD_PRELOAD=$preload} \
        taskset -c "$cpu" "$bin" start.xyz "$budget" "$seed" anneal "$@" > anneal.out 2> anneal.err)
    code=$?
    t1=$EPOCHREALTIME
    echo "exit $code, wall $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }') s"
    grep -E '^(calls|charged) [0-9]+$' "$dir/anneal.out"
    sort -u "$dir/anneal.err" | grep '^engine_package '
}

{
    echo "# conformer_cost.sh: CPU $cpu, budget $budget, seed $seed"
    echo "# commit $(git rev-parse HEAD 2> /dev/null || echo 'unknown, not a git checkout')"
    echo "## machine: uname -m, nproc, CPU model"
    uname -m
    nproc
    sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1
    echo "## start $start"
    echo "sha256 $start_sha256"
    echo "## tools"
    cargo --version
    rustc --version
    xtb --version 2>&1 | grep -o -m1 'xtb version [0-9.]* ([0-9a-f]*)'
    python3 --version
    if [ -n "$preload" ]; then echo "openblas-serial preload: on"; else echo "openblas-serial preload: off, not installed"; fi
    echo "## cargo build --release --example conformer_search"
    cargo build --release --example conformer_search 2>&1
    echo "exit $?"
    bin=$(cd "${CARGO_TARGET_DIR:-target}" && pwd)/release/examples/conformer_search
    echo "conformer_search sha256 $(sha256sum < "$bin" | cut -d' ' -f1)"
    run communicating "${communicating[@]}"
    run v1 "${v1[@]}"
} 2>&1 | neutral
