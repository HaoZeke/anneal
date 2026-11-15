#!/usr/bin/env bash
# Copyright (c) anneal developers
#
# Hashes the same runs on two builds, so a change that should leave a
# potential alone can be shown to. h2h_seed rows hash the trace of every
# new lowest energy with its call index and the lowest geometry. The
# h2h_morse and h2h_metal rows hash the result row with the clock columns
# removed. Each line ends in same or differs. The first line names the
# revisions and the host.
#
#   trace_identity.sh <develop examples> <develop revision> \
#       <branch examples> <branch revision> > identity.txt

set -euo pipefail

develop=${1:?develop examples directory}
develop_rev=${2:?develop revision}
branch=${3:?branch examples directory}
branch_rev=${4:?branch revision}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export develop branch scratch

seed_hash() {
    local bin=$1 kind=$2 n=$3 rho=$4 budget=$5 seed=$6
    local prefix
    prefix=$scratch/$(printf %s "$bin" | sha1sum | cut -c1-12)-$kind-$n-$rho-$seed
    "$bin/h2h_seed" "$kind" "$n" "$rho" "$budget" "$seed" recommended - "$prefix" > /dev/null
    printf '%s/%s' "$(sha1sum < "$prefix.trace.csv" | cut -c1-16)" "$(sha1sum < "$prefix.xyz" | cut -c1-16)"
}

row_hash() {
    "$@" | sed -n '2,$p' | cut -d, -f1-7,9 | sha1sum | cut -c1-16
}

one() {
    local tool=$1 a b
    shift
    case $tool in
        seed)
            a=$(seed_hash "$develop" "$@")
            b=$(seed_hash "$branch" "$@")
            ;;
        morse)
            a=$(row_hash "$develop/h2h_morse" run "$@")
            b=$(row_hash "$branch/h2h_morse" run "$@")
            ;;
        metal)
            a=$(H2H_PRIORS=0 row_hash "$develop/h2h_metal" run "$@")
            b=$(H2H_PRIORS=0 row_hash "$branch/h2h_metal" run "$@")
            ;;
    esac
    local verdict=differs
    [ "$a" = "$b" ] && verdict=same
    printf 'h2h_%s %s develop=%s branch=%s %s\n' "$tool" "$*" "$a" "$b" "$verdict"
}
export -f seed_hash row_hash one

printf '# code_revision develop=%s branch=%s host=%s\n' "$develop_rev" "$branch_rev" "$(uname -n)"
{
    for seed in 2000 2001 2002; do
        for n in 38 55 75 76 77 98 102 103 104 150; do
            echo "seed lj $n 0 400000 $seed"
        done
        echo "seed morse 80 6 400000 $seed"
        echo "seed morse 80 14 400000 $seed"
        echo "seed tip4p 8 0 100000 $seed"
        echo "seed tip4p 12 0 100000 $seed"
        echo "seed tip4p 20 0 100000 $seed"
        echo "morse m80r6 $seed"
        echo "morse m80r14 $seed"
    done
    for seed in $(seq 2000 2011); do
        echo "metal au55 $seed"
    done
} | xargs -P "${JOBS:-8}" -L 1 bash -c 'one "$@"' _ | sort -V
