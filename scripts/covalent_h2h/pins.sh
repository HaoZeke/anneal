#!/usr/bin/env bash
# Copyright (c) anneal developers
#
# Prints the Lennard-Jones, Morse, Gupta and TIP4P pins with every clock
# column removed, so two builds compare with diff. The h2h_seed rows spend
# the whole budget and hash every new lowest energy with its call index.
#
#   pins.sh <examples directory> > pins.txt

set -euo pipefail

bin=${1:?examples directory}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

seed_row() {
    local system=$1 n=$2 rho=$3 budget=$4 seed=$5
    local prefix=$scratch/$system-$n-$seed
    "$bin/h2h_seed" "$system" "$n" "$rho" "$budget" "$seed" recommended - "$prefix" > /dev/null
    printf 'h2h_seed %s %s %s %s %s trace=%s lowest=%s\n' "$system" "$n" "$rho" "$budget" "$seed" \
        "$(sha1sum < "$prefix.trace.csv" | cut -c1-40)" "$(sha1sum < "$prefix.xyz" | cut -c1-40)"
}

for seed in 2000 2001 2002; do
    seed_row lj 13 0 20000 "$seed"
    seed_row lj 38 0 200000 "$seed"
    seed_row lj 55 0 100000 "$seed"
    seed_row lj 75 0 200000 "$seed"
    seed_row morse 38 14 100000 "$seed"
done
for seed in 2000 2001; do
    seed_row tip4p 8 0 50000 "$seed"
    seed_row tip4p 12 0 50000 "$seed"
done

# system,code,seed,budget,hit,evals_to_hit,charged and best; the arm lines
# carry draws, accepts and the best quench of each move.
for system in m38r6 m38r14 m55r14; do
    for seed in 2000 2001 2002; do
        "$bin/h2h_morse" run "$system" "$seed" | sed -n '2,$p' | sed 's/^\([^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*\),[^,]*,\([^,]*\).*/\1,\2/'
    done
done
for system in zn38 cu19au19 pt13; do
    for seed in 2000 2001 2002; do
        "$bin/h2h_metal" run "$system" "$seed" | sed -n '2,$p' | sed 's/^\([^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*\),[^,]*,\([^,]*\).*/\1,\2/'
    done
done
