#!/bin/bash
# Copyright (c) anneal developers
# One cell at a time, pinned to cpu 0.
#
# Usage: run_serial.sh WORK BIN
# WORK receives logs. BIN holds lj_dls, lj_requench, h2h_seed
# and h2h-base. The caller chooses WORK. This file names no
# machine directory.
set -eu
export TZ=America/Chicago
work=$1
bin=$2
mkdir -p "$work/logs" "$work/requench" "$work/receipts"
if [[ ! -f $work/receipts/before-first-search.txt ]]; then
  date '+%Y-%m-%dT%H:%M:%S%z' > "$work/receipts/before-first-search.txt"
  hostname > "$work/receipts/uts.txt"
  lscpu > "$work/receipts/lscpu.txt"
  cat /proc/loadavg > "$work/receipts/load-at-start.txt"
  echo "pin=0" > "$work/receipts/affinity.txt"
fi
header() {
  echo "start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "uts=$(hostname)"
  echo "load=$(cat /proc/loadavg)"
  echo "affinity=$(taskset -pc $$)"
  echo "pin=0"
  echo "lscpu<<"
  lscpu
  echo "lscpu>>"
}
run_dls() {
  local n=$1 budget=$2 seed=$3 tie=$4 stop=$5 tag=$6
  local dir=$work/logs/$tag
  mkdir -p "$dir"
  local log=$dir/$seed.log
  if [[ -e $log ]]; then
    echo "skip $tag $seed"
    return
  fi
  local prefix=$dir/$seed
  {
    header
    echo "n=$n"
    echo "seed=$seed"
    echo "tie=$tie"
    echo "budget=$budget"
    echo "stop=$stop"
    taskset -c 0 "$bin/lj_dls" "$n" "$budget" "$seed" \
      "$tie" "$stop" "$prefix"
    echo "dls_exit=$?"
    echo "end=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  } > "$log" 2> "$dir/$seed.err"
  if [[ -f $prefix.xyz ]]; then
    "$bin/lj_requench" "$prefix.xyz" \
      > "$work/requench/${tag}-$seed.txt" \
      2> "$work/requench/${tag}-$seed.err"
  fi
  echo "done $tag $seed $(date '+%H:%M:%S')" >> "$work/status.txt"
}
run_h2h() {
  local which=$1 n=$2 budget=$3 seed=$4 arm=$5 stop=$6 tag=$7
  local dir=$work/logs/$tag
  mkdir -p "$dir"
  local log=$dir/$seed.log
  if [[ -e $log ]]; then
    echo "skip $tag $seed"
    return
  fi
  local prefix=$dir/$seed
  local prog=$bin/h2h_seed
  if [[ $which == base ]]; then
    prog=$bin/h2h-base
  fi
  {
    header
    echo "which=$which"
    echo "n=$n"
    echo "seed=$seed"
    echo "arm=$arm"
    echo "budget=$budget"
    echo "stop=$stop"
    taskset -c 0 "$prog" lj "$n" 0 "$budget" "$seed" \
      "$arm" "$stop" "$prefix"
    echo "h2h_exit=$?"
    echo "end=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  } > "$log" 2> "$dir/$seed.err"
  if [[ -f $prefix.xyz ]]; then
    "$bin/lj_requench" "$prefix.xyz" \
      > "$work/requench/${tag}-$seed.txt" \
      2> "$work/requench/${tag}-$seed.err"
  fi
  echo "done $tag $seed $(date '+%H:%M:%S')" >> "$work/status.txt"
}
for tie in lowest-index uniform; do
  for seed in $(seq 3000 3029); do
    run_dls 104 2000000 "$seed" "$tie" -582.086641 "$tie"
  done
done
for spec in "38 400000 -173.928426" "55 1000000 -279.248469" \
            "75 4000000 -397.492330" "98 4000000 -543.665360"; do
  set -- $spec
  n=$1
  budget=$2
  stop=$3
  for seed in 4100 4101 4102 4103 4104 4105; do
    run_dls "$n" "$budget" "$seed" lowest-index "$stop" "dls-n${n}"
  done
done
for spec in "38 400000 -173.928426" "55 1000000 -279.248469" \
            "75 4000000 -397.492330" "98 4000000 -543.665360"; do
  set -- $spec
  n=$1
  budget=$2
  stop=$3
  for seed in 4100 4101 4102 4103 4104 4105; do
    run_h2h tip "$n" "$budget" "$seed" recommended \
      "$stop" "regress-tip-n${n}"
    run_h2h base "$n" "$budget" "$seed" recommended \
      "$stop" "regress-base-n${n}"
  done
done
for arm in temperature-1.5 angular; do
  for seed in $(seq 3000 3029); do
    run_h2h tip 104 2000000 "$seed" "$arm" -582.086641 "$arm"
  done
done
echo "SERIAL_DONE $(date '+%Y-%m-%dT%H:%M:%S%z')" >> "$work/status.txt"
