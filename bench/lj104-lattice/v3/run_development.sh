#!/bin/bash
# Copyright (c) anneal developers
# Development seeds 3000-3029, one cell at a time, cpu 0.
#
# Usage: run_development.sh WORK BIN
# WORK receives the logs. BIN holds lj_dls and lj_requench.
set -eu
export TZ=America/Chicago
work=$1
bin=$2
mkdir -p "$work/logs/lowest-index" "$work/logs/uniform" \
  "$work/requench" "$work/receipts"
if [[ ! -f $work/receipts/T.txt ]]; then
  date '+%Y-%m-%dT%H:%M:%S%z' > "$work/receipts/T.txt"
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
  local tie=$1 seed=$2
  local dir=$work/logs/$tie
  local log=$dir/$seed.log
  if [[ -e $log ]]; then
    echo "skip $tie $seed"
    return
  fi
  local prefix=$dir/$seed
  {
    header
    echo "n=104"
    echo "seed=$seed"
    echo "tie=$tie"
    echo "budget=2000000"
    echo "stop=-582.086641"
    taskset -c 0 "$bin/lj_dls" 104 2000000 "$seed" \
      "$tie" -582.086641 "$prefix"
    echo "dls_exit=$?"
    echo "end=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  } > "$log" 2> "$dir/$seed.err"
  if [[ -f $prefix.xyz ]]; then
    "$bin/lj_requench" "$prefix.xyz" \
      > "$work/requench/${tie}-$seed.txt" \
      2> "$work/requench/${tie}-$seed.err"
  fi
  echo "done $tie $seed $(date '+%H:%M:%S')" >> "$work/status.txt"
}
for tie in lowest-index uniform; do
  for seed in $(seq 3000 3029); do
    run_dls "$tie" "$seed"
  done
done
echo "SERIAL_DONE $(date '+%Y-%m-%dT%H:%M:%S%z')" >> "$work/status.txt"
