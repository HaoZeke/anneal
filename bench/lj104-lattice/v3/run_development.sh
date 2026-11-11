#!/bin/bash
# Copyright (c) anneal developers
# Development seeds 3000-3029, one cell at a time, cpu 0.
# The runner re-execs in a UTS namespace whose host name is bench.
#
# Usage: run_development.sh WORK BIN
# WORK receives the logs. BIN holds lj_dls and lj_requench.
set -eu
export TZ=America/Chicago
if [[ "${ANNEAL_BENCH_UTS:-}" != 1 ]]; then
  export ANNEAL_BENCH_UTS=1
  exec sudo unshare --uts sh -c 'hostname bench; exec "$@"' _ "$0" "$@"
fi
work=$1
bin=$2
mkdir -p "$work/logs/lowest-index" "$work/logs/uniform" \
  "$work/requench" "$work/receipts"
if [[ ! -f $work/receipts/T.txt ]]; then
  date '+%Y-%m-%dT%H:%M:%S%z' > "$work/receipts/T.txt"
  hostname > "$work/receipts/uts.txt"
  hostname > "$work/receipts/hostname.txt"
  cat /proc/sys/kernel/random/boot_id > "$work/receipts/boot_id.txt"
  lscpu > "$work/receipts/lscpu.txt"
  cat /proc/meminfo > "$work/receipts/meminfo.txt"
  nproc > "$work/receipts/nproc.txt"
  cat /proc/loadavg > "$work/receipts/load-at-start.txt"
  echo "pin=0" > "$work/receipts/affinity.txt"
  (
    cd "$bin"
    sha256sum lj_dls lj_requench
  ) > "$work/receipts/binaries.sha"
  printf '%s\n' "sudo unshare --uts sh -c 'hostname bench; exec <cmd>'" \
    > "$work/receipts/wrapper.txt"
fi
header() {
  echo "start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "uts=$(hostname)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "lj_dls_sha256=$(sha256sum "$bin/lj_dls" | awk '{print $1}')"
  echo "lj_requench_sha256=$(sha256sum "$bin/lj_requench" | awk '{print $1}')"
  echo "wrapper=sudo unshare --uts sh -c 'hostname bench; exec <cmd>'"
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
    sudo unshare --uts sh -c 'hostname bench; exec "$@"' _ \
      taskset -c 0 "$bin/lj_dls" 104 2000000 "$seed" \
      "$tie" -582.086641 "$prefix"
    echo "dls_exit=$?"
    echo "end=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  } > "$log" 2> "$dir/$seed.err"
  if [[ -f $prefix.xyz ]]; then
    sudo unshare --uts sh -c 'hostname bench; exec "$@"' _ \
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
