#!/bin/bash
# Copyright (c) anneal developers
# wrapper header
# Every timed command for both arms runs as:
#   sudo unshare --uts sh -c 'hostname bench; exec <cmd>'
# This launcher is itself started that way. sudo clears PATH, so the
# UTS shell sources /home/ubuntu/work/bench-env.sh before exec.
set -u
. /home/ubuntu/work/bench-env.sh
OUT=/home/ubuntu/work/out-15004-bench
BIN=/home/ubuntu/work/target-h2o16/release/examples/gfn2_water_h2h
START_ROOT=/home/ubuntu/work/starts-v9
SEED=15004
SCRIPT=/home/ubuntu/work/h2o16_v9_one_bench.sh
mkdir -p "$OUT"
{
  echo "wrapper: sudo unshare --uts sh -c 'hostname bench; exec <cmd>'"
  echo "timed_commands: anneal search, anneal xtb score, crest search, crest xtb score"
  echo "uname_n $(uname -n)"
  echo "proc_hostname $(tr -d '\n' < /proc/sys/kernel/hostname)"
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nproc $(nproc)"
  echo "rustc $(rustc -V)"
  echo "cargo $(cargo -V)"
  echo "which_rustc $(command -v rustc)"
  echo "which_xtb $(command -v xtb)"
  echo "which_crest $(command -v crest)"
  echo "xtb_path $XTB_REAL"
  echo "crest_path $CREST_BIN"
  echo "engine_path $RGPOT_XTB_ENGINE"
  echo "ld_library_path $LD_LIBRARY_PATH"
  echo "bin $BIN"
  echo "start_root $START_ROOT"
  echo "script $SCRIPT"
  echo "seed $SEED"
  echo "command sudo unshare --uts sh -c 'hostname bench; exec bash $SCRIPT $BIN $START_ROOT $OUT $SEED'"
  sha256sum "$BIN" "$XTB_REAL" "$CREST_BIN" "$RGPOT_XTB_ENGINE" "$LD_LIBRARY_PATH/libxtb.so.6.7.1" "$SCRIPT"
  echo "=== xtb version ==="
  "$XTB_REAL" --version
  echo "=== crest version ==="
  "$CREST_BIN" --version
  echo "=== lscpu ==="
  lscpu
  echo "=== meminfo ==="
  cat /proc/meminfo
} > "$OUT/host-before.txt"
host=$(tr -d '\n' < /proc/sys/kernel/hostname)
if [[ $host != bench || $(uname -n) != bench ]]; then
  echo "refusing to search: hostname is not bench" >&2
  exit 1
fi
echo "DRIVER_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/driver.log"
start_unix=$(date +%s.%N)
printf 'start_unix %s\n' "$start_unix" > "$OUT/time-driver.txt"
bash "$SCRIPT" "$BIN" "$START_ROOT" "$OUT" "$SEED" \
  > "$OUT/driver.stdout" 2> "$OUT/driver.stderr"
ec=$?
end_unix=$(date +%s.%N)
python3 -c 'import sys; a,b,e=sys.argv[1:]; print("end_unix %s" % b); print("wall_s %.6f" % (float(b)-float(a))); print("exit %s" % e)' \
  "$start_unix" "$end_unix" "$ec" >> "$OUT/time-driver.txt"
{
  echo "=== end ==="
  echo "uname_n $(uname -n)"
  echo "proc_hostname $(tr -d '\n' < /proc/sys/kernel/hostname)"
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nproc $(nproc)"
  echo "rustc $(rustc -V)"
  echo "cargo $(cargo -V)"
  lscpu
  cat /proc/meminfo
} > "$OUT/host-after.txt"
echo "DRIVER_EXIT $ec $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/driver.log"
exit $ec
