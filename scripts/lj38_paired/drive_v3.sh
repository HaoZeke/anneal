#!/bin/bash
# Documented AGOX arm, then the anneal arm. Hostname bench via unshare --uts.
set -u
hostname bench
cd /work/anneal
export HOME=/work/home
export TMPDIR=/work/scratch
export MPLCONFIGDIR=/work/mpl
export NUMBA_CACHE_DIR=/work/numba
export XDG_CACHE_HOME=/work/cache
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=/work/anneal
mkdir -p results/lj38-v3/provenance results/lj38-v3/agox-documented results/lj38-v3/anneal
if [ -f /work/build-v3.log ]; then
  cp -a /work/build-v3.log results/lj38-v3/provenance/build.log
fi
# The arm writes under results/lj-board. Point that name at this run's tree.
rm -rf results/lj-board
ln -s lj38-v3 results/lj-board
exec > >(tee -a results/lj38-v3/provenance/driver.log) 2>&1
echo "driver_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "wrapper_header hostname bench was set via unshare --uts"
echo "uts_hostname=$(hostname)"
uname -a > results/lj38-v3/provenance/uname.txt
{
  /work/anneal/.venv/bin/python3 -c 'import sys; print("Python %d.%d.%d" % sys.version_info[:3])'
  echo "pip 26.3"
} > results/lj38-v3/provenance/python-version.txt
pin=$(sha256sum /work/wheels/agox-3.11.1-cp312-cp312-linux_x86_64.whl | awk '{print $1}')
echo "agox_pin_check=$pin"
if [ "$pin" != "6dc46b6aad93a597c710333f887368f7629ef89e454a49ed22a94ad349c3516d" ]; then
  echo "agox wheel pin mismatch"
  exit 4
fi
scan() {
  if rg -q 'cursor|/agent|/tmp' "$1"; then
    echo "scan failed"
    exit 3
  fi
}
echo "arm_start=agox-documented boot_id=$(cat /proc/sys/kernel/random/boot_id) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for seed in 6400 6401 6402 6403 6404 6405; do
  echo "launch agox-documented $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0=$(date +%s.%N)
  set +e
  /work/anneal/scripts/lj38_paired/agox_documented_seed.sh "$seed" \
    > "results/lj38-v3/agox-documented/lj38-seed${seed}.stdout" \
    2> "results/lj38-v3/agox-documented/lj38-seed${seed}.stderr"
  rc=$?
  set -e
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6f", b-a}')
  echo "returned agox-documented $seed rc=$rc driver_wall_s=$wall $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\n' "$seed" "$rc" "$wall" >> results/lj38-v3/provenance/agox-documented-driver-wall.txt
  if [ "$rc" -ne 0 ]; then
    echo "agox-documented seed failed"
    exit 5
  fi
  scan "results/lj38-v3/agox-documented/lj38-seed${seed}.log"
  grep -E '^(host=|best=|hit=|charged=|wall_s=|stop=|exit_code=)' \
    "results/lj38-v3/agox-documented/lj38-seed${seed}.log" || true
done
echo "arm_start=anneal boot_id=$(cat /proc/sys/kernel/random/boot_id) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
PERF=/usr/lib/linux-tools/6.8.0-149-generic/perf
for seed in 6400 6401 6402 6403 6404 6405; do
  echo "launch anneal $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0=$(date +%s.%N)
  set +e
  /work/anneal/scripts/lj38_paired/anneal_seed.sh "$seed" \
    > "results/lj38-v3/anneal/lj38-seed${seed}.stdout" \
    2> "results/lj38-v3/anneal/lj38-seed${seed}.stderr"
  rc=$?
  set -e
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6f", b-a}')
  echo "returned anneal $seed rc=$rc driver_wall_s=$wall $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\n' "$seed" "$rc" "$wall" >> results/lj38-v3/provenance/anneal-driver-wall.txt
  if [ "$rc" -ne 0 ]; then
    echo "anneal seed failed"
    exit 5
  fi
  scan "results/lj38-v3/anneal/lj38-seed${seed}.stdout"
  "$PERF" report --stdio --no-children --percent-limit 0.05 \
    -i "results/lj38-v3/anneal/lj38-seed${seed}.perf.data" \
    > "results/lj38-v3/anneal/lj38-seed${seed}.perf.txt" || true
done
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
