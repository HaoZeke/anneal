#!/bin/bash
# Both arms, seeds 6400-6405, inside one UTS namespace named bench.
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
mkdir -p results/lj-board/provenance results/lj-board/agox results/lj-board/anneal
exec > >(tee -a results/lj-board/provenance/driver.log) 2>&1
echo "driver_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "wrapper_header hostname bench was set via unshare --uts"
echo "uts_hostname=$(hostname)"
uname -a > results/lj-board/provenance/uname.txt
{
  /work/anneal/.venv/bin/python3 -c 'import sys; print("Python %d.%d.%d" % sys.version_info[:3])'
  echo "pip 26.3"
} > results/lj-board/provenance/python-version.txt
{
  echo "rustc=rustc 1.88.0 (6b00bc388 2025-06-23)"
  echo "cargo=cargo 1.88.0 (873a06493 2025-05-10)"
  echo "binary_sha256=ec3b443bb9abbf2b3615b1691f4c5ff5d9060604c6f7ad1969f95b36a58bc2a0"
  echo "remap=--remap-path-prefix"
  echo "ase_wheel_sha256=7b9dd103f007810339c24acfee2f6b677c0c48443b21d3c98e52959246cf4ebf"
  echo "agox_wheel_sha256=6dc46b6aad93a597c710333f887368f7629ef89e454a49ed22a94ad349c3516d"
} > results/lj-board/provenance/anneal-build.txt
pin=$(sha256sum /work/wheels/agox-3.11.1-cp312-cp312-linux_x86_64.whl | awk '{print $1}')
echo "agox_pin_check=$pin"
if [ "$pin" != "6dc46b6aad93a597c710333f887368f7629ef89e454a49ed22a94ad349c3516d" ]; then
  echo "agox wheel pin mismatch"
  exit 4
fi
cp -a /work/lj-board-notes/NOTES.md results/lj-board/provenance/NOTES.md
scan() {
  if rg -q 'cursor|/agent|/tmp' "$1"; then
    echo "scan failed"
    exit 3
  fi
}
for seed in 6400 6401 6402 6403 6404 6405; do
  echo "launch agox $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0=$(date +%s.%N)
  set +e
  /work/run/agox_seed.sh "$seed" \
    > "results/lj-board/agox/lj38-seed${seed}.stdout" \
    2> "results/lj-board/agox/lj38-seed${seed}.stderr"
  rc=$?
  set -e
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6f", b-a}')
  echo "returned agox $seed rc=$rc driver_wall_s=$wall $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\n' "$seed" "$rc" "$wall" >> results/lj-board/provenance/agox-driver-wall.txt
  if [ "$rc" -ne 0 ]; then
    echo "agox seed failed"
    exit 5
  fi
  log="results/lj-board/agox/lj38-seed${seed}.log"
  scan "$log"
  scan "results/lj-board/agox/lj38-seed${seed}.stdout"
  grep -E '^(host=|best=|hit=|charged=|wall_s=|stop=|exit_code=)' "$log" || true
done
PERF=/usr/lib/linux-tools/6.8.0-149-generic/perf
for seed in 6400 6401 6402 6403 6404 6405; do
  echo "launch anneal $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0=$(date +%s.%N)
  set +e
  /work/run/anneal_seed.sh "$seed" \
    > "results/lj-board/anneal/lj38-seed${seed}.stdout" \
    2> "results/lj-board/anneal/lj38-seed${seed}.stderr"
  rc=$?
  set -e
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6f", b-a}')
  echo "returned anneal $seed rc=$rc driver_wall_s=$wall $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\n' "$seed" "$rc" "$wall" >> results/lj-board/provenance/anneal-driver-wall.txt
  if [ "$rc" -ne 0 ]; then
    echo "anneal seed failed"
    exit 5
  fi
  scan "results/lj-board/anneal/lj38-seed${seed}.stdout"
  "$PERF" report --stdio --no-children --percent-limit 0.05 \
    -i "results/lj-board/anneal/lj38-seed${seed}.perf.data" \
    > "results/lj-board/anneal/lj38-seed${seed}.perf.txt" || true
done
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
