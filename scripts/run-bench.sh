# Copyright (c) anneal developers
# Lennard-Jones rows from /work. One size and seed at a time.
# bash scripts/run-bench.sh --build
# bash scripts/run-bench.sh --all
# bash scripts/run-bench.sh SIZE SEED
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
bin=/work/lj_board_row
log_root=/work/v33/logs
script_path=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

build_line() {
  local cargo_home="${CARGO_HOME-}"
  printf '%s\n' "cd ${root} && CARGO_TARGET_DIR=/work/target TMPDIR=/work/rustc-tmp RUSTFLAGS='--remap-path-prefix=${root}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo' cargo +1.88.0 build --release --locked --example lj_board_row -j 4"
}

do_build() {
  mkdir -p /work/target /work/rustc-tmp /work/hw
  local line
  line=$(build_line)
  printf '%s\n' "$line" > /work/hw/build-command.txt
  # shellcheck disable=SC2086
  bash -c "$line"
  cp -f /work/target/release/examples/lj_board_row "$bin"
}

command_for() {
  local n="$1"
  local seed="$2"
  local side="$3"
  printf '%s\n' "sudo unshare --uts sh -c 'hostname bench; cd /work; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export NUMBA_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; export BLIS_NUM_THREADS=1; export OMP_PROC_BIND=TRUE; export OMP_PLACES=cores; export RAYON_NUM_THREADS=1; boot_id=\$(cat /proc/sys/kernel/random/boot_id); uptime=\$(cat /proc/uptime); printf \"boot_id=%s\\n\" \"\$boot_id\"; printf \"uptime=%s\\n\" \"\$uptime\"; printf \"hostname=%s\\n\" \"\$(hostname)\"; printf \"boot_id=%s\\nuptime=%s\\nhostname=%s\\n\" \"\$boot_id\" \"\$uptime\" \"\$(hostname)\" > ${side}; exec /work/lj_board_row ${n} ${seed}'"
}

run_one() {
  local n="$1"
  local seed="$2"
  local name="lj${n}-seed${seed}"
  local log="${log_root}/${name}.log"
  local side="${log_root}/${name}.sidecar"
  local sha script_sha wrapper
  sha=$(sha256sum "$bin" | awk '{print $1}')
  script_sha=$(sha256sum "$script_path" | awk '{print $1}')
  wrapper=$(command_for "$n" "$seed" "$side")
  wrapper=${wrapper%$'\n'}
  mkdir -p "$log_root"
  echo "launch n=$n seed=$seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local t0 t1
  t0=$(date +%s.%N)
  {
    printf 'binary_sha256=%s\n' "$sha"
    printf 'script_sha256=%s\n' "$script_sha"
    printf 'wrapper=%s\n' "$wrapper"
    printf 'cwd=/work\n'
    printf 'shell_start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash -c "$wrapper"
    printf 'process_exit=%s\n' "$?"
  } > "$log" 2>&1
  t1=$(date +%s.%N)
  python3 -c 'import sys; print("process_wall_s={:.6f}".format(float(sys.argv[2]) - float(sys.argv[1])))' "$t0" "$t1" >> "$log"
  if ! grep -q '^hostname=bench$' "$log"; then
    echo "hostname was not bench" >&2
    exit 1
  fi
  if ! grep -q '^boot_id=.\+' "$log"; then
    echo "boot id missing" >&2
    exit 1
  fi
  if ! grep -q '/proc/sys/kernel/random/boot_id' "$log"; then
    echo "boot id path missing" >&2
    exit 1
  fi
  if ! grep -q '^uptime=' "$log"; then
    echo "uptime missing" >&2
    exit 1
  fi
  if ! grep -q '^boot_id=' "$side"; then
    echo "sidecar missing boot id" >&2
    exit 1
  fi
  if ! grep -q '^uptime=' "$side"; then
    echo "sidecar missing uptime" >&2
    exit 1
  fi
  local got
  # Read every wrapper line. Taking the first one after sed finishes
  # keeps the producer from dying under pipefail.
  got=$(sed -n 's/^wrapper=//p' "$log")
  got=${got%%$'\n'*}
  local again
  again=$(command_for "$n" "$seed" "$side")
  again=${again%$'\n'}
  if [[ "$got" != "$again" ]]; then
    echo "wrapper line does not match the script" >&2
    exit 1
  fi
  local got_sha again_sha
  got_sha=$(printf '%s' "$got" | sha256sum | awk '{print $1}')
  again_sha=$(printf '%s' "$again" | sha256sum | awk '{print $1}')
  if [[ "$got_sha" != "$again_sha" ]]; then
    echo "wrapper sha does not match" >&2
    exit 1
  fi
  local prc
  prc=$(grep '^process_exit=' "$log" | tail -n 1 | cut -d= -f2)
  if [[ "$prc" != "0" ]]; then
    echo "arm failed n=$n seed=$seed" >&2
    tail -n 20 "$log" >&2
    exit 1
  fi
  if ! grep -q 'exit_code=0' "$log"; then
    echo "tool exit was not 0" >&2
    exit 1
  fi
  if ! grep -q 'energy_calls=' "$log"; then
    echo "call counts missing" >&2
    exit 1
  fi
  if ! grep -q 'gradient_calls=' "$log"; then
    echo "call counts missing" >&2
    exit 1
  fi
  if ! grep -q 'stop=hit' "$log" && ! grep -q 'stop=ceiling' "$log"; then
    echo "stop was not hit or ceiling" >&2
    exit 1
  fi
  echo "returned n=$n seed=$seed process_exit=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

if [[ "${1:-}" == "--build" ]]; then
  do_build
  exit 0
fi

if [[ "${1:-}" == "--all" ]]; then
  cd /work
  mkdir -p "$log_root" /work/hw
  sha256sum "$bin" > /work/hw/binary.sha256
  build_line > /work/hw/build-command.txt
  sha256sum "$script_path" > /work/hw/script.sha256
  for n in 38 55 75 98 104; do
    for seed in 6400 6401 6402 6403 6404 6405; do
      run_one "$n" "$seed"
    done
  done
  echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

n="${1:?size}"
seed="${2:?seed}"
cd /work
mkdir -p "$log_root"
run_one "$n" "$seed"
