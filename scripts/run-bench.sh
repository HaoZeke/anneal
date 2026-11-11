# Copyright (c) anneal developers
# One Lennard-Jones row under hostname bench.
# bash scripts/run-bench.sh SIZE SEED
# bash scripts/run-bench.sh --check LOG_DIR
set -u

command_for() {
  local n="$1"
  local seed="$2"
  printf "%s %s %s'\n" "sudo unshare --uts sh -c 'hostname bench; cd /work; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export NUMBA_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; export BLIS_NUM_THREADS=1; export OMP_PROC_BIND=TRUE; export OMP_PLACES=cores; export RAYON_NUM_THREADS=1; exec /work/lj_board_row" "$n" "$seed"
}

if [[ "${1:-}" == "--check" ]]; then
  root="${2:?log directory}"
  fail=0
  count=0
  for n in 38 55 75 98 104; do
    for seed in 6400 6401 6402 6403 6404 6405; do
      name="lj${n}-seed${seed}.log"
      log="${root}/${name}"
      got=$(command_for "$n" "$seed")
      got=${got%$'\n'}
      have=$(sed -n 's/^wrapper=//p' "$log" | head -n 1)
      count=$((count + 1))
      if [[ "$got" == "$have" ]]; then
        printf '%s identical\n' "$name"
      else
        printf '%s MISMATCH\n' "$name"
        diff -u <(printf '%s\n' "$have") <(printf '%s\n' "$got") || true
        fail=1
      fi
    done
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "stop: commands do not match"
    exit 1
  fi
  printf '%s identical\n' "$count"
  exit 0
fi

n="${1:?size}"
seed="${2:?seed}"
exec bash -c "$(command_for "$n" "$seed")"
