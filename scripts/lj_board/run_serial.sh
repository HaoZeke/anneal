#!/bin/bash
# One ASE or AGOX process at a time, at the paper ceilings.
# A finished hit or ceiling is kept. A wall or a missing log is run again.
set -u
cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export PYTHONUNBUFFERED=1
export LJ_BOARD_ROOT=results/lj-board

mkdir -p "$LJ_BOARD_ROOT/provenance"

done_ok() {
  local log="$1"
  [[ -f "$log" ]] || return 1
  grep -q '^exit_code=0$' "$log" || return 1
  grep -q '^stop=hit$' "$log" && return 0
  grep -q '^stop=ceiling$' "$log" && return 0
  return 1
}

for n in 38 55 75 98 104; do
  for code in ase agox; do
    for seed in 6400 6401 6402 6403 6404 6405; do
      log="$LJ_BOARD_ROOT/$code/lj${n}-seed${seed}.log"
      if done_ok "$log"; then
        echo "keep $code $n $seed"
        continue
      fi
      echo "launch $code $n $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      python3 scripts/lj_board/run_arm.py "$code" "$n" "$seed"
      rc=$?
      echo "returned $code $n $seed rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      if [[ "$rc" -eq 75 ]]; then
        echo "wall stopped the panel"
        exit 75
      fi
      if [[ "$rc" -ne 0 ]]; then
        echo "arm failed"
        exit "$rc"
      fi
    done
  done
done

shopt -s nullglob
logs=("$LJ_BOARD_ROOT"/ase/lj*-seed*.log "$LJ_BOARD_ROOT"/agox/lj*-seed*.log)
python3 scripts/lj_board/summarize.py "${logs[@]}" > "$LJ_BOARD_ROOT/NUMBERS.tsv"
echo "wrote $LJ_BOARD_ROOT/NUMBERS.tsv"
