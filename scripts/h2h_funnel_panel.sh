#!/usr/bin/env bash
# Interleaved base and tip Lennard-Jones panel.
# One process, pinned to CPU 0. For each seed the base binary runs, then
# the tip binary. Budgets come from the base binary's budget command,
# which is the paper map.
#
# Each binary is run with that phase's repository as the working
# directory, so the row records the commit the phase was built from.
#
# Usage: scripts/h2h_funnel_panel.sh <base-binary> <tip-binary> <out-dir> <base-repo> <tip-repo>
set -euo pipefail

base_bin=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
tip_bin=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
mkdir -p "$3"
out=$(cd "$3" && pwd)
base_repo=$(cd "$4" && pwd)
tip_repo=$(cd "$5" && pwd)
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

systems=(lj38 lj55 lj75 lj98 lj104)
seeds=(2000 2001 2002 2003 2004 2005)

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_DYNAMIC=false

mkdir -p "$out/logs" "$out/bin"
cp -a "$base_bin" "$out/bin/base"
cp -a "$tip_bin" "$out/bin/tip"
base_bin=$out/bin/base
tip_bin=$out/bin/tip

{
  echo "rustc=$(rustc --version)"
  echo "base_sha256=$(sha256sum "$base_bin")"
  echo "tip_sha256=$(sha256sum "$tip_bin")"
  git rev-parse HEAD
  git rev-parse 'HEAD^{tree}'
  echo "===== lscpu ====="
  lscpu
} >"$out/environment.txt"

rows=$out/rows.csv
echo 'system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins' >"$rows"

run_one() {
  local phase=$1
  local bin=$2
  local repo=$3
  local system=$4
  local seed=$5
  local budget=$6
  local log=$out/logs/$system-$phase-$seed.log
  local coords=$out/$system/start-$seed.xyz
  mkdir -p "$out/$system" "$(dirname "$log")"
  {
    echo "phase=$phase"
    echo "system=$system"
    echo "seed=$seed"
    echo "budget=$budget"
    echo "binary_sha256=$(sha256sum "$bin")"
    echo "command=taskset -c 0 env H2H_BUDGET=$budget $bin run $system $seed $coords"
    echo "pins=$(taskset -cp $$ | awk '{print $NF}')"
    date -u +'start=%Y-%m-%dT%H:%M:%SZ'
  } >"$log"
  set +e
  (
    cd "$repo"
    taskset -c 0 env H2H_BUDGET="$budget" RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$bin" run "$system" "$seed" "$coords"
  ) >"$out/$system/$phase-$seed.row" 2>>"$log"
  local status=$?
  set -e
  echo "exit=$status" >>"$log"
  cat "$out/$system/$phase-$seed.row" >>"$log"
  date -u +'end=%Y-%m-%dT%H:%M:%SZ' >>"$log"
  if [[ $status -ne 0 ]]; then
    echo "fail $system $phase $seed" >&2
    return "$status"
  fi
  # The proof labels every row anneal. The phase names the binary.
  tail -n 1 "$out/$system/$phase-$seed.row" \
    | awk -F, -v phase="$phase" 'BEGIN { OFS = "," } { $2 = phase; print }' >>"$rows"
}

for system in "${systems[@]}"; do
  budget=$("$base_bin" budget "$system" | sed -n '1p')
  echo "system=$system budget=$budget"
  for seed in "${seeds[@]}"; do
    run_one base "$base_bin" "$base_repo" "$system" "$seed" "$budget"
    run_one tip "$tip_bin" "$tip_repo" "$system" "$seed" "$budget"
  done
done

python3 scripts/h2h_panel.py "$rows" "$out/PANEL.tsv"
echo "wrote $rows and $out/PANEL.tsv"
