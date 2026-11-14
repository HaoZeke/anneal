#!/usr/bin/env bash
# Copyright (c) anneal developers
# Serial anneal versus GMIN board.
# One thread, pinned to CPU 0. Anneal then GMIN for each seed.
# Usage: scripts/h2h_gmin_board.sh <system>
# Systems: lj38 lj55 lj75 lj98 lj104
set -euo pipefail

system=${1:?system}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
out=${H2H_OUT:-h2h-board}
binary=${H2H_ANNEAL:-target/release/examples/h2h_gmin_proof}
gmin=${H2H_GMIN:-GMIN}
seeds=(2000 2001 2002 2003 2004 2005)
data_template=examples/h2h_gmin_lj.data

case $system in
  lj38|lj55|lj75|lj98|lj104) ;;
  *) echo "unknown system $system" >&2; exit 2 ;;
esac

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_DYNAMIC=false

head_sha=$(git rev-parse HEAD)
tree_sha=$(git rev-parse 'HEAD^{tree}')
host=$(hostname)
cpu=$(awk -F: '/^model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)
flags=$(awk -F: '/^flags/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)

mkdir -p "$out/$system" "$out/logs/$system"
printf '%s\n' "$cpu" >"$out/cpu-model.txt"
printf '%s\n' "$flags" >"$out/cpu-flags.txt"
printf '%s\n' "$host" >"$out/host.txt"
printf '%s\n' "$head_sha" >"$out/HEAD"
printf '%s\n' "$tree_sha" >"$out/TREE"

rows=$out/rows.csv
if [[ ! -s $rows ]]; then
  echo 'system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins' >"$rows"
fi

run_anneal() {
  local seed=$1
  local budget=$2
  local log=$out/logs/$system/anneal-$seed.log
  local coords=$out/$system/start-$seed.xyz
  if [[ -f $log ]] && grep -q '^exit=0$' "$log"; then
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  {
    echo "head=$head_sha"
    echo "tree=$tree_sha"
    echo "host=$host"
    echo "cpu=$cpu"
    echo "command=taskset -c 0 env H2H_BUDGET=$budget $binary run $system $seed $coords"
    echo "pins=$(taskset -cp $$ | awk '{print $NF}')"
  } >"$log"
  set +e
  taskset -c 0 env H2H_BUDGET="$budget" "$binary" run "$system" "$seed" "$coords" >"$out/$system/anneal-$seed.row" 2>>"$log"
  local status=$?
  set -e
  echo "exit=$status" >>"$log"
  cat "$out/$system/anneal-$seed.row" >>"$log"
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi
  tail -n 1 "$out/$system/anneal-$seed.row" >>"$rows"
}

run_gmin() {
  local seed=$1
  local budget=$2
  local target=$3
  local log=$out/logs/$system/gmin-$seed.log
  local coords=$out/$system/start-$seed.xyz
  local work=$out/$system/gmin-$seed
  if [[ -f $log ]] && grep -q '^exit=0$' "$log"; then
    return 0
  fi
  if [[ ! -s $coords ]]; then
    echo "missing start $coords" >&2
    return 1
  fi
  mkdir -p "$work"
  cp "$data_template" "$work/data"
  printf 'RANSEED %s\n' "$seed" >>"$work/data"
  cp "$coords" "$work/coords"
  {
    echo "head=$head_sha"
    echo "tree=$tree_sha"
    echo "host=$host"
    echo "cpu=$cpu"
    echo "command=taskset -c 0 env H2H_SYSTEM=$system H2H_SEED=$seed H2H_BUDGET=$budget H2H_TARGET=$target H2H_TOL=0.001 GMIN"
    echo "pins=$(taskset -cp $$ | awk '{print $NF}')"
    echo "data=examples/h2h_gmin_lj.data"
  } >"$log"
  set +e
  (
    cd "$work"
    taskset -c 0 env \
      H2H_SYSTEM="$system" \
      H2H_SEED="$seed" \
      H2H_BUDGET="$budget" \
      H2H_TARGET="$target" \
      H2H_TOL=0.001 \
      H2H_ROW=row.txt \
      OMP_NUM_THREADS=1 \
      "$gmin" >gmin.out 2>gmin.err
  )
  local status=$?
  set -e
  echo "exit=$status" >>"$log"
  if [[ -f $work/row.txt ]]; then
    cat "$work/row.txt" >>"$log"
  fi
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi
  python3 - "$work/row.txt" "$head_sha" "$tree_sha" "$cpu" >>"$rows" <<'PY'
import sys
line = open(sys.argv[1]).read().strip()
fields = {}
for part in line.split()[1:]:
    key, value = part.split("=", 1)
    fields[key] = value
pins = "0"
print(",".join([
    fields["system"], fields["code"], fields["seed"], fields["budget"],
    fields["hit"], fields["evals_to_hit"], fields["charged"], fields["wall_s"],
    fields["best"], sys.argv[2], sys.argv[3], sys.argv[4].replace(",", " "), pins,
]))
PY
}

budget_line=$("$binary" budget "$system")
budget=$(printf '%s\n' "$budget_line" | sed -n '1p')
target=$(printf '%s\n' "$budget_line" | sed -n '2p')
for seed in "${seeds[@]}"; do
  run_anneal "$seed" "$budget"
  run_gmin "$seed" "$budget" "$target"
done

"$binary" summarize "$rows" "$out/summary.csv"
echo "wrote $rows and $out/summary.csv"
