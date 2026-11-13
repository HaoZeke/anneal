#!/usr/bin/env bash
# Gupta gold against GMIN at 38, 75 and 98.
#
# Serial, one core, hostname bench. Paper budgets: 38 -> 400000,
# 75 and 98 -> 4000000. Cambridge Cu-Au Gupta headers stop Au38.
# The archives have no pure-gold 75 or 98, so those two spend the
# budget and the comparison is the best energy.
#
# GMIN is the Darby binary Gupta keyword (BGUPTAT) with the gold row
# A=0.2061, p=10.229, q=4.036, xi=1.790, r0=2.884, the same container
# and quench tolerances as scripts/metal_h2h/run_darby.py. Both arms of
# a head-to-head start from the coordinates `h2h_metal coords` writes.
# Each row carries the anneal revision it was measured on. A re-run
# replaces rows of that revision and leaves the others in seeds.csv.
set -euo pipefail
if [[ $(hostname) != bench ]]; then
  echo "uts hostname is $(hostname), expected bench" >&2
  exit 2
fi
unset H2H_GENERAL H2H_BASIN_STRIDE H2H_BUDGET H2H_BEST H2H_REFERENCE \
  H2H_PRIORS H2H_REPLICAS H2H_TARGET H2H_TOL H2H_SEED H2H_SYSTEM H2H_ROW
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/target/release/examples/h2h_metal}
GMIN=${GMIN:-/work/metal-darby/bin/GMIN}
WORK=${WORK:-/tmp/gold-bench}
REF38=-130.54001708
REF55=-191.03085280
REV=${GOLD_REVISION:-$(git -C "$ROOT" rev-parse --short=12 HEAD)}
mkdir -p "$WORK"
# A, p, q, xi, r0. Same printing as run_darby.py's {:.10g}.
GOLD_ROW="0.2061 10.229 4.036 1.79 2.884"

echo "GOLD_START $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
echo "HEAD $(git -C "$ROOT" rev-parse HEAD)"
echo "BIN $BIN"
echo "GMIN $GMIN"

publish() {
  mkdir -p "$ROOT/bench/gold-gupta"
  python3 - "$WORK/rows.csv" "$ROOT/bench/gold-gupta/seeds.csv" "$REV" << 'PY'
import csv, sys
fresh, stored, revision = sys.argv[1], sys.argv[2], sys.argv[3]
fields = ["system","code","seed","budget","hit","evals_to_hit","charged","wall_s","best","pot_s","outside_s","revision"]
kept = []
try:
    with open(stored, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("revision") == revision:
                continue
            kept.append({key: row.get(key, "") for key in fields})
except FileNotFoundError:
    pass
with open(fresh, newline="") as handle:
    fresh_rows = list(csv.DictReader(handle))
with open(stored, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(kept)
    writer.writerows(fresh_rows)
PY
  chmod a+r "$WORK/rows.csv" "$ROOT/bench/gold-gupta/seeds.csv" || true
}

anneal_one() {
  local system=$1 seed=$2
  taskset -c 0 "$BIN" run "$system" "$seed"
}

gmin_one() {
  local system=$1 n=$2 seed=$3 budget=$4 target=$5
  local d=$WORK/gmin/$system/$seed
  rm -rf "$d"
  mkdir -p "$d"
  taskset -c 0 "$BIN" coords "$system" "$seed" "$d/coords"
  cat > "$d/data" << EOF
BGUPTAT ${n} ${GOLD_ROW}
BGUPTATAB ${GOLD_ROW}
BGUPTATBB ${GOLD_ROW}
SLOPPYCONV 5.0D-3
TIGHTCONV 1.0D-3
EDIFF 0.003
MAXIT 600 500
SAVE 2
CENTRE
RADIUS 50.0
TEMPERATURE 1.0
STEPS 90000000 1.0
STEP 0.40 0.51
NEWRESTART 40
RANSEED ${seed}
EOF
  (
    cd "$d"
    if [[ -n $target ]]; then
      H2H_TARGET=$target H2H_TOL=0.001 \
        H2H_BUDGET=$budget H2H_SEED=$seed H2H_SYSTEM=$system \
        H2H_ROW=$d/row.txt taskset -c 0 "$GMIN" > stdout.txt
    else
      env -u H2H_TARGET \
        H2H_BUDGET=$budget H2H_SEED=$seed H2H_SYSTEM=$system \
        H2H_ROW=$d/row.txt taskset -c 0 "$GMIN" > stdout.txt
    fi
  )
  if [[ ! -s $d/row.txt ]]; then
    echo "GMIN produced no row for $system seed $seed" >&2
    tail -n 50 "$d/stdout.txt" >&2 || true
    exit 1
  fi
  cat "$d/row.txt"
}

record_anneal() {
  local line
  line=$(anneal_one "$@" | awk -F, 'NF>=11 && $1!="system" {print}')
  echo "$line" | awk -F, -v rev="$REV" 'BEGIN{OFS=","} {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,rev}' | tee -a "$WORK/rows.csv"
  publish
}

record_gmin() {
  local line
  line=$(gmin_one "$@")
  python3 -c '
import sys
line=sys.argv[1]
d={}
for tok in line.split():
    if "=" in tok:
        k,v=tok.split("=",1); d[k]=v
wall=float(d["wall_s"]); pot=float(d["pot_s"])
hit="true" if d["hit"]=="1" else "false"
print(",".join([d["system"],"gmin",d["seed"],d["budget"],hit,d["evals_to_hit"],d["charged"],f"{wall:.6f}",d["best"],f"{pot:.6f}",f"{wall-pot:.6f}",sys.argv[2]]))
' "$line" "$REV" | tee -a "$WORK/rows.csv"
  publish
}

run_pair() {
  local system=$1 n=$2 budget=$3 target=$4 seed=$5
  echo "ANNEAL $system $seed"
  record_anneal "$system" "$seed"
  echo "GMIN $system $seed"
  record_gmin "$system" "$n" "$seed" "$budget" "$target"
}

if [[ ${1:-} == --one ]]; then
  system=${2:?system}
  arm=${3:?arm}
  seed=${4:?seed}
  case $system in
    au38) n=38; budget=400000; target=$REF38 ;;
    au55) n=55; budget=1000000; target=$REF55 ;;
    au75) n=75; budget=4000000; target= ;;
    au98) n=98; budget=4000000; target= ;;
    *) echo "unknown system $system" >&2; exit 2 ;;
  esac
  if [[ $arm == anneal ]]; then
    anneal_one "$system" "$seed"
  elif [[ $arm == gmin ]]; then
    gmin_one "$system" "$n" "$seed" "$budget" "$target"
  else
    echo "unknown arm $arm" >&2
    exit 2
  fi
  exit 0
fi

if [[ ${1:-} == --from-large ]]; then
  for seed in $(seq 2000 2011); do
    run_pair au75 75 4000000 "" "$seed"
  done
  for seed in $(seq 2000 2011); do
    run_pair au98 98 4000000 "" "$seed"
  done
  echo "GOLD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

echo "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,revision" > "$WORK/rows.csv"

for seed in $(seq 2000 2011); do
  run_pair au38 38 400000 "$REF38" "$seed"
done

for seed in $(seq 2000 2011); do
  run_pair au75 75 4000000 "" "$seed"
done
for seed in $(seq 2000 2011); do
  run_pair au98 98 4000000 "" "$seed"
done

echo "GOLD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
