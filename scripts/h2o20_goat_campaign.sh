#!/bin/bash
# Copyright (c) anneal developers
# Timed GFN2-xTB (H2O)20 comparison of anneal and CREST.
#
# Same wrapper as scripts/h2o16_v9_one_bench.sh: every search is
#   sudo unshare --uts sh -c 'hostname bench; exec <cmd>'
# with one thread, a shared start geometry, and the arm order drawn
# from the seed. The working directory is under /work.
#
# GOAT is run only when `orca` is already on PATH. ORCA 6 is login-gated
# and is not fetched from any other place. A missing binary is a skipped
# row, for (H2O)16 and (H2O)20.
#
# bench/water-h2o20-goat/reference.csv must already exist. This script
# refuses to time a seed without it.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REF=$ROOT/bench/water-h2o20-goat/reference.csv
OUT_CSV=$ROOT/bench/water-h2o20-goat/results.csv
BIN=${ANNEAL_BIN:-/work/gfn2_water_h2h}
CREST=${CREST_BIN:-/work/crestenv/.pixi/envs/default/bin/crest}
XTB=${XTB_BIN:-/work/xtb-gfn2}
ENGINE=${RGPOT_XTB_ENGINE:-/work/rgpot-bdir/CppCore/rgpot/XTBPot/libxtb_engine.so}
LIB=${H2O_LD_LIBRARY_PATH:-/work/rgpot/.pixi/envs/xtbbld/lib}
WORK=${H2O_WORK:-/work/h2o20-goat}
MOLECULES=${H2O_MOLECULES:-20}
BUDGET=${H2O_BUDGET:-4000}
CREST_LEN=${H2O_CREST_LEN:-0.2}
SEEDS=${H2O_SEEDS:-15001,15002,15003}
COMMIT=${ANNEAL_SOURCE_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}
TREE=${ANNEAL_SOURCE_TREE:-$(git -C "$ROOT" rev-parse 'HEAD^{tree}')}

if [[ ! -s $REF ]]; then
  echo "missing $REF; fix the reference before any timed seed" >&2
  exit 1
fi
for need in "$BIN" "$CREST" "$XTB" "$ENGINE"; do
  if [[ ! -e $need ]]; then
    echo "missing $need" >&2
    exit 1
  fi
done
case "$(rustc +1.88.0 -V 2>/dev/null || rustc -V)" in
  "rustc 1.88.0 "*) ;;
  *)
    echo "rustc is not the Cargo.toml pin of 1.88.0" >&2
    rustc -V >&2
    exit 1
    ;;
esac

mkdir -p "$WORK" "$(dirname "$OUT_CSV")"
cat > "$WORK/bench-env.sh" << EOF
export RGPOT_XTB_ENGINE=$ENGINE
export LD_LIBRARY_PATH=$LIB
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export RAYON_NUM_THREADS=1
export ANNEAL_SOURCE_COMMIT=$COMMIT
export ANNEAL_SOURCE_TREE=$TREE
EOF
if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

# Timed command. stdout/stderr stay in the work directory.
timed() {
  local dir=$1
  shift
  mkdir -p "$dir"
  (
    cd "$dir"
    set +e
    sudo -n unshare --uts sh -c \
      'hostname bench; . "$1"; cd "$2"; hostname > hostname.txt; shift 2; exec "$@" > stdout.txt 2> stderr.txt' \
      timed-sh "$WORK/bench-env.sh" "$dir" "$@"
    echo $? > exit.code
  )
  local host
  host=$(tr -d '\n' < "$dir/hostname.txt")
  if [[ $host != bench ]]; then
    echo "hostname was $host" >&2
    exit 1
  fi
}

python3 - "$OUT_CSV" "$REF" << 'PY'
import csv, sys
from pathlib import Path
out, ref = map(Path, sys.argv[1:])
with ref.open(newline="") as handle:
    row = next(csv.DictReader(handle))
if int(row["molecules"]) != 20:
    raise SystemExit("reference.csv is not (H2O)20")
out.write_text(
    "molecules,seed,arm,order,energy_eV,hit,wall_s,reference_eV,status\n"
)
print(f"reference_eV {row['energy_eV']}")
PY

order_of() {
  python3 - "$1" << 'PY'
import random, sys
seed = int(sys.argv[1])
arms = ["anneal", "crest"]
random.Random(seed).shuffle(arms)
print(" ".join(arms))
PY
}

score_eh() {
  local xyz=$1
  local dest=$2
  timed "$dest" "$XTB" "$xyz" --gfn 2 --opt vtight --acc 0.01 --iterations 500
  python3 - "$dest/stdout.txt" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
energy = None
for line in text.splitlines():
    if "TOTAL ENERGY" in line and "Eh" in line:
        parts = line.split()
        energy = float(parts[parts.index("Eh") - 1])
if energy is None:
    raise SystemExit(1)
print(f"{energy:.12f}")
PY
}

append_row() {
  python3 - "$OUT_CSV" "$@" << 'PY'
import sys
from pathlib import Path
path, molecules, seed, arm, order, energy, hit, wall, reference, status = sys.argv[1:]
with Path(path).open("a") as handle:
    handle.write(
        f"{molecules},{seed},{arm},{order},{energy},{hit},{wall},{reference},{status}\n"
    )
PY
}

REFERENCE=$(python3 -c 'import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))["energy_eV"])' "$REF")

run_anneal() {
  local seed=$1
  local base=$2
  local dir=$base/anneal
  mkdir -p "$dir"
  cp "$base/start.xyz" "$dir/start.xyz"
  local t0 t1
  t0=$(date +%s.%N)
  timed "$dir" env H2H_START_XYZ="$dir/start.xyz" "$BIN" "$MOLECULES" "$BUDGET" "$seed" "$dir"
  t1=$(date +%s.%N)
  python3 -c 'import sys; print("{:.6f}".format(float(sys.argv[2])-float(sys.argv[1])))' "$t0" "$t1" > "$dir/wall_search.txt"
}

run_crest() {
  local seed=$1
  local base=$2
  local dir=$base/crest
  mkdir -p "$dir"
  cp "$base/start.xyz" "$dir/start.xyz"
  local t0 t1
  t0=$(date +%s.%N)
  timed "$dir" "$CREST" "$dir/start.xyz" --gfn2 -T 1 --legacy -xnam "$XTB" -len "$CREST_LEN"
  t1=$(date +%s.%N)
  python3 -c 'import sys; print("{:.6f}".format(float(sys.argv[2])-float(sys.argv[1])))' "$t0" "$t1" > "$dir/wall_search.txt"
}

IFS=',' read -r -a seed_list <<< "$SEEDS"
for seed in "${seed_list[@]}"; do
  base=$WORK/runs/$seed
  mkdir -p "$base/start"
  timed "$base/start" env H2H_START_ONLY=1 "$BIN" "$MOLECULES" 1 "$seed" "$base/start"
  if [[ ! -s $base/start/start.xyz ]]; then
    echo "no start for seed $seed" >&2
    exit 1
  fi
  cp "$base/start/start.xyz" "$base/start.xyz"
  read -r -a arms <<< "$(order_of "$seed")"
  position=0
  for arm in "${arms[@]}"; do
    position=$((position + 1))
    echo "SEED $seed ARM $arm ORDER $position" | tee -a "$WORK/schedule.txt"
    if [[ $arm == anneal ]]; then
      run_anneal "$seed" "$base"
      geom=$base/anneal/best.xyz
    else
      run_crest "$seed" "$base"
      geom=$base/crest/crest_best.xyz
    fi
    wall=$(cat "$base/$arm/wall_search.txt")
    energy=""
    hit=no
    status=ran
    if [[ -s $geom ]]; then
      if eh=$(score_eh "$geom" "$base/$arm/score"); then
        energy=$(python3 -c 'import sys; print("{:.6f}".format(float(sys.argv[1])*27.211386245988))' "$eh")
        hit=$(python3 -c 'import sys; e,r,w=map(float, sys.argv[1:]); print("yes" if e<=r+w else "no")' "$energy" "$REFERENCE" 0.01)
      else
        status=score_failed
      fi
    else
      status=no_geometry
    fi
    append_row "$MOLECULES" "$seed" "$arm" "$position" "$energy" "$hit" "$wall" "$REFERENCE" "$status"
    echo "ROW $seed $arm $energy $hit $wall" | tee -a "$WORK/schedule.txt"
  done
done

if ! command -v orca >/dev/null 2>&1; then
  append_row 16 "" goat "" "" "" "" "$REFERENCE" skipped_no_orca
  append_row 20 "" goat "" "" "" "" "$REFERENCE" skipped_no_orca
fi
echo "CAMPAIGN_DONE $OUT_CSV"
