#!/usr/bin/env bash
# Copyright (c) anneal developers
# GFN2-xTB Si10 and C10, anneal's general arm against CREST, one UTS
# namespace, hostname bench, core 0, working directory under /work.
# Arguments go to gfn2_crest.py. GFN2_CODES=anneal-general reruns the
# anneal arm alone and keeps the CREST rows already in the rows file.
# GFN2_ROWS and GFN2_SUMMARY move the rows and the summary off
# bench/covalent-h2h/gfn2.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=$ROOT/bench/covalent-h2h/gfn2
WORK=${GFN2_WORK:-/work/covalent-h2h/gfn2}
CREST=${CREST_BIN:-/work/crest/crest/crest}
XTB=${XTB_BIN:-/work/mamba/xtb/bin/xtb}
ENGINE=${RGPOT_XTB_ENGINE:-/work/rgpot-xtb-build/CppCore/rgpot/XTBPot/libxtb_engine.so}
LIB=${GFN2_LD_LIBRARY_PATH:-/work/mamba/xtb/lib}
TARGET=${CARGO_TARGET_DIR:-/work/pin/tip}
CODES=${GFN2_CODES:-anneal-general crest}
ROWS=${GFN2_ROWS:-$OUT/seeds.csv}
SUMMARY=${GFN2_SUMMARY:-$OUT/summary.csv}

for need in "$CREST" "$XTB" "$ENGINE"; do
  if [[ ! -e $need ]]; then
    echo "missing $need" >&2
    exit 1
  fi
done
(cd "$ROOT" && CARGO_TARGET_DIR=$TARGET cargo +1.88.0 build --release --features rgpot-ex \
  --example gfn2_covalent_h2h)

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$(dirname "$ROWS")" "$(dirname "$SUMMARY")"
cp -a "$TARGET/release/examples/gfn2_covalent_h2h" "$WORK/bin/gfn2_covalent_h2h"
git -C "$ROOT" rev-parse HEAD > "$WORK/anneal-head"
if [[ $CODES == anneal-general && -s $ROWS ]]; then
  /usr/bin/python3 - "$ROWS" << 'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline="") as handle:
    reader = csv.DictReader(handle)
    rows = [row for row in reader if row["code"] != "anneal-general"]
    fields = reader.fieldnames
with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
PY
fi

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home TMPDIR=$WORK/tmp
export RGPOT_XTB_ENGINE=$ENGINE
export LD_LIBRARY_PATH=$LIB
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd $WORK
exec taskset -c 0 /usr/bin/python3 -u $HERE/gfn2_crest.py \\
  --anneal $WORK/bin/gfn2_covalent_h2h --crest $CREST --xtb $XTB \\
  --out $WORK/runs --rows $ROWS --summary $SUMMARY \\
  --codes $CODES "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu "$0" "$@"' "$WORK/bin/inside.sh" "$@"
