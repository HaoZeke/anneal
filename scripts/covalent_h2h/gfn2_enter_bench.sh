#!/usr/bin/env bash
# Copyright (c) anneal developers
# GFN2-xTB silicon and carbon clusters, anneal's general arm against
# CREST, one UTS namespace, hostname bench, one core (GFN2_CORE, default
# 0), working directory under /work. Arguments go to gfn2_crest.py;
# GFN2_CODES picks the arms. Rows and summaries are appended to rows.csv
# and summaries.csv, never rewritten, and seeds.csv is read as history.
# GFN2_ROWS and GFN2_SUMMARY move them off bench/covalent-h2h/gfn2. A tree
# with uncommitted changes to the build is refused, so the revision names
# the binary.
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
ROWS=${GFN2_ROWS:-$OUT/rows.csv}
SUMMARY=${GFN2_SUMMARY:-$OUT/summaries.csv}
CORE=${GFN2_CORE:-0}
PY=${GFN2_PYTHON:-/usr/bin/python3}

if [[ -n $(git -C "$ROOT" status --porcelain -- src examples Cargo.toml Cargo.lock build.rs) ]]; then
  echo "uncommitted changes to the anneal build" >&2
  exit 1
fi
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

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home TMPDIR=$WORK/tmp
export RGPOT_XTB_ENGINE=$ENGINE
export LD_LIBRARY_PATH=$LIB
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PANEL_PYTHON=$PY
cd $WORK
taskset -c $CORE $PY $ROOT/scripts/session_metadata.py \\
  --output $OUT/sessions.jsonl --campaign gfn2-crest-full --root $ROOT -- \\
  snakemake -s workflow/Snakefile covalent_gfn2_crest --cores 1 --force
exec taskset -c $CORE $PY -u $HERE/gfn2_crest.py \\
  --anneal $WORK/bin/gfn2_covalent_h2h --crest $CREST --xtb $XTB \\
  --engine $ENGINE --head-file $WORK/anneal-head \\
  --out $WORK/runs --rows $ROWS --history $OUT/seeds.csv --summary $SUMMARY \\
  --codes $CODES "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu "$0" "$@"' "$WORK/bin/inside.sh" "$@"
