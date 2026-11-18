#!/usr/bin/env bash
# Copyright (c) anneal developers
# GFN2 (H2O)16, anneal against CREST on one objective with one call cap.
# One UTS namespace, hostname bench, one core (H2O16_CORE, default 0).
# Arguments go to run_capped.py after the defaults, so --seed, --arm,
# --rows and --out override them. With no arguments the rows and the
# summary are appended under bench/h2o16-crest.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${H2O16_WORK:-/work/h2o16-crest-capped}
CREST=${CREST_BIN:-/work/crest/crest/crest}
XTB=${XTB_BIN:-/work/mamba/xtb/bin/xtb}
ENGINE=${RGPOT_XTB_ENGINE:-/work/rgpot-xtb-build/CppCore/rgpot/XTBPot/libxtb_engine.so}
LIB=${H2O16_LD_LIBRARY_PATH:-$(dirname "$XTB")/../lib}
PYTHON=${H2O16_PYTHON:-$(command -v python3)}
TARGET=${CARGO_TARGET_DIR:-/work/pin/tip}
CORE=${H2O16_CORE:-0}

for need in "$CREST" "$XTB" "$ENGINE"; do
  if [[ ! -e $need ]]; then
    echo "missing $need" >&2
    exit 1
  fi
done
"$PYTHON" -c 'import scipy.stats'
(cd "$ROOT" && CARGO_TARGET_DIR=$TARGET cargo +1.88.0 build --release --features rgpot-ex \
  --example gfn2_water_h2h --example gfn2_water_crest_driver)

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp"
cp -a "$TARGET/release/examples/gfn2_water_h2h" "$TARGET/release/examples/gfn2_water_crest_driver" \
  "$WORK/bin/"
cp -a "$HERE/run_capped.py" "$ROOT/scripts/target_tol.py" "$WORK/bin/"
git -C "$ROOT" rev-parse HEAD > "$WORK/anneal-head"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home TMPDIR=$WORK/tmp
export LD_LIBRARY_PATH=$LIB
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export ANNEAL_SOURCE_COMMIT=\$(cat $WORK/anneal-head)
cd $WORK
if [ "\$#" -eq 0 ]; then
  set -- --rows $ROOT/bench/h2o16-crest/capped-rows.csv \\
    --summary $ROOT/bench/h2o16-crest/capped-summary.csv
fi
exec taskset -c $CORE $PYTHON -u $WORK/bin/run_capped.py \\
  --anneal $WORK/bin/gfn2_water_h2h --driver $WORK/bin/gfn2_water_crest_driver \\
  --crest $CREST --xtb $XTB --engine $ENGINE --head-file $WORK/anneal-head \\
  --core $CORE --out $WORK/runs/\$(cat $WORK/anneal-head) "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu "$0" "$@"' "$WORK/bin/inside.sh" "$@"
