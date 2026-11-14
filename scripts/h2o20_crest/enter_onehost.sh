#!/usr/bin/env bash
# Copyright (c) anneal developers
# GFN2 (H2O)20, anneal against CREST, one call cap, one core.
# One UTS namespace, hostname bench, core 0.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${H2O20_WORK:-/work/h2o20-fix}
STARTS=${H2O20_STARTS:-/work/h2o20-onehost/runs/starts}
CREST=${CREST_BIN:-/work/bin/crest}
XTB=${XTB_BIN:-/work/bin/xtb}
# The shell may already export an older engine that has no rgpot_xtb_reset.
ENGINE=${H2O20_ENGINE:-/work/lib/engine/libxtb_engine.so}
PYTHON=${H2O20_PYTHON:-$(command -v python3)}
ANNEAL=${H2O20_ANNEAL:-/work/h2o20-fix-build/release/examples/gfn2_water_h2h}
DRIVER=${H2O20_DRIVER:-/work/h2o20-fix-build/release/examples/gfn2_water_crest_driver}
CODE_REVISION=${H2O20_CODE_REVISION:-746ec03fc290730a35d5478f14505222f3296e77}
CORE=${H2O20_CORE:-0}

for need in "$CREST" "$XTB" "$ENGINE" "$ANNEAL" "$DRIVER" "$STARTS"; do
  if [[ ! -e $need ]]; then
    echo "missing $need" >&2
    exit 1
  fi
done

mkdir -p "$WORK/home" "$WORK/tmp" "$WORK/stamps"
printf '%s\n' "$CODE_REVISION" > "$WORK/anneal-head"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  "hostname bench; unset LD_LIBRARY_PATH; export HOME='$WORK/home' TMPDIR='$WORK/tmp'; \
   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1; \
   export RGPOT_XTB_ENGINE='$ENGINE'; \
   export ANNEAL_SOURCE_COMMIT=\$(cat '$WORK/anneal-head'); \
   exec taskset -c $CORE '$PYTHON' -u '$HERE/run_onehost.py' \
     --anneal '$ANNEAL' --driver '$DRIVER' --crest '$CREST' --xtb '$XTB' \
     --engine '$ENGINE' --head-file '$WORK/anneal-head' --core '$CORE' \
     --out '$WORK/runs' --starts '$STARTS' \"\$@\"" timed-sh "$@"
