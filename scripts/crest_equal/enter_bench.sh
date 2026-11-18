#!/usr/bin/env bash
# Copyright (c) anneal developers
# (H2O)6 GFN2, anneal against CREST. Hostname bench, one core.
# Binaries come from the environment: CREST_BIN, XTB_BIN, RGPOT_XTB_ENGINE,
# ANNEAL_BIN, DRIVER_BIN. Extra arguments go to run_panel.py.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
: "${CREST_BIN:?}"
: "${XTB_BIN:?}"
: "${RGPOT_XTB_ENGINE:?}"
: "${ANNEAL_BIN:?}"
: "${DRIVER_BIN:?}"
PYTHON=${PYTHON:-python3}
CORE=${CREST_EQUAL_CORE:-0}

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c 'hostname bench; exec "$@"' _ env \
  PATH="$PATH" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  RGPOT_XTB_ENGINE="$RGPOT_XTB_ENGINE" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  GOTO_NUM_THREADS=1 \
  OMP_THREAD_LIMIT=1 \
  taskset -c "$CORE" "$PYTHON" -u "$HERE/run_panel.py" \
  --anneal "$ANNEAL_BIN" \
  --driver "$DRIVER_BIN" \
  --crest "$CREST_BIN" \
  --xtb "$XTB_BIN" \
  --engine "$RGPOT_XTB_ENGINE" \
  --core "$CORE" \
  --rows "$ROOT/bench/crest-equal/rows.csv" \
  --summary "$ROOT/bench/crest-equal/summary.csv" \
  --stamp "$ROOT/bench/crest-equal/panel.stamp" \
  "$@"
