#!/usr/bin/env bash
# Copyright (c) anneal developers
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUN="$HERE/run_campaign.py"
CORE=${WATER192_CORE:-0}
TOOLCHAIN=${WATER192_TOOLCHAIN:-1.88.0}
FEATURES=${WATER192_FEATURES:-rgpot-ex,featomic,bank-rpc,ira}
IRA_ROOT=${IRA_ROOT:-$ROOT/.ci/ira}
IRA_LIB_DIR=${IRA_LIB_DIR:-$IRA_ROOT/lib}

cd "$ROOT"

if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  rm -rf "$IRA_ROOT/src/Obj"
  mkdir -p "$IRA_ROOT/src/Obj" "$IRA_ROOT/include" "$IRA_ROOT/lib"
  (cd "$IRA_ROOT/src" && LIBLAPACK='' make shlib FFLAGS='-fPIC -cpp -O3 -ffree-line-length-512 -funroll-loops')
  gfortran -o "$IRA_ROOT/lib/libira.so.new" -shared \
    -J"$IRA_ROOT/include" -I"$IRA_ROOT/include" \
    "$IRA_ROOT/src/Obj"/*.o \
    -Wl,-soname,libira.so
  mv -f "$IRA_ROOT/lib/libira.so.new" "$IRA_ROOT/lib/libira.so"
fi

export IRA_LIB_DIR
export LD_LIBRARY_PATH="$IRA_LIB_DIR:${LD_LIBRARY_PATH:-}"
CARGO_TARGET_DIR=$(cargo +"$TOOLCHAIN" metadata --format-version 1 --no-deps \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
export CARGO_TARGET_DIR

cargo +"$TOOLCHAIN" build --release --locked --features "$FEATURES" \
  --example water_hexamer \
  --bin water_hexamer_tools

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export ASE_PYTHON=${ASE_PYTHON:-python3}
export WATER192_CORE="$CORE"
export WATER192_WORK=${WATER192_WORK:-target-water192/campaign}

if [[ -n "${WATER192_HOST_ALIAS:-}" ]] && command -v unshare >/dev/null 2>&1; then
  if sudo -n true >/dev/null 2>&1; then
    if ! grep -Eq "(^|[[:space:]])${WATER192_HOST_ALIAS}([[:space:]]|$)" /etc/hosts; then
      echo "127.0.0.1 ${WATER192_HOST_ALIAS}" | sudo -n tee -a /etc/hosts >/dev/null || true
    fi
    RUN_USER=${WATER192_RUN_USER:-${USER:-ubuntu}}
    exec sudo -n unshare --uts sh -c '
      hostname "$1"
      shift
      run_user="$1"
      shift
      exec sudo -n -u "$run_user" env "$@"
    ' _ "$WATER192_HOST_ALIAS" "$RUN_USER" \
      IRA_LIB_DIR="$IRA_LIB_DIR" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
      WATER192_CORE="$WATER192_CORE" \
      WATER192_WORK="$WATER192_WORK" \
      ASE_PYTHON="$ASE_PYTHON" \
      CREST_BIN="${CREST_BIN:-}" \
      XTB_BIN="${XTB_BIN:-}" \
      taskset -c "$CORE" python3 -u "$RUN" "$@"
  fi
fi

exec taskset -c "$CORE" python3 -u "$RUN" "$@"
