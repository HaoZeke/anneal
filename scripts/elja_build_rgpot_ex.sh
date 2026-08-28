#!/usr/bin/env bash
# Build in-process rgpot engines (libxtb_engine.so, librgpot_cuh2.so) and
# the molecular_cluster / slab_adsorption examples. Not potserv.
#
# Engines: pixi meson on an srun (conda compilers, portable).
# Rust examples: cargo on the login image (compute has no glibc-devel).
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
RGPOT=${RGPOT_ROOT:-$HOME/rgpot}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
RGPOT_EXPECTED_COMMIT=${RGPOT_EXPECTED_COMMIT:?set RGPOT_EXPECTED_COMMIT}
BDIR=${RGPOT_BDIR:-$RGPOT/bdir-xtb-${RGPOT_EXPECTED_COMMIT:0:12}}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}

dynamic_symbol_is_defined() {
  local library=$1 symbol=$2
  nm -D --defined-only "$library" |
    awk -v symbol="$symbol" '$NF == symbol { found = 1 } END { exit found ? 0 : 1 }'
}

if [[ -z ${SLURM_JOB_ID:-} && ${1:-} != login-cargo ]]; then
  echo "elja_build_rgpot_ex.sh: engines under srun, cargo as login-cargo" >&2
  exit 1
fi

verify_rgpot_source() {
  local head
  head=$(git -C "$RGPOT" rev-parse HEAD)
  if [[ $head != "$RGPOT_EXPECTED_COMMIT" ]]; then
    echo "RGPOT_EXPECTED_COMMIT=$RGPOT_EXPECTED_COMMIT does not match rgpot HEAD=$head" >&2
    exit 2
  fi
  if ! git -C "$RGPOT" diff --quiet HEAD --; then
    echo "rgpot tracked source differs from HEAD=$head" >&2
    git -C "$RGPOT" status --short >&2
    exit 2
  fi
  local source_status
  source_status=$(git -C "$RGPOT" status --porcelain=v1 --untracked-files=all)
  if [[ -n $source_status ]]; then
    echo "rgpot source tree differs from HEAD=$head" >&2
    git -C "$RGPOT" status --short >&2
    exit 2
  fi
  if [[ ! -s $RGPOT/pixi.lock ]]; then
    echo "missing tracked rgpot Pixi lockfile: $RGPOT/pixi.lock" >&2
    exit 2
  fi
}

build_engines() {
  if [[ ! -x $PIXI ]]; then
    echo "missing pixi at $PIXI" >&2
    exit 1
  fi
  verify_rgpot_source
  cd "$RGPOT"
  # xtbbld = default compilers + conda-forge xtb.
  "$PIXI" install --locked -e xtbbld
  if [[ ! -f $BDIR/build.ninja ]]; then
    "$PIXI" run --locked -e xtbbld meson setup "$BDIR" \
      -Dwith_xtb=true \
      -Dwith_fortran_pots=enabled \
      -Dwith_tests=false \
      -Dwith_rpc=false \
      -Dwith_cache=false \
      --buildtype=release
  fi
  "$PIXI" run --locked -e xtbbld meson compile -C "$BDIR"
  local xtbso="" cuh2so
  cuh2so=$BDIR/CppCore/rgpot/fortran/librgpot_cuh2.so
  "$PIXI" run --locked -e xtbbld \
    "$ROOT/scripts/link_rgpot_cuh2_engine.sh" \
    "$RGPOT" \
    "$BDIR" \
    "$ROOT/scripts/rgpot_cuh2.exports" \
    "$cuh2so"
  if [[ -e $BDIR/CppCore/rgpot/XTBPot/libxtb_engine.so ]]; then
    xtbso=$BDIR/CppCore/rgpot/XTBPot/libxtb_engine.so
  elif [[ -e $BDIR/CppCore/libxtb_engine.so ]]; then
    xtbso=$BDIR/CppCore/libxtb_engine.so
  fi
  if [[ -z $xtbso ]]; then
    echo "libxtb_engine.so missing under $BDIR" >&2
    exit 1
  fi
  dynamic_symbol_is_defined "$cuh2so" rgpot_cuh2_force || {
    echo "$cuh2so has no rgpot_cuh2_force" >&2
    nm -D --defined-only "$cuh2so" >&2
    exit 1
  }
  dynamic_symbol_is_defined "$xtbso" rgpot_xtb_force || {
    echo "$xtbso has no rgpot_xtb_force" >&2
    exit 1
  }
  mkdir -p "$ROOT/engines"
  cp "$xtbso" "$ROOT/engines/libxtb_engine.so"
  cp "$cuh2so" "$ROOT/engines/librgpot_cuh2.so"
  git -C "$RGPOT" rev-parse HEAD >"$ROOT/RGPOT_SOURCE_COMMIT"
  sha256sum "$RGPOT/pixi.lock" | awk '{print $1}' >"$ROOT/RGPOT_PIXI_LOCK_SHA256"
  echo "ENGINES_OK xtb=$xtbso cuh2=$cuh2so"
}

build_examples() {
  verify_rgpot_source
  export PATH="${HOME}/.cargo/bin:${PATH}"
  cd "$ROOT"
  if ! git diff --quiet HEAD --; then
    echo "refusing build: tracked source differs from HEAD" >&2
    git status --short >&2
    exit 2
  fi
  cargo build --offline --locked --release --features rgpot-ex,featomic,bank-rpc \
    --example molecular_cluster --example slab_adsorption --example bank_server \
    --example bank_peek
  local mol slab bank peek
  mol=$ROOT/target/release/examples/molecular_cluster
  slab=$ROOT/target/release/examples/slab_adsorption
  bank=$ROOT/target/release/examples/bank_server
  peek=$ROOT/target/release/examples/bank_peek
  [[ -x $mol && -x $slab && -x $bank && -x $peek ]] || {
    echo "examples missing" >&2
    exit 1
  }
  [[ -e $ROOT/engines/libxtb_engine.so && -e $ROOT/engines/librgpot_cuh2.so ]] || {
    echo "molecular engines missing below $ROOT/engines" >&2
    exit 1
  }
  if [[ $(<RGPOT_SOURCE_COMMIT) != "$RGPOT_EXPECTED_COMMIT" ]]; then
    echo "staged rgpot source record does not match $RGPOT_EXPECTED_COMMIT" >&2
    exit 2
  fi
  rgpot_lock_sha256=$(sha256sum "$RGPOT/pixi.lock" | awk '{print $1}')
  if [[ $(<RGPOT_PIXI_LOCK_SHA256) != "$rgpot_lock_sha256" ]]; then
    echo "staged rgpot Pixi lock digest does not match $RGPOT/pixi.lock" >&2
    exit 2
  fi
  git rev-parse HEAD >SOURCE_COMMIT
  {
    printf 'source_commit=%s\n' "$(<SOURCE_COMMIT)"
    printf 'rgpot_source_commit=%s\n' "$(<RGPOT_SOURCE_COMMIT)"
    printf 'rgpot_pixi_lock_sha256=%s\n' "$(<RGPOT_PIXI_LOCK_SHA256)"
    printf 'pixi=%s\n' "$("$PIXI" --version)"
  } >MOLSLAB_BUILD_PROVENANCE
  sha256sum \
    target/release/examples/molecular_cluster \
    target/release/examples/slab_adsorption \
    target/release/examples/bank_server \
    target/release/examples/bank_peek \
    engines/libxtb_engine.so \
    engines/librgpot_cuh2.so \
    scripts/link_rgpot_cuh2_engine.sh \
    scripts/rgpot_cuh2.exports \
    SOURCE_COMMIT \
    RGPOT_SOURCE_COMMIT \
    RGPOT_PIXI_LOCK_SHA256 \
    MOLSLAB_BUILD_PROVENANCE \
    >MOLSLAB_BUILD_SHA256SUMS
  sha256sum -c MOLSLAB_BUILD_SHA256SUMS
  echo "EXAMPLES_OK $mol $slab $bank $peek"
}

smoke() {
  verify_rgpot_source
  (cd "$ROOT" && sha256sum -c MOLSLAB_BUILD_SHA256SUMS)
  export RGPOT_XTB_ENGINE=$ROOT/engines/libxtb_engine.so
  export RGPOT_CUH2_LIBRARY=$ROOT/engines/librgpot_cuh2.so
  export LD_LIBRARY_PATH="${RGPOT}/.pixi/envs/xtbbld/lib:${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
  echo "SMOKE water dimer"
  "$ROOT/target/release/examples/molecular_cluster" 2 20 1
  echo "SMOKE cuh2 FCC slab"
  "$ROOT/target/release/examples/slab_adsorption" \
    "$ROOT/examples/fixtures/cuh2_fcc_slab.con" 15 1
  echo "SMOKE_OK"
}

case ${1:-engines} in
  engines) build_engines ;;
  login-cargo) build_examples ;;
  smoke) smoke ;;
  all)
    build_engines
    build_examples
    smoke
    ;;
  *)
    echo "usage: $0 engines|login-cargo|smoke|all" >&2
    exit 2
    ;;
esac
