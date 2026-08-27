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
BDIR=${RGPOT_BDIR:-$RGPOT/bdir-xtb}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}

if [[ -z ${SLURM_JOB_ID:-} && ${1:-} != login-cargo ]]; then
  echo "elja_build_rgpot_ex.sh: engines under srun, cargo as login-cargo" >&2
  exit 1
fi

build_engines() {
  if [[ ! -x $PIXI ]]; then
    echo "missing pixi at $PIXI" >&2
    exit 1
  fi
  cd "$RGPOT"
  # xtbbld = default compilers + conda-forge xtb.
  "$PIXI" install -e xtbbld
  if [[ ! -f $BDIR/build.ninja ]]; then
    "$PIXI" run -e xtbbld meson setup "$BDIR" \
      -Dwith_xtb=true \
      -Dwith_fortran_pots=enabled \
      -Dwith_tests=false \
      -Dwith_rpc=false \
      -Dwith_cache=false \
      --buildtype=release
  fi
  "$PIXI" run -e xtbbld meson compile -C "$BDIR"
  local xtbso="" cuh2so=""
  if [[ -e $BDIR/CppCore/rgpot/XTBPot/libxtb_engine.so ]]; then
    xtbso=$BDIR/CppCore/rgpot/XTBPot/libxtb_engine.so
  elif [[ -e $BDIR/CppCore/libxtb_engine.so ]]; then
    xtbso=$BDIR/CppCore/libxtb_engine.so
  fi
  if [[ -e $BDIR/CppCore/rgpot/fortran/librgpot_cuh2.so ]]; then
    cuh2so=$BDIR/CppCore/rgpot/fortran/librgpot_cuh2.so
  elif [[ -e $BDIR/CppCore/librgpot_cuh2.so ]]; then
    cuh2so=$BDIR/CppCore/librgpot_cuh2.so
  fi
  if [[ -z $xtbso ]]; then
    echo "libxtb_engine.so missing under $BDIR" >&2
    exit 1
  fi
  if [[ -z $cuh2so ]]; then
    echo "librgpot_cuh2.so missing under $BDIR" >&2
    exit 1
  fi
  nm -D "$cuh2so" | grep -q rgpot_cuh2_force || {
    echo "$cuh2so has no rgpot_cuh2_force" >&2
    nm -D "$cuh2so" | head >&2
    exit 1
  }
  nm -D "$xtbso" | grep -q rgpot_xtb_force || {
    echo "$xtbso has no rgpot_xtb_force" >&2
    exit 1
  }
  mkdir -p "$ROOT/engines"
  ln -sfn "$xtbso" "$ROOT/engines/libxtb_engine.so"
  ln -sfn "$cuh2so" "$ROOT/engines/librgpot_cuh2.so"
  echo "ENGINES_OK xtb=$xtbso cuh2=$cuh2so"
}

build_examples() {
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
  git rev-parse HEAD >SOURCE_COMMIT
  sha256sum \
    target/release/examples/molecular_cluster \
    target/release/examples/slab_adsorption \
    target/release/examples/bank_server \
    target/release/examples/bank_peek \
    engines/libxtb_engine.so \
    engines/librgpot_cuh2.so \
    >MOLSLAB_BUILD_SHA256SUMS
  sha256sum -c MOLSLAB_BUILD_SHA256SUMS
  echo "EXAMPLES_OK $mol $slab $bank $peek"
}

smoke() {
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
