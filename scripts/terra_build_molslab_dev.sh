#!/usr/bin/env bash
# Development build of the molecular and slab examples on Terra.
#
# Deviation from scripts/terra_build_rgpot_ex.sh: no EXPECTED_COMMIT,
# no clean-tree gate, no sealed SHA256SUMS. The sealed recipe refuses a
# dirty development worktree. Engines are built with the same meson
# flags when they are missing; cargo then builds the three examples
# with rgpot-ex,ira,featomic,bank-rpc (ira plus the example
# required-features the sealed recipe already uses).
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_build_molslab_dev.sh requires a Slurm allocation" >&2
  exit 1
fi

ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
RGPOT=${RGPOT_ROOT:-$HOME/Git/Github/TheochemUI/gpr_optim/subprojects/rgpot}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTB_CACHE_LIB=${XTB_CACHE_LIB:-$HOME/.cache/rattler/cache/pkgs/xtb-6.7.1-h8876d29_4/lib}
RGPOT_BDIR=${RGPOT_BDIR:-$HOME/.cache/anneal-audit/rgpot-molslab-dev-${SLURM_JOB_ID}}

dynamic_symbol_is_defined() {
  local library=$1 symbol=$2
  nm -D --defined-only "$library" |
    awk -v symbol="$symbol" '$NF == symbol { found = 1 } END { exit found ? 0 : 1 }'
}

if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing IRA library: $IRA_LIB_DIR/libira.so" >&2
  exit 1
fi

mkdir -p "$ROOT/engines"
xtbso=$ROOT/engines/libxtb_engine.so
cuh2so=$ROOT/engines/librgpot_cuh2.so

if [[ ! -e $xtbso || ! -e $cuh2so ]]; then
  if [[ ! -x $PIXI ]]; then
    echo "missing Pixi executable: $PIXI" >&2
    exit 1
  fi
  if [[ ! -s $RGPOT/pixi.lock ]]; then
    echo "missing rgpot Pixi lockfile: $RGPOT/pixi.lock" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$RGPOT_BDIR")"
  cd "$RGPOT"
  "$PIXI" install --locked -e xtbbld
  if [[ ! -f $RGPOT_BDIR/build.ninja ]]; then
    "$PIXI" run --locked -e xtbbld meson setup "$RGPOT_BDIR" "$RGPOT" \
      -Dwith_xtb=true \
      -Dwith_fortran_pots=enabled \
      -Dwith_tests=false \
      -Dwith_rpc=false \
      -Dwith_cache=false \
      --buildtype=release
  fi
  "$PIXI" run --locked -e xtbbld meson compile -C "$RGPOT_BDIR"
  built_cuh2=$RGPOT_BDIR/CppCore/rgpot/fortran/librgpot_cuh2.so
  "$PIXI" run --locked -e xtbbld \
    "$ROOT/scripts/link_rgpot_cuh2_engine.sh" \
    "$RGPOT" \
    "$RGPOT_BDIR" \
    "$ROOT/scripts/rgpot_cuh2.exports" \
    "$built_cuh2"
  built_xtb=
  for candidate in \
    "$RGPOT_BDIR/CppCore/rgpot/XTBPot/libxtb_engine.so" \
    "$RGPOT_BDIR/CppCore/libxtb_engine.so"; do
    if [[ -e $candidate ]]; then
      built_xtb=$candidate
      break
    fi
  done
  if [[ -z $built_xtb || ! -e $built_cuh2 ]]; then
    echo "rgpot engine build did not produce both required libraries" >&2
    exit 1
  fi
  cp "$built_xtb" "$xtbso"
  cp "$built_cuh2" "$cuh2so"
fi

dynamic_symbol_is_defined "$xtbso" rgpot_xtb_force || {
  echo "$xtbso has no rgpot_xtb_force" >&2
  exit 1
}
dynamic_symbol_is_defined "$cuh2so" rgpot_cuh2_force || {
  echo "$cuh2so has no rgpot_cuh2_force" >&2
  exit 1
}

export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8}
export IRA_LIB_DIR
xtb_lib=
if [[ -d $RGPOT/.pixi/envs/xtbbld/lib ]]; then
  xtb_lib=$RGPOT/.pixi/envs/xtbbld/lib
elif [[ -d $XTB_CACHE_LIB ]]; then
  xtb_lib=$XTB_CACHE_LIB
fi
export LD_LIBRARY_PATH="${xtb_lib:+$xtb_lib:}$IRA_LIB_DIR:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
# Deviation from the packet's `rgpot-ex,ira --examples`: the three
# named examples declare featomic and bank-rpc in Cargo.toml.
cargo build --release --features rgpot-ex,ira,featomic,bank-rpc \
  --example water_ride_search \
  --example molecular_cluster \
  --example slab_adsorption

for executable in \
  target/release/examples/water_ride_search \
  target/release/examples/molecular_cluster \
  target/release/examples/slab_adsorption; do
  if [[ ! -x $executable ]]; then
    echo "missing molecular campaign executable: $executable" >&2
    exit 1
  fi
  if ldd "$executable" | grep -F "not found" >/dev/null; then
    echo "unresolved libraries in $executable" >&2
    ldd "$executable" >&2
    exit 1
  fi
done
for engine in engines/libxtb_engine.so engines/librgpot_cuh2.so; do
  if ldd "$engine" | grep -F "not found" >/dev/null; then
    echo "unresolved libraries in $engine" >&2
    ldd "$engine" >&2
    exit 1
  fi
done

printf 'TERRA_MOLSLAB_DEV_BUILD_OK source=%s host=%s job=%s\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$(hostname)" "$SLURM_JOB_ID"
