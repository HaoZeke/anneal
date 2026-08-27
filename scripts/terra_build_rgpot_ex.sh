#!/usr/bin/env bash
# Build and seal the molecular and CuH2 campaign artifacts on Terra.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_build_rgpot_ex.sh requires a Slurm allocation" >&2
  exit 1
fi

ROOT=${LJ_ROOT:-$HOME/anneal-molslab}
RGPOT=${RGPOT_ROOT:-$HOME/Git/Github/TheochemUI/gpr_optim/subprojects/rgpot}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
RGPOT_EXPECTED_COMMIT=${RGPOT_EXPECTED_COMMIT:?set RGPOT_EXPECTED_COMMIT}
RGPOT_BDIR=${RGPOT_BDIR:-$HOME/.cache/anneal-audit/rgpot-${RGPOT_EXPECTED_COMMIT:0:12}-${SLURM_JOB_ID}}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}

if [[ ! -x $PIXI ]]; then
  echo "missing Pixi executable: $PIXI" >&2
  exit 1
fi
if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing IRA library: $IRA_LIB_DIR/libira.so" >&2
  exit 1
fi

MAIN_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
if [[ $MAIN_COMMIT != "$EXPECTED_COMMIT" ]]; then
  echo "EXPECTED_COMMIT=$EXPECTED_COMMIT does not match anneal HEAD=$MAIN_COMMIT" >&2
  exit 2
fi
if ! (cd "$ROOT" && git diff --quiet HEAD --); then
  echo "anneal tracked source differs from HEAD=$MAIN_COMMIT" >&2
  (cd "$ROOT" && git status --short >&2)
  exit 2
fi

RGPOT_COMMIT=$(git -C "$RGPOT" rev-parse HEAD)
if [[ $RGPOT_COMMIT != "$RGPOT_EXPECTED_COMMIT" ]]; then
  echo "RGPOT_EXPECTED_COMMIT=$RGPOT_EXPECTED_COMMIT does not match rgpot HEAD=$RGPOT_COMMIT" >&2
  exit 2
fi
if ! git -C "$RGPOT" diff --quiet HEAD --; then
  echo "rgpot tracked source differs from HEAD=$RGPOT_COMMIT" >&2
  git -C "$RGPOT" status --short >&2
  exit 2
fi
if [[ ! -s $RGPOT/pixi.lock ]]; then
  echo "missing tracked rgpot Pixi lockfile: $RGPOT/pixi.lock" >&2
  exit 2
fi
if [[ -e $RGPOT_BDIR ]]; then
  echo "refusing to reuse rgpot build directory: $RGPOT_BDIR" >&2
  exit 2
fi

mkdir -p "$(dirname "$RGPOT_BDIR")"
cd "$RGPOT"
"$PIXI" install --locked -e xtbbld
"$PIXI" run --locked -e xtbbld meson setup "$RGPOT_BDIR" "$RGPOT" \
  -Dwith_xtb=true \
  -Dwith_fortran_pots=enabled \
  -Dwith_tests=false \
  -Dwith_rpc=false \
  -Dwith_cache=false \
  --buildtype=release
"$PIXI" run --locked -e xtbbld meson compile -C "$RGPOT_BDIR"

xtbso=
cuh2so=
for candidate in \
  "$RGPOT_BDIR/CppCore/rgpot/XTBPot/libxtb_engine.so" \
  "$RGPOT_BDIR/CppCore/libxtb_engine.so"; do
  if [[ -e $candidate ]]; then
    xtbso=$candidate
    break
  fi
done
for candidate in \
  "$RGPOT_BDIR/CppCore/rgpot/fortran/librgpot_cuh2.so" \
  "$RGPOT_BDIR/CppCore/librgpot_cuh2.so"; do
  if [[ -e $candidate ]]; then
    cuh2so=$candidate
    break
  fi
done
if [[ -z $xtbso || -z $cuh2so ]]; then
  echo "rgpot engine build did not produce both required libraries" >&2
  exit 1
fi
nm -D "$xtbso" | grep -q rgpot_xtb_force || {
  echo "$xtbso has no rgpot_xtb_force" >&2
  exit 1
}
nm -D "$cuh2so" | grep -q rgpot_cuh2_force || {
  echo "$cuh2so has no rgpot_cuh2_force" >&2
  exit 1
}

mkdir -p "$ROOT/engines"
cp "$xtbso" "$ROOT/engines/libxtb_engine.so"
cp "$cuh2so" "$ROOT/engines/librgpot_cuh2.so"

export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8}
export IRA_LIB_DIR
export LD_LIBRARY_PATH="$RGPOT/.pixi/envs/xtbbld/lib:$IRA_LIB_DIR:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
cargo build --locked --release --features rgpot-ex,featomic,bank-rpc \
  --example molecular_cluster \
  --example slab_adsorption \
  --example bank_server \
  --example bank_peek

for executable in \
  target/release/examples/molecular_cluster \
  target/release/examples/slab_adsorption \
  target/release/examples/bank_server \
  target/release/examples/bank_peek; do
  if [[ ! -x $executable ]]; then
    echo "missing molecular campaign executable: $executable" >&2
    exit 1
  fi
  if ldd "$executable" | grep -q "not found"; then
    echo "unresolved libraries in $executable" >&2
    ldd "$executable" >&2
    exit 1
  fi
done
for engine in engines/libxtb_engine.so engines/librgpot_cuh2.so; do
  if ldd "$engine" | grep -q "not found"; then
    echo "unresolved libraries in $engine" >&2
    ldd "$engine" >&2
    exit 1
  fi
done

git rev-parse HEAD >SOURCE_COMMIT
git -C "$RGPOT" rev-parse HEAD >RGPOT_SOURCE_COMMIT
sha256sum "$RGPOT/pixi.lock" | awk '{print $1}' >RGPOT_PIXI_LOCK_SHA256
{
  printf 'source_commit=%s\n' "$MAIN_COMMIT"
  printf 'rgpot_source_commit=%s\n' "$RGPOT_COMMIT"
  printf 'rgpot_pixi_lock_sha256=%s\n' "$(cat RGPOT_PIXI_LOCK_SHA256)"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  printf 'pixi=%s\n' "$("$PIXI" --version)"
} >MOLSLAB_BUILD_PROVENANCE
sha256sum \
  target/release/examples/molecular_cluster \
  target/release/examples/slab_adsorption \
  target/release/examples/bank_server \
  target/release/examples/bank_peek \
  engines/libxtb_engine.so \
  engines/librgpot_cuh2.so \
  SOURCE_COMMIT \
  RGPOT_SOURCE_COMMIT \
  RGPOT_PIXI_LOCK_SHA256 \
  MOLSLAB_BUILD_PROVENANCE \
  >MOLSLAB_BUILD_SHA256SUMS
sha256sum -c MOLSLAB_BUILD_SHA256SUMS

if ! (cd "$ROOT" && git diff --quiet HEAD --); then
  echo "anneal tracked source changed during the build" >&2
  exit 2
fi
if ! git -C "$RGPOT" diff --quiet HEAD --; then
  echo "rgpot tracked source changed during the build" >&2
  exit 2
fi

export RGPOT_XTB_ENGINE=$ROOT/engines/libxtb_engine.so
export RGPOT_CUH2_LIBRARY=$ROOT/engines/librgpot_cuh2.so
export RGPOT_BUILD_IDENTITY=$(sha256sum "$RGPOT_XTB_ENGINE" | awk '{print $1}')
"$ROOT/target/release/examples/molecular_cluster" 2 20 1
"$ROOT/target/release/examples/slab_adsorption" \
  "$ROOT/examples/fixtures/cuh2_fcc_slab.con" 15 1

printf 'TERRA_MOLSLAB_BUILD_OK source=%s rgpot=%s build=%s\n' \
  "$MAIN_COMMIT" "$RGPOT_COMMIT" "$RGPOT_BDIR"
