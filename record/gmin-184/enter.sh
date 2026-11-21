#!/usr/bin/env bash
# Serial record of anneal against GMIN. One core, one thread, release
# builds. Pass the source tarball whose file hash is in tarball.txt.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REV=9837358ae1027067ff654fac8b48969020853452
WORK=${RECORD_WORK:-$ROOT/build/record-184-gmin}
TARBALL=

while [[ $# -gt 0 ]]; do
  case $1 in
    --tarball)
      TARBALL=$2
      shift 2
      ;;
    *)
      echo "unknown argument" >&2
      exit 2
      ;;
  esac
done

if [[ -z $TARBALL || ! -s $TARBALL ]]; then
  echo "pass --tarball" >&2
  exit 2
fi

want=$(awk -F= '$1=="sha256" {print $2}' "$HERE/tarball.txt")
got=$(sha256sum "$TARBALL" | awk '{print $1}')
if [[ $got != "$want" ]]; then
  echo "tarball file hash does not match the recorded file" >&2
  exit 1
fi

mkdir -p "$WORK/bin" "$WORK/logs" "$WORK/runs" "$WORK/rustc-tmp"
if [[ ! -f $WORK/wales/GMIN/source/main.F90 ]]; then
  tar -xjf "$TARBALL" -C "$WORK" \
    ./wales/GMIN \
    ./wales/CMakeModules \
    ./wales/INTERFACES \
    ./wales/SCRIPTS/all \
    ./wales/HiRE
fi
python3 "$HERE/apply_patches.py" --root "$ROOT" --dest "$WORK/wales/GMIN"

if [[ ! -x $WORK/bin/GMIN ]]; then
  build=$WORK/wales/GMIN/build/gfortran
  mkdir -p "$build"
  (
    cd "$build"
    FC=gfortran CC=gcc CXX=g++ cmake ../../source \
      -DWITH_ALIGN=OFF \
      -DWITH_MPI=OFF \
      -DWITH_OPENMP=OFF \
      -DWITH_SPARSE=OFF \
      -DWITH_PYTHON=OFF \
      -DWITH_MYBLAS=OFF \
      -DWITH_MYLAPACK=OFF
    make -j"$(nproc)" GMIN
  )
  cp -a "$build/GMIN" "$WORK/bin/GMIN"
  chmod +x "$WORK/bin/GMIN"
fi

src=$WORK/anneal-src
if [[ ! -d $src/.git ]]; then
  git -C "$ROOT" worktree add --detach "$src" "$REV"
fi
git -C "$src" checkout --detach -q "$REV"
if [[ -n $(git -C "$src" status --porcelain) ]]; then
  echo "anneal source tree is dirty" >&2
  exit 1
fi
if [[ $(git -C "$src" rev-parse HEAD) != "$REV" ]]; then
  echo "anneal source is not the pinned commit" >&2
  exit 1
fi

export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${src}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo"
if [[ ! -x $CARGO_TARGET_DIR/release/examples/h2h_metal || ! -x $CARGO_TARGET_DIR/release/examples/h2h_morse ]]; then
  cargo +1.88.0 build --release --locked \
    --manifest-path "$src/Cargo.toml" \
    --example h2h_metal --example h2h_morse -j 4
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
exec taskset -c 0 python3 -u "$HERE/run_record.py" \
  --root "$ROOT" \
  --rows "$HERE/rows.csv" \
  --inputs "$HERE/inputs" \
  --identity "$HERE/tarball.txt" \
  --tarball "$TARBALL" \
  --tree "$src" \
  --anneal-bin "$CARGO_TARGET_DIR/release/examples" \
  --gmin "$WORK/bin/GMIN" \
  --work "$WORK/runs" \
  --logs "$WORK/logs"
