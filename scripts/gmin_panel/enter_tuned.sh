#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build the pinned anneal examples and the GMIN image, then run the tuned
# panel on core 0. The work directory comes from GMIN_TUNED_WORK.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${GMIN_TUNED_WORK:?set GMIN_TUNED_WORK}
BASE=5eb79cdcb0a986984dc7b487e5a3cc93224246f9
EXPECT=9cbec1a446352bc9ef064bd2b1a5767194655919
SHA=ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776

export GMIN_PANEL_WORK="$WORK"
export GMIN_EXPECT_REV="$EXPECT"
export GMIN_TARBALL_SHA256="$SHA"
export GMIN_PANEL_EXTRA_PATCHES="$ROOT/scripts/tip4p_h2h/newtip-timer.patch"
export LJOS_SEAT="${LJOS_SEAT:-benchbot}"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/scratch" "$WORK/logs" "$WORK/runs" "$ROOT/bench/gmin-tuned"
bash "$HERE/build_gmin.sh"

if [[ ! -d $WORK/anneal-src ]]; then
  git -C "$ROOT" worktree add --detach "$WORK/anneal-src" "$BASE"
fi
git -C "$WORK/anneal-src" checkout --detach -q "$BASE"
if [[ -n "$(git -C "$WORK/anneal-src" status --porcelain)" ]]; then
  echo "anneal source differs from $BASE" >&2
  exit 1
fi

if [[ ! -x $WORK/bin/h2h_gmin_proof || ! -x $WORK/bin/h2h_seed ]]; then
  mkdir -p "$WORK/rustc-tmp"
  cargo_home=${CARGO_HOME:-$HOME/.cargo}
  (
    cd "$WORK/anneal-src"
    CARGO_TARGET_DIR="$WORK/target" TMPDIR="$WORK/rustc-tmp" \
      RUSTFLAGS="--remap-path-prefix=${WORK}/anneal-src=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo" \
      cargo +1.88.0 build --release --locked \
        --example h2h_gmin_proof --example h2h_morse --example h2h_metal --example h2h_seed -j 4
  )
  for example in h2h_gmin_proof h2h_morse h2h_metal h2h_seed; do
    cp -a "$WORK/target/release/examples/$example" "$WORK/bin/"
  done
fi

{
  echo "gmin_svnrev=$EXPECT"
  echo "tarball_sha256=$SHA"
  echo "panel_svnrev=aac1548a8aec13b53ad12a8d6904625081685d2e"
  echo "cmake=-DWITH_ALIGN=OFF -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_SPARSE=OFF -DWITH_PYTHON=OFF -DWITH_MYBLAS=OFF -DWITH_MYLAPACK=OFF"
  echo "gfortran=$(gfortran -dumpfullversion)"
  echo "rustc=$(rustup run 1.88.0 rustc --version)"
  echo "anneal_commit=$BASE"
  echo "anneal_tree=$(git -C "$ROOT" rev-parse "${BASE}^{tree}")"
  echo "gmin_binary_sha256=$(sha256sum "$WORK/bin/GMIN" | cut -d' ' -f1)"
  for example in h2h_gmin_proof h2h_morse h2h_metal h2h_seed; do
    echo "example_${example}=$(sha256sum "$WORK/bin/$example" | cut -d' ' -f1)"
  done
} > "$WORK/build-record.txt"
cp -f "$WORK/build-record.txt" "$ROOT/bench/gmin-tuned/build.txt"

user_name=$(id -un)
cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/scratch
export BIN=$WORK/bin
export ANNEAL=$WORK/anneal-src
export ROOT=$ROOT
export WORK=$WORK/runs/current
export LJOS_SEAT=${LJOS_SEAT:-benchbot}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export OMP_THREAD_LIMIT=1 GOTO_NUM_THREADS=1 MKL_DOMAIN_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OPENBLAS_DYNAMIC=FALSE
mkdir -p "\$WORK"
cd "\$ROOT"
exec taskset -c 0 /usr/bin/python3 -u $HERE/tuned.py \\
  --root "\$ROOT" \\
  --bin "\$BIN" \\
  --anneal "\$ANNEAL" \\
  --work "\$WORK" \\
  --logs $WORK/logs \\
  --build $WORK/build-record.txt \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u ${user_name} $WORK/inside.sh \"\$@\"" _ "$@"
