#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build Arm A (develop 21cd57b6) and Arm B (15f38ac2 + anneal-blj-fix5.mbox)
# in detached worktrees, stage lj100-binary inputs, and run the serial gate
# inside a UTS namespace with hostname bench on core 0.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${BLJ_GATE_WORK:-/work/blj-fix5-gate}
ARM_A_REV=${ARM_A_REV:-21cd57b6eb0b4d79ff86a5769632b65f3e6ccd82}
ARM_B_BASE=${ARM_B_BASE:-15f38ac242101fa9c5ffa528566cfd485af0f157}
ARM_B_TREE_PREFIX=${ARM_B_TREE_PREFIX:-208b1b1d}
PATCH_MBOX=${PATCH_MBOX:-}
LJ100_URL="https://raw.githubusercontent.com/lode-org/optbench/main/inputs/sha256/249f808b9820a4f1e24a9118fa38438ebee90ee1f830149495303e230373bde8/lj100-binary-a42b58.tgz"
LJ100_SHA256="249f808b9820a4f1e24a9118fa38438ebee90ee1f830149495303e230373bde8"
PYTHON_BIN=${PYTHON_BIN:-python3}
if [[ -x /work/venv/bin/python ]]; then
  PYTHON_BIN=/work/venv/bin/python
fi

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -un)" /work
fi

mkdir -p "$WORK/bin-a" "$WORK/bin-b" "$WORK/user" "$WORK/scratch" "$WORK/logs" "$WORK/rustc-tmp" "$WORK/stamps"

if [[ ${1:-} == --summarize ]]; then
  shift
  exec "$PYTHON_BIN" "$HERE/summarize.py" --rows "$HERE/rows.csv" --summary "$HERE/summary.csv" --readme "$HERE/README.md" "$@"
fi

ensure_worktree() {
  local dest=$1 rev=$2
  if [[ ! -d $dest ]]; then
    git -C "$ROOT" worktree add --detach "$dest" "$rev"
  fi
}

ensure_worktree "$WORK/arm-a" "$ARM_A_REV"
if [[ $(git -C "$WORK/arm-a" rev-parse HEAD) != "$ARM_A_REV" ]]; then
  git -C "$WORK/arm-a" checkout --detach -q "$ARM_A_REV"
fi
if [[ -n $(git -C "$WORK/arm-a" status --porcelain) ]]; then
  echo "$WORK/arm-a is dirty" >&2
  exit 1
fi

if [[ ! -d $WORK/arm-b ]]; then
  if [[ -z $PATCH_MBOX ]]; then
    echo "set PATCH_MBOX to initialize $WORK/arm-b" >&2
    exit 1
  fi
  git -C "$ROOT" worktree add --detach "$WORK/arm-b" "$ARM_B_BASE"
  GIT_COMMITTER_NAME='Rohit Goswami' GIT_COMMITTER_EMAIL='rgoswami@ieee.org' \
    git -C "$WORK/arm-b" am "$PATCH_MBOX"
fi
if [[ -n $(git -C "$WORK/arm-b" status --porcelain) ]]; then
  echo "$WORK/arm-b is dirty" >&2
  exit 1
fi
arm_b_tree=$(git -C "$WORK/arm-b" rev-parse 'HEAD^{tree}')
if [[ $arm_b_tree != ${ARM_B_TREE_PREFIX}* ]]; then
  echo "Arm B tree $arm_b_tree does not match $ARM_B_TREE_PREFIX*" >&2
  exit 1
fi

build_arm() {
  local arm=$1 src=$2 target=$3 driver_dir=$4 bin_dir=$5 extra_feat=${6:-}
  mkdir -p "$target" "$driver_dir/src"
  (
    cd "$src"
    CARGO_TARGET_DIR="$target" TMPDIR="$WORK/rustc-tmp" \
      RUSTFLAGS="--remap-path-prefix=${src}=/src --remap-path-prefix=${WORK}=/work --remap-path-prefix=${HOME}=/user" \
      cargo +1.88.0 build --release --locked --features rgsaddle --example h2h_seed -j 4
  )
  cp -f "$target/release/examples/h2h_seed" "$bin_dir/h2h_seed"

  cp -f "$HERE/driver/src/main.rs" "$driver_dir/src/main.rs"
  sed "s|path = \"../../..\"|path = \"../arm-${arm}\"|" "$HERE/driver/Cargo.toml" > "$driver_dir/Cargo.toml"
  cp -f "$src/Cargo.lock" "$driver_dir/Cargo.lock"
  (
    cd "$driver_dir"
    local feat_args=()
    if [[ -n $extra_feat ]]; then
      feat_args=(--features "$extra_feat")
    fi
    CARGO_TARGET_DIR="$target" TMPDIR="$WORK/rustc-tmp" \
      RUSTFLAGS="--remap-path-prefix=${driver_dir}=/driver --remap-path-prefix=${src}=/src --remap-path-prefix=${WORK}=/work --remap-path-prefix=${HOME}=/user" \
      cargo +1.88.0 build --release "${feat_args[@]}" -j 4
  )
  cp -f "$target/release/blj-fix5-gate-driver" "$bin_dir/gate_driver"
  chmod +x "$bin_dir/h2h_seed" "$bin_dir/gate_driver"
}

if [[ ${BLJ_GATE_SKIP_BUILD:-0} != 1 ]]; then
  build_arm a "$WORK/arm-a" "$WORK/target-a" "$WORK/driver-a" "$WORK/bin-a" ""
  build_arm b "$WORK/arm-b" "$WORK/target-b" "$WORK/driver-b" "$WORK/bin-b" "hook-binary-lj"
fi

LJ100_TGZ="$WORK/lj100-binary-a42b58.tgz"
LJ100_DIR="$WORK/inputs/lj100-binary-a42b58"
mkdir -p "$WORK/inputs"
if [[ ! -s $LJ100_TGZ ]]; then
  curl -fsSL --retry 3 -o "$LJ100_TGZ" "$LJ100_URL"
fi
echo "${LJ100_SHA256}  ${LJ100_TGZ}" | sha256sum -c -
if [[ ! -f $LJ100_DIR/coords.0 ]]; then
  tar -xzf "$LJ100_TGZ" -C "$WORK/inputs"
fi

touch "$WORK/stamps/build.stamp"
if [[ ${1:-} == --build-only ]]; then
  exit 0
fi

cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/user
export TMPDIR=$WORK/scratch
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd $WORK
exec taskset -c 0 $PYTHON_BIN -u $HERE/run_gate.py \\
  --rows $HERE/rows.csv \\
  --arm-a-tree $WORK/arm-a \\
  --arm-b-tree $WORK/arm-b \\
  --arm-a-bin $WORK/bin-a \\
  --arm-b-bin $WORK/bin-b \\
  --lj100-dir $LJ100_DIR \\
  --logs $WORK/logs \\
  --scratch $WORK/scratch \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $WORK/inside.sh \"\$@\"" _ "$@"
