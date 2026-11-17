# Copyright (c) anneal developers
# Same-session gate of the fresh-start branch against develop. Builds both
# revisions in detached worktrees under /work, a second build of each with
# the LJ103 table entry, then runs the panel in one UTS namespace with
# hostname bench, one process on core 0.
#
#   enter_bench.sh [run_gate.py options]
#
# DEVELOP_REV and BRANCH_REV name the two revisions. BRANCH_REV defaults to
# the checkout this script is in.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVELOP=${DEVELOP_REV:-c2de652dca0bc75e2b0cb4afffac91fd656404e4}
BRANCH=${BRANCH_REV:-$(git -C "$ROOT" rev-parse HEAD)}
WORK=/work/branch-gate
OUT=$ROOT/bench/branch-gate
PATCH=$ROOT/bench/paper-0120/lj103-table.patch
cargo_home=${CARGO_HOME:-$HOME/.cargo}
examples="--example h2h_gmin_proof --example h2h_morse --example h2h_metal"

tree() {
  local dest=$1 rev=$2
  if [ ! -d "$dest" ]; then
    git -C "$ROOT" worktree add --detach "$dest" "$rev"
  fi
  git -C "$dest" checkout -q -- .
  git -C "$dest" checkout --detach -q "$rev"
}

build() {
  local src=$1 target=$2
  shift 2
  mkdir -p "$target" "$WORK/rustc-tmp"
  (
    cd "$src"
    CARGO_TARGET_DIR=$target TMPDIR=$WORK/rustc-tmp \
      RUSTFLAGS="--remap-path-prefix=${src}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo" \
      cargo +1.88.0 build --release --locked "$@" -j 4
  )
}

mkdir -p "$WORK/home" "$WORK/scratch" "$WORK/logs" "$OUT"
for arm in develop branch; do
  if [ "$arm" = develop ]; then rev=$DEVELOP; else rev=$BRANCH; fi
  tree "$WORK/$arm-src" "$rev"
  if [ -n "$(git -C "$WORK/$arm-src" status --porcelain)" ]; then
    echo "$WORK/$arm-src differs from $rev" >&2
    exit 1
  fi
  mkdir -p "$WORK/$arm-bin"
  # shellcheck disable=SC2086
  build "$WORK/$arm-src" "$WORK/$arm-target" $examples
  for example in h2h_gmin_proof h2h_morse h2h_metal; do
    cp -f "$WORK/$arm-target/release/examples/$example" "$WORK/$arm-bin/"
  done
  tree "$WORK/$arm-lj103-src" "$rev"
  git -C "$WORK/$arm-lj103-src" apply "$PATCH"
  build "$WORK/$arm-lj103-src" "$WORK/$arm-lj103-target" --example h2h_gmin_proof
  cp -f "$WORK/$arm-lj103-target/release/examples/h2h_gmin_proof" "$WORK/$arm-bin/h2h_gmin_proof_lj103"
done
sha256sum "$WORK"/develop-bin/* "$WORK"/branch-bin/* > "$WORK/bin-sha256.txt"
cp -f "$HERE/run_gate.py" "$WORK/run_gate.py"
# Captured before HOME moves, so the summary can import scipy.
user_site=$(python3 -c 'import site; print(site.getusersitepackages())')

cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/scratch
export PYTHONPATH=$user_site\${PYTHONPATH:+:\$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd $WORK
exec taskset -c 0 /usr/bin/python3 -u $WORK/run_gate.py \\
  --rows $OUT/rows.csv \\
  --develop $DEVELOP \\
  --branch $BRANCH \\
  --develop-bin $WORK/develop-bin \\
  --branch-bin $WORK/branch-bin \\
  --develop-tree $WORK/develop-src \\
  --branch-tree $WORK/branch-src \\
  --logs $WORK/logs \\
  --compare $OUT/compare.csv \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $WORK/inside.sh \"\$@\"" _ "$@"
