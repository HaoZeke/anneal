#!/usr/bin/env bash
# Copyright (c) anneal developers
# Builds develop and this checkout, then runs one ablation campaign in a
# UTS namespace with hostname bench, under /work.
#
#   enter_bench.sh <develop revision> <campaign> [run_ablation.py options]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/generic-alloy
OUT=$ROOT/bench/generic-alloy
develop_rev=$(git -C "$ROOT" rev-parse "${1:?develop revision}")
campaign=${2:?campaign}
shift 2
# The branch revision is the last commit to touch what the build reads.
built="src examples Cargo.toml Cargo.lock"
branch_rev=$(git -C "$ROOT" log -1 --format=%H -- $built)
if [ -n "$(git -C "$ROOT" status --porcelain -- $built)" ]; then
  echo "src or examples differ from HEAD; commit before timing" >&2
  exit 1
fi
cargo_home=${CARGO_HOME:-$HOME/.cargo}

build() {
  local tree=$1 name=$2
  mkdir -p "$WORK/target-$name" "$WORK/rustc-tmp" "$WORK/$name"
  (
    cd "$tree"
    CARGO_TARGET_DIR=$WORK/target-$name TMPDIR=$WORK/rustc-tmp \
      RUSTFLAGS="--remap-path-prefix=${tree}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo" \
      cargo +1.88.0 build --release --locked --example h2h_metal --example h2h_seed -j 4
  )
  cp -f "$WORK/target-$name/release/examples/h2h_metal" "$WORK/target-$name/release/examples/h2h_seed" "$WORK/$name/"
}

if [ ! -d "$WORK/develop-tree" ]; then
  git -C "$ROOT" worktree add --detach "$WORK/develop-tree" "$develop_rev"
fi
git -C "$WORK/develop-tree" checkout --detach -q "$develop_rev"
build "$WORK/develop-tree" develop
build "$ROOT" branch
scratch=$ROOT/build/generic-alloy
# Captured before HOME moves, so the campaign can import what the
# operator already installed for this user.
user_site=$(python3 -c 'import site; print(site.getusersitepackages())')
mkdir -p "$WORK/bin" "$scratch/user" "$scratch/scratch" "$WORK/logs" "$OUT"
cp -f "$HERE/run_ablation.py" "$WORK/bin/"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$scratch/user
export TMPDIR=$scratch/scratch
export PYTHONPATH=$user_site\${PYTHONPATH:+:\$PYTHONPATH}
cd $WORK
exec /usr/bin/python3 -u $WORK/bin/run_ablation.py \\
  --campaign $campaign \\
  --develop $WORK/develop --develop-revision $develop_rev \\
  --branch $WORK/branch --branch-revision $branch_rev \\
  --rows $OUT/$campaign-rows.csv \\
  --summary $OUT/$campaign-summary-${branch_rev:0:8}.csv \\
  --logs $WORK/logs \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $WORK/bin/inside.sh \"\$@\"" _ "$@"
