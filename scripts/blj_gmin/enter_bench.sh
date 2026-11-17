#!/usr/bin/env bash
# Copyright (c) anneal developers
# anneal against GMIN on the Doye-Meyer binary Lennard-Jones clusters.
# Builds the anneal revision in a detached worktree and GMIN with the
# evaluator cap and a timer inside BLJ_CLUST, then runs both arms in one
# UTS namespace with hostname bench, one process on core 0.
#
#   enter_bench.sh [run_panel.py options]
#   enter_bench.sh --summarize [--tex]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REV=${BLJ_REV:-$(git -C "$ROOT" rev-parse HEAD)}
WORK=${BLJ_WORK:-/work/blj-gmin}
OUT=$ROOT/bench/blj-gmin
cargo_home=${CARGO_HOME:-$HOME/.cargo}

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -un)" /work
fi

if [[ ${1:-} == --summarize ]]; then
  shift
  if [[ -z ${BLJ_REV:-} && -f $WORK/revision.txt ]]; then
    REV=$(tr -d '[:space:]' < "$WORK/revision.txt")
  fi
  exec python3 "$HERE/run_panel.py" --summarize --root "$ROOT" --rows "$OUT/rows.csv" --revision "$REV" "$@"
fi

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/scratch" "$WORK/logs" "$OUT"
echo "$REV" > "$WORK/revision.txt"

tree() {
  local dest=$1
  if [ ! -d "$dest" ]; then
    git -C "$ROOT" worktree add --detach "$dest" "$REV"
  fi
  git -C "$dest" checkout --detach -q "$REV"
}

tree "$WORK/anneal-src"
if [ -n "$(git -C "$WORK/anneal-src" status --porcelain)" ]; then
  echo "$WORK/anneal-src differs from $REV" >&2
  exit 1
fi
mkdir -p "$WORK/target" "$WORK/rustc-tmp"
(
  cd "$WORK/anneal-src"
  CARGO_TARGET_DIR=$WORK/target TMPDIR=$WORK/rustc-tmp \
    RUSTFLAGS="--remap-path-prefix=${WORK}/anneal-src=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo" \
    cargo +1.88.0 build --release --locked --example h2h_blj -j 4
)
cp -f "$WORK/target/release/examples/h2h_blj" "$WORK/bin/"

BLJ_WORK=$WORK bash "$HERE/build_gmin.sh"
sha256sum "$WORK"/bin/* > "$WORK/bin-sha256.txt"
cp -f "$WORK/gmin-source.txt" "$OUT/gmin-source.txt"
(cd "$WORK/bin" && sha256sum -- *) > "$OUT/bin-sha256.txt"
cp -f "$HERE/run_panel.py" "$WORK/run_panel.py"
cp -f "$ROOT/examples/h2h_gmin_lj.data" "$WORK/h2h_gmin_lj.data"
user_site=$(python3 -c 'import site; print(site.getusersitepackages())')

cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/scratch
export PYTHONPATH=$user_site\${PYTHONPATH:+:\$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd $WORK
exec taskset -c 0 /usr/bin/python3 -u $WORK/run_panel.py \\
  --root $ROOT \\
  --rows $OUT/rows.csv \\
  --revision $REV \\
  --bin $WORK/bin \\
  --tree $WORK/anneal-src \\
  --work $WORK/runs \\
  --logs $WORK/logs \\
  --gmin-source $WORK/gmin-source.txt \\
  --lj-data $WORK/h2h_gmin_lj.data \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $WORK/inside.sh \"\$@\"" _ "$@"
