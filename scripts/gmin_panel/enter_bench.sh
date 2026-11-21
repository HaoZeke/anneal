#!/usr/bin/env bash
# Copyright (c) anneal developers
# anneal against GMIN on the paper panel. Builds the anneal revision in a
# detached worktree under /work, a second build of the same revision with
# the LJ103 table entry, and GMIN with the cap and the potential timers,
# then runs both arms in one UTS namespace with hostname bench, one
# process on core 0.
#
#   enter_bench.sh [run_panel.py options]
#   enter_bench.sh --summarize [--tex]   # rewrite the tables from rows.csv only
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REV=${GMIN_PANEL_REV:-c2de652dca0bc75e2b0cb4afffac91fd656404e4}
WORK=${GMIN_PANEL_WORK:-/work/gmin-panel}
OUT=${GMIN_PANEL_OUT:-$ROOT/bench/gmin-panel}
cargo_home=${CARGO_HOME:-$HOME/.cargo}

if [[ ${1:-} == --summarize ]]; then
  shift
  exec python3 "$HERE/run_panel.py" --summarize --root "$ROOT" --rows "$OUT/rows.csv" --revision "$REV" "$@"
fi

tree() {
  local dest=$1
  if [ ! -d "$dest" ]; then
    git -C "$ROOT" worktree add --detach "$dest" "$REV"
  fi
  git -C "$dest" checkout --detach -q "$REV"
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

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/scratch" "$WORK/logs" "$OUT"
tree "$WORK/anneal-src"
if [ -n "$(git -C "$WORK/anneal-src" status --porcelain)" ]; then
  echo "$WORK/anneal-src differs from $REV" >&2
  exit 1
fi
build "$WORK/anneal-src" "$WORK/target" --example h2h_gmin_proof --example h2h_morse --example h2h_metal
for example in h2h_gmin_proof h2h_morse h2h_metal; do
  cp -f "$WORK/target/release/examples/$example" "$WORK/bin/"
done

if "$WORK/bin/h2h_gmin_proof" budget lj103 >/dev/null 2>&1; then
  cp -f "$WORK/bin/h2h_gmin_proof" "$WORK/bin/h2h_gmin_proof_lj103"
else
  tree "$WORK/lj103-src"
  git -C "$WORK/lj103-src" checkout -q -- examples/h2h_gmin_proof.rs
  git -C "$WORK/lj103-src" apply "$ROOT/bench/paper-0120/lj103-table.patch"
  build "$WORK/lj103-src" "$WORK/target-lj103" --example h2h_gmin_proof
  cp -f "$WORK/target-lj103/release/examples/h2h_gmin_proof" "$WORK/bin/h2h_gmin_proof_lj103"
fi

GMIN_PANEL_WORK=$WORK bash "$HERE/build_gmin.sh"
sha256sum "$WORK"/bin/* > "$WORK/bin-sha256.txt"
cp -f "$WORK/gmin-source.txt" "$OUT/gmin-source.txt"
(cd "$WORK/bin" && sha256sum -- *) > "$OUT/bin-sha256.txt"
cp -f "$HERE/run_panel.py" "$ROOT/scripts/target_tol.py" \
  "$ROOT/scripts/gmin_pin.py" "$WORK/"
cp -f "$ROOT/examples/h2h_gmin_lj.data" "$ROOT/examples/h2h_gmin_morse.data" "$WORK/"
# Captured before HOME moves.
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
  --svnrev $WORK/src/wales/GMIN/source/SVNREV \\
  --lj-data $WORK/h2h_gmin_lj.data \\
  --morse-data $WORK/h2h_gmin_morse.data \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $WORK/inside.sh \"\$@\"" _ "$@"
