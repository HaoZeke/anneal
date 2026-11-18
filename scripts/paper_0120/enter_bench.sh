# Copyright (c) anneal developers
# The paper headline panel on the release build. Builds the release
# revision in a detached worktree under /work, a second build of the same
# revision with the LJ103 table entry, then runs the panel in one UTS
# namespace with hostname bench, one process on core 0.
#
#   enter_bench.sh [run_panel.py options]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REV=${PAPER_REV:-07f89cadbd9ccd8386e92408a8ff0e8153b180e3}
WORK=/work/paper-0120
OUT=$ROOT/bench/paper-0120
cargo_home=${CARGO_HOME:-$HOME/.cargo}
examples="--example h2h_gmin_proof --example h2h_morse --example h2h_metal --example h2h_seed"

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
tree "$WORK/release-src"
if [ -n "$(git -C "$WORK/release-src" status --porcelain)" ]; then
  echo "$WORK/release-src differs from $REV" >&2
  exit 1
fi
# shellcheck disable=SC2086
build "$WORK/release-src" "$WORK/target" --lib $examples
for example in h2h_gmin_proof h2h_morse h2h_metal h2h_seed; do
  cp -f "$WORK/target/release/examples/$example" "$WORK/bin/"
done

tree "$WORK/lj103-src"
git -C "$WORK/lj103-src" checkout -q -- examples/h2h_gmin_proof.rs
git -C "$WORK/lj103-src" apply "$OUT/lj103-table.patch"
build "$WORK/lj103-src" "$WORK/target-lj103" --example h2h_gmin_proof
cp -f "$WORK/target-lj103/release/examples/h2h_gmin_proof" "$WORK/bin/h2h_gmin_proof_lj103"
sha256sum "$WORK"/bin/* > "$WORK/bin-sha256.txt"
cp -f "$HERE/run_panel.py" "$ROOT/scripts/target_tol.py" "$WORK/"
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
exec taskset -c 0 /usr/bin/python3 -u $WORK/run_panel.py \\
  --root $ROOT \\
  --rows $OUT/rows.csv \\
  --revision $REV \\
  --bin $WORK/bin \\
  --tree $WORK/release-src \\
  --logs $WORK/logs \\
  --summary $OUT/summary.csv \\
  --compare $OUT/compare.csv \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $WORK/inside.sh \"\$@\"" _ "$@"
