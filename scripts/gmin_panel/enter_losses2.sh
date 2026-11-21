#!/usr/bin/env bash
# Copyright (c) anneal developers
# Pair-scaled Morse against the parent preset and GMIN. Builds the parent
# revision and HEAD under $OUTPUT (or /work/gmin-losses2), then runs both
# anneal arms and GMIN in one UTS namespace with hostname bench, one
# process on core 0. Rows stay in that directory. Nothing is written into
# the repository.
#
#   enter_losses2.sh [--system NAME] [--seed N]
#   enter_losses2.sh --summarize
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REV=$(git -C "$ROOT" rev-parse HEAD)
BASE_REV=${GMIN_LOSSES2_BASE:-5eb79cdcb0a986984dc7b487e5a3cc93224246f9}
OUT=${OUTPUT:-/work/gmin-losses2}
cargo_home=${CARGO_HOME:-$HOME/.cargo}

case "$(realpath -m "$OUT")" in
  "$(realpath -m "$ROOT")"|"$(realpath -m "$ROOT")"/*)
    echo "$OUT is inside the repository" >&2
    exit 1
    ;;
esac

if [[ ${1:-} == --summarize ]]; then
  shift
  exec python3 "$HERE/run_losses2.py" --summarize --revision "$REV" --base-revision "$BASE_REV" "$@"
fi

tree() {
  local dest=$1 revision=$2
  if [ ! -d "$dest" ]; then
    git -C "$ROOT" worktree add --detach "$dest" "$revision"
  fi
  git -C "$dest" checkout --detach -q "$revision"
  if [ -n "$(git -C "$dest" status --porcelain)" ]; then
    echo "$dest differs from $revision" >&2
    exit 1
  fi
}

build() {
  local src=$1 target=$2
  shift 2
  mkdir -p "$target" "$OUT/rustc-tmp"
  (
    cd "$src"
    CARGO_TARGET_DIR=$target TMPDIR=$OUT/rustc-tmp \
      RUSTFLAGS="--remap-path-prefix=${src}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo" \
      cargo +1.88.0 build --release --locked "$@" -j 4
  )
}

mkdir -p "$OUT/base-bin" "$OUT/head-bin" "$OUT/home" "$OUT/scratch" "$OUT/logs" "$OUT/runs"
tree "$OUT/base-src" "$BASE_REV"
tree "$OUT/head-src" "$REV"
build "$OUT/base-src" "$OUT/base-target" --example h2h_gmin_proof --example h2h_morse --example h2h_metal
build "$OUT/head-src" "$OUT/head-target" --example h2h_gmin_proof --example h2h_morse --example h2h_metal
for example in h2h_gmin_proof h2h_morse h2h_metal; do
  cp -f "$OUT/base-target/release/examples/$example" "$OUT/base-bin/"
  cp -f "$OUT/head-target/release/examples/$example" "$OUT/head-bin/"
done

GMIN_PANEL_WORK=$OUT bash "$HERE/build_gmin.sh"
cp -f "$OUT/bin/GMIN" "$OUT/base-bin/GMIN"
cp -f "$HERE/run_losses2.py" "$HERE/run_panel.py" \
  "$ROOT/scripts/target_tol.py" "$ROOT/scripts/gmin_pin.py" "$OUT/"
cp -f "$ROOT/examples/h2h_gmin_lj.data" "$ROOT/examples/h2h_gmin_morse.data" "$OUT/"
user_site=$(python3 -c 'import site; print(site.getusersitepackages())')

cat > "$OUT/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$OUT/home
export TMPDIR=$OUT/scratch
export OUTPUT=$OUT
export PYTHONPATH=$user_site\${PYTHONPATH:+:\$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd $OUT
exec taskset -c 0 /usr/bin/python3 -u $OUT/run_losses2.py \\
  --revision $REV \\
  --base-revision $BASE_REV \\
  --base-bin $OUT/base-bin \\
  --base-tree $OUT/base-src \\
  --head-bin $OUT/head-bin \\
  --head-tree $OUT/head-src \\
  --work $OUT/runs \\
  --logs $OUT/logs \\
  --gmin-source $OUT/gmin-source.txt \\
  --svnrev $OUT/gmin-src/wales/GMIN/source/SVNREV \\
  --lj-data $OUT/h2h_gmin_lj.data \\
  --morse-data $OUT/h2h_gmin_morse.data \\
  --rows $OUT/rows.csv \\
  --summary $OUT/summary.csv \\
  "\$@"
EOF
chmod +x "$OUT/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $OUT/inside.sh \"\$@\"" _ "$@"
