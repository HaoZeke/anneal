#!/usr/bin/env bash
# Copyright (c) anneal developers
# anneal against GMIN on the paper panel. In fix mode, builds the reviewed
# base and candidate revision separately, applying only the shared structural-
# template ablation to the base. Runs both arms in one UTS namespace with
# hostname bench, one process on core 0.
#
#   enter_bench.sh [run_panel.py options]
#   enter_bench.sh --summarize [--tex]   # rewrite the tables from rows.csv only
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REV=${GMIN_PANEL_REV:-c2de652dca0bc75e2b0cb4afffac91fd656404e4}
BASE_REV=${GMIN_PANEL_BASE_REV:-628773d6837ea2d959a27256ff4163b21162fd4d}
WORK=${GMIN_PANEL_WORK:-/work/gmin-panel}
MODE=${GMIN_PANEL_MODE:-panel}
if [[ $MODE == fix ]]; then
  OUT=${GMIN_PANEL_OUT:-$ROOT/bench/gmin-losses}
  RUNNER=run_fix_gate.py
  SUMMARY=(--summary "$OUT/summary.csv")
  BASE_SUMMARY=(--base-revision "$BASE_REV")
  BASE_RUN=(--base-revision "$BASE_REV" --base-bin "$WORK/base-bin" --base-tree "$WORK/base-src")
else
  OUT=${GMIN_PANEL_OUT:-$ROOT/bench/gmin-panel}
  RUNNER=run_panel.py
  SUMMARY=()
  BASE_SUMMARY=()
  BASE_RUN=()
fi
cargo_home=${CARGO_HOME:-$HOME/.cargo}

if [[ ${1:-} == --summarize ]]; then
  shift
  exec python3 "$HERE/$RUNNER" --summarize --root "$ROOT" --rows "$OUT/rows.csv" \
    --revision "$REV" "${BASE_SUMMARY[@]}" "${SUMMARY[@]}" "$@"
fi

tree() {
  local dest=$1 revision=$2
  if [ ! -d "$dest" ]; then
    git -C "$ROOT" worktree add --detach "$dest" "$revision"
  fi
  git -C "$dest" checkout --detach -q "$revision"
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

apply_once() {
  local src=$1 patch=$2
  if git -C "$src" apply --check "$patch"; then
    git -C "$src" apply "$patch"
  elif ! git -C "$src" apply --reverse --check "$patch"; then
    echo "$src neither accepts nor already contains $patch" >&2
    exit 1
  fi
}

build_bins() {
  local label=$1 src=$2 revision=$3 target=$4 bin_dir=$5
  mkdir -p "$bin_dir"
  build "$src" "$target" --example h2h_gmin_proof --example h2h_morse --example h2h_metal
  for example in h2h_gmin_proof h2h_morse h2h_metal; do
    cp -f "$target/release/examples/$example" "$bin_dir/"
  done
  if "$bin_dir/h2h_gmin_proof" budget lj103 >/dev/null 2>&1; then
    cp -f "$bin_dir/h2h_gmin_proof" "$bin_dir/h2h_gmin_proof_lj103"
    return
  fi
  local lj_src="$WORK/$label-lj103-src"
  tree "$lj_src" "$revision"
  if [[ $label == base ]]; then
    apply_once "$lj_src" "$ROOT/bench/gmin-losses/no-template-base.patch"
  fi
  apply_once "$lj_src" "$ROOT/bench/paper-0120/lj103-table.patch"
  build "$lj_src" "$target-lj103" --example h2h_gmin_proof
  cp -f "$target-lj103/release/examples/h2h_gmin_proof" "$bin_dir/h2h_gmin_proof_lj103"
}

mkdir -p "$WORK/bin" "$WORK/base-bin" "$WORK/home" "$WORK/scratch" "$WORK/logs" "$OUT"
tree "$WORK/anneal-src" "$REV"
if [ -n "$(git -C "$WORK/anneal-src" status --porcelain)" ]; then
  echo "$WORK/anneal-src differs from candidate $REV" >&2
  exit 1
fi
build_bins candidate "$WORK/anneal-src" "$REV" "$WORK/target" "$WORK/bin"

if [[ $MODE == fix ]]; then
  tree "$WORK/base-src" "$BASE_REV"
  apply_once "$WORK/base-src" "$ROOT/bench/gmin-losses/no-template-base.patch"
  git -C "$WORK/base-src" add examples src/methods/cluster_hopping/config.rs
  base_tree=$(git -C "$WORK/base-src" write-tree)
  patch_sha256=$(sha256sum "$ROOT/bench/gmin-losses/no-template-base.patch" | cut -d' ' -f1)
  cat > "$OUT/baseline-source.txt" << EOF
revision=$BASE_REV
template_ablation_patch_sha256=$patch_sha256
patched_tree=$base_tree
EOF
  build_bins base "$WORK/base-src" "$BASE_REV" "$WORK/base-target" "$WORK/base-bin"
fi

GMIN_PANEL_WORK=$WORK bash "$HERE/build_gmin.sh"
sha256sum "$WORK"/bin/* > "$WORK/bin-sha256.txt"
if [[ $MODE == fix ]]; then
  sha256sum "$WORK"/base-bin/* > "$WORK/base-bin-sha256.txt"
fi
cp -f "$WORK/gmin-source.txt" "$OUT/gmin-source.txt"
(cd "$WORK/bin" && sha256sum -- *) > "$OUT/bin-sha256.txt"
cp -f "$HERE/run_panel.py" "$WORK/run_panel.py"
cp -f "$HERE/run_fix_gate.py" "$WORK/run_fix_gate.py"
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
exec taskset -c 0 /usr/bin/python3 -u $WORK/$RUNNER \\
  --root $ROOT \\
  --rows $OUT/rows.csv \\
  ${SUMMARY[*]} \\
  --revision $REV \\
  ${BASE_RUN[*]} \\
  --bin $WORK/bin \\
  --tree $WORK/anneal-src \\
  --work $WORK/runs \\
  --logs $WORK/logs \\
  --gmin-source $WORK/gmin-source.txt \\
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
