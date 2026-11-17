# Copyright (c) anneal developers
# Same-session gate of plateau_heat. Builds the base revision, the revision
# that adds the flag and, when GATE_RECORDS_REV is set, the revision that
# picks the heat by records per call, in detached worktrees under /work,
# then runs run_gate.py in one UTS namespace with hostname bench, one
# process on core 0. sc108-55 loads rgpot's Sutton-Chen library from
# RGPOT_SC_LIB.
#
#   RGPOT_SC_LIB=/path/librgpot.so GATE_HEAT_REV=... [GATE_RECORDS_REV=...] \
#     enter_bench.sh [run_gate.py options]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BASE_REV=${GATE_BASE_REV:-24ed9551f2a574622fae77ec5f8a4077e1dfd336}
HEAT_REV=${GATE_HEAT_REV:?the revision that adds plateau_heat}
RECORDS_REV=${GATE_RECORDS_REV:-}
WORK=${GATE_WORK:-/work/au55-general2/gate}
OUT=${GATE_OUT:-$ROOT/bench/au55-general2}
cargo_home=${CARGO_HOME:-$HOME/.cargo}
: "${RGPOT_SC_LIB:?the rgpot Sutton-Chen library}"

tree() {
  local dest=$1 rev=$2
  if [ ! -d "$dest" ]; then
    git -C "$ROOT" worktree add --detach "$dest" "$rev"
  fi
  git -C "$dest" checkout --detach -q "$rev"
  if [ -n "$(git -C "$dest" status --porcelain)" ]; then
    echo "$dest differs from $rev" >&2
    exit 1
  fi
}

build() {
  local src=$1 target=$2 bin=$3
  mkdir -p "$target" "$bin" "$WORK/rustc-tmp"
  (
    cd "$src"
    export CARGO_TARGET_DIR=$target TMPDIR=$WORK/rustc-tmp
    export RUSTFLAGS="--remap-path-prefix=${src}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
    cargo +1.88.0 build --release --locked --example h2h_gmin_proof --example h2h_morse --example h2h_metal -j 4
    cargo +1.88.0 build --release --locked --features rgpot-ex --example h2h_sc -j 4
  )
  for example in h2h_gmin_proof h2h_morse h2h_metal h2h_sc; do
    cp -f "$target/release/examples/$example" "$bin/"
  done
}

mkdir -p "$WORK/home" "$WORK/scratch" "$WORK/logs" "$OUT"
tree "$WORK/base-src" "$BASE_REV"
tree "$WORK/heat-src" "$HEAT_REV"
build "$WORK/base-src" "$WORK/target-base" "$WORK/bin-base"
build "$WORK/heat-src" "$WORK/target-heat" "$WORK/bin-heat"
records_args=""
if [ -n "$RECORDS_REV" ]; then
  tree "$WORK/records-src" "$RECORDS_REV"
  build "$WORK/records-src" "$WORK/target-records" "$WORK/bin-records"
  records_args="--records-revision $RECORDS_REV --records-bin $WORK/bin-records --records-tree $WORK/records-src"
fi
sha256sum "$WORK"/bin-*/* "$RGPOT_SC_LIB" > "$WORK/bin-sha256.txt"
cp -f "$HERE/run_gate.py" "$WORK/run_gate.py"
user_site=$(python3 -c 'import site; print(site.getusersitepackages())')
lib_dir=$(dirname "$RGPOT_SC_LIB")

cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/scratch
export PYTHONPATH=$user_site\${PYTHONPATH:+:\$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export RGPOT_SC_LIB=$RGPOT_SC_LIB
export LD_LIBRARY_PATH=$lib_dir\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
cd $WORK
exec taskset -c 0 /usr/bin/python3 -u $WORK/run_gate.py \\
  --rows $OUT/rows.csv \\
  --base-revision $BASE_REV \\
  --heat-revision $HEAT_REV \\
  --base-bin $WORK/bin-base \\
  --heat-bin $WORK/bin-heat \\
  --base-tree $WORK/base-src \\
  --heat-tree $WORK/heat-src \\
  $records_args \\
  --logs $WORK/logs \\
  --summary $OUT/summary.csv \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u $(id -un) $WORK/inside.sh \"\$@\"" _ "$@"
