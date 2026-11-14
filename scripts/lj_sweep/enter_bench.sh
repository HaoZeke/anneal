#!/usr/bin/env bash
# Anneal arm only. GMIN rows are reused. Hostname bench, one process.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/lj-sweep
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$WORK/rustc-tmp" "$WORK/target"

cargo_home=${CARGO_HOME:-$HOME/.cargo}
cd "$ROOT"
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/rustc-tmp
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
cargo +1.88.0 build --release --locked --example h2h_gmin_proof -j 4
cp -a "$WORK/target/release/examples/h2h_gmin_proof" "$WORK/bin/h2h_gmin_proof"
chmod +x "$WORK/bin/h2h_gmin_proof"
git rev-parse HEAD > "$WORK/anneal-head"
cp -a "$HERE/run_campaign.py" "$WORK/bin/run_campaign.py"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
cd $WORK
exec taskset -c 0 /usr/bin/python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/h2h_gmin_proof \\
  --root $ROOT \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs \\
  --summary $ROOT/bench/lj-sweep/summary.csv \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/lj-sweep/bin/inside.sh "$@"' _ "$@"
