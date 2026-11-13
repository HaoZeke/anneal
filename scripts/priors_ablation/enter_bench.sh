#!/usr/bin/env bash
# All three arms in one UTS namespace, hostname bench, working directory /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/abl
LIB=${RGPOT_SC_LIB:-/work/rgpot/bdir/CppCore/librgpot.so}
cargo_home=${CARGO_HOME:-$HOME/.cargo}

mkdir -p "$WORK/bin" "$WORK/target" "$WORK/rustc-tmp" "$WORK/home" "$WORK/tmp" "$WORK/runs"
(
  cd "$ROOT"
  export CARGO_TARGET_DIR=$WORK/target
  export TMPDIR=$WORK/rustc-tmp
  export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
  cargo +1.88.0 build --release --locked --offline -j 4 \
    --example h2h_gmin_proof --example h2h_morse --example h2h_metal --example h2h_seed
  cargo +1.88.0 build --release --locked --offline -j 4 --features rgpot-ex --example h2h_sc
  for b in h2h_gmin_proof h2h_morse h2h_metal h2h_seed h2h_sc; do
    cp -a "$WORK/target/release/examples/$b" "$WORK/bin/$b"
  done
  git rev-parse HEAD > "$WORK/anneal-head"
)
cp -a "$HERE/run_ablation.py" "$WORK/bin/run_ablation.py"

cat > "$WORK/bin/inside.sh" << INNER
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/usr/bin:/bin
export RGPOT_SC_LIB=$LIB
export LD_LIBRARY_PATH=$(dirname "$LIB")
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd /work
exec /usr/bin/python3 -u $WORK/bin/run_ablation.py \\
  --bins $WORK/bin --runs $WORK/runs --rows $WORK/rows.csv "\$@"
INNER
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/abl/bin/inside.sh "$@"' _ "$@"
