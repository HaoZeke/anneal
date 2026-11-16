#!/usr/bin/env bash
# Copyright (c) anneal developers
# One UTS namespace, hostname bench. The GFN2 water proof and the CREST
# head-to-head share that session. Extra arguments reach run_campaign.py.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/water-gfn2-scc2

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$WORK/runs" "$ROOT/bench/water-gfn2-scc2"

export PATH=/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin:$PATH
CARGO_TARGET_DIR=$WORK/target rustup run 1.88.0 cargo build \
  --locked --offline --release --example gfn2_water_h2h --features rgpot-ex \
  --manifest-path "$ROOT/Cargo.toml"
cp -a "$WORK/target/release/examples/gfn2_water_h2h" "$WORK/bin/gfn2_water_h2h"
cp -a "$HERE/run_campaign.py" "$WORK/bin/"
git -C "$ROOT" rev-parse HEAD > "$WORK/anneal-head"

CREST=${CREST_BIN:-/work/crestenv/.pixi/envs/default/bin/crest}
XTB=${XTB_BIN:-/work/xtb-gfn2}
UNFIXED=${UNFIXED_BIN:-}

user=$(id -un)
cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RGPOT_XTB_ENGINE=\${RGPOT_XTB_ENGINE:-/work/rgpot-bdir/CppCore/rgpot/XTBPot/libxtb_engine.so}
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-/work/rgpot/.pixi/envs/xtbbld/lib}
cd $WORK
extra=""
if [[ -n "${UNFIXED}" ]]; then
  extra="--unfixed ${UNFIXED}"
fi
exec taskset -c 0 python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/gfn2_water_h2h \\
  \$extra \\
  --crest $CREST \\
  --xtb $XTB \\
  --runs $WORK/runs \\
  --out $ROOT/bench/water-gfn2-scc2 \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u "$0" '"$WORK"'/bin/inside.sh "$@"' "$user" "$@"
