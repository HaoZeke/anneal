#!/usr/bin/env bash
# Copyright (c) anneal developers
# Anneal alone on every Sutton-Chen system with GMIN head-to-head rows: the
# sc-n12 panel, 9-6 at N=78, and the sc96-55 check against e3ed7dad (see
# run_sc_anneal.py). One UTS namespace, hostname bench, working directory
# /work, one job at a time on core 0. The GMIN rows are not rerun. The
# harness must be committed, because its HEAD is the harness_revision of
# every row.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
RUNS=$WORK/runs-sc-anneal
LOGS=$WORK/logs-sc-anneal
STARTS78=$WORK/sc96-78/starts
CHECK=e3ed7dadfce4f0c0aa3b9616f6f07dd9b3171e0a

harness=(scripts/metal_h2h scripts/sc96_78_common.py scripts/sc96_78_hits.py examples/sc_h2h.rs)
if ! git -C "$ROOT" diff --quiet HEAD -- "${harness[@]}" ||
  [[ -n $(git -C "$ROOT" ls-files --others --exclude-standard -- "${harness[@]}") ]]; then
  echo "commit the harness before the run" >&2
  exit 1
fi

bash "$HERE/build_sc.sh"
bash "$HERE/build_rgpot_sc.sh"
if [[ ! -x $WORK/bin/h2h_sc-${CHECK:0:8} ]]; then
  rm -rf "$WORK/src-${CHECK:0:8}"
  git -C "$ROOT" worktree prune
  git -C "$ROOT" worktree add --detach "$WORK/src-${CHECK:0:8}" "$CHECK"
  (cd "$WORK/src-${CHECK:0:8}" &&
    CARGO_TARGET_DIR=$WORK/target-${CHECK:0:8} cargo +1.88.0 build --release --locked --features rgpot-ex \
      --example h2h_sc -j 4)
  cp -a "$WORK/target-${CHECK:0:8}/release/examples/h2h_sc" "$WORK/bin/h2h_sc-${CHECK:0:8}"
  git -C "$ROOT" worktree remove --force "$WORK/src-${CHECK:0:8}"
fi
mkdir -p "$WORK/home" "$WORK/tmp" "$RUNS" "$LOGS" "$STARTS78"

# The N=78 starts and the Cambridge energy need numpy and the network, so
# they are written here, outside the namespace.
python3 - "$ROOT" "$STARTS78" "$WORK/sc96-78" << 'PY'
import sys
from pathlib import Path

root, starts, scratch = (Path(arg) for arg in sys.argv[1:])
sys.path.insert(0, str(root / "scripts"))
from sc96_78_common import reference_energy, write_start

for seed in range(9532, 9564):
    write_start(starts / f"seed{seed}.xyz", 78, seed)
(scratch / "reference.txt").write_text(f"{reference_energy(scratch):.4f}\n")
PY

cat > "$WORK/bin/inside-sc-anneal.sh" << INNER
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export RGPOT_SC_LIB=$WORK/rgpot-sc/librgpot.so
cd /work
exec /usr/bin/python3 -u $HERE/run_sc_anneal.py \\
  --binary $WORK/bin/h2h_sc \\
  --sc-h2h $WORK/bin/sc_h2h \\
  --check-binary $WORK/bin/h2h_sc-${CHECK:0:8} \\
  --check-revision $CHECK \\
  --runs $RUNS \\
  --logs $LOGS \\
  --panel-seeds $ROOT/bench/metal-cluster-h2h/sc-n12/seeds.csv \\
  --panel-summary $ROOT/bench/metal-cluster-h2h/sc-n12/results.csv \\
  --check-rows $ROOT/bench/metal-cluster-h2h/sc-n12/sc96-55-seeds.csv \\
  --n78-rows $ROOT/data/sc96_78_hits.csv \\
  --starts78 $STARTS78 \\
  --reference78 $WORK/sc96-78/reference.txt \\
  --head-file $WORK/anneal-head \\
  "\$@"
INNER
chmod +x "$WORK/bin/inside-sc-anneal.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-sc-anneal.sh "$@"' _ "$@"
