#!/usr/bin/env bash
# Copyright (c) anneal developers
# GMIN alone on Sutton-Chen 9-6, N=38 and N=78. One UTS namespace, hostname
# bench, working directory /work, one job at a time. The anneal rows are not
# rerun. The harness must be committed, because its HEAD is the
# harness_revision of every row.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
RUNS=$WORK/runs-sc96-gmin
LOGS=$WORK/logs-sc96-gmin
STARTS78=$WORK/sc96-78/starts

harness=(scripts/metal_h2h scripts/sc96_78_common.py
  bench/metal-cluster-h2h/gmin-sc-time.patch bench/lj104-lattice/v2/h2h-gmin.patch)
if ! git -C "$ROOT" diff --quiet HEAD -- "${harness[@]}" ||
  [[ -n $(git -C "$ROOT" ls-files --others --exclude-standard -- "${harness[@]}") ]]; then
  echo "commit the harness before the run" >&2
  exit 1
fi

bash "$HERE/build_sc.sh"
bash "$HERE/build_gmin_sc.sh"
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$RUNS" "$LOGS" "$STARTS78"
cp -a "$HERE/run_sc.py" "$HERE/run_sc_n12.py" "$HERE/run_sc96_gmin.py" "$WORK/bin/"
chmod +x "$WORK/bin/h2h_sc" "$WORK/bin/GMIN" "$WORK/bin/run_sc96_gmin.py"

# The N=78 starts and the Cambridge energy need numpy and the network, so
# they are written here, outside the namespace.
python3 - "$ROOT" "$STARTS78" "$WORK/sc96-78" << 'EOF'
import sys
from pathlib import Path

root, starts, scratch = (Path(arg) for arg in sys.argv[1:])
sys.path.insert(0, str(root / "scripts"))
from sc96_78_common import reference_energy, write_start

for seed in range(9532, 9564):
    write_start(starts / f"seed{seed}.xyz", 78, seed)
(scratch / "reference.txt").write_text(f"{reference_energy(scratch):.4f}\n")
EOF

cat > "$WORK/bin/inside-sc96-gmin.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd /work
exec /usr/bin/python3 -u $WORK/bin/run_sc96_gmin.py \\
  --binary $WORK/bin/h2h_sc \\
  --gmin $WORK/bin/GMIN \\
  --runs $RUNS \\
  --logs $LOGS \\
  --n38-seeds $ROOT/bench/metal-cluster-h2h/sc-n12/seeds.csv \\
  --n38-summary $ROOT/bench/metal-cluster-h2h/sc-n12/results.csv \\
  --n78-rows $ROOT/data/sc96_78_hits.csv \\
  --starts78 $STARTS78 \\
  --reference78 $WORK/sc96-78/reference.txt \\
  --source $WORK/gmin-source.txt \\
  --head-file $WORK/anneal-head \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside-sc96-gmin.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-sc96-gmin.sh "$@"' _ "$@"
