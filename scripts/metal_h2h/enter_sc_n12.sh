#!/usr/bin/env bash
# Copyright (c) anneal developers
# Twelve-seed Sutton–Chen head-to-head. One UTS namespace, hostname bench.
# An existing results, seed, or session file is refused. The six-seed table
# is not opened.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
OUTPUT=${SC_H2H_N12_OUTPUT:-$WORK/output}
if [[ -e $OUTPUT/results.csv || -e $OUTPUT/seeds.csv || -e $OUTPUT/sessions.jsonl ]]; then
  echo "refusing to overwrite output under $OUTPUT" >&2
  exit 1
fi
LOGS=$OUTPUT/logs
RUNS=$WORK/runs-n12
LIB=${RGPOT_SC_LIB:-/work/rgpot/bdir-xtb/CppCore/librgpot.so}

bash "$HERE/build_sc.sh"
bash "$HERE/build_gmin_sc.sh"
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$RUNS" "$LOGS" "$OUTPUT" /work \
  "$ROOT/bench/metal-cluster-h2h/sc-n12"
cp -a "$HERE/run_sc.py" "$HERE/run_sc_n12.py" \
  "$ROOT/scripts/target_tol.py" "$ROOT/scripts/gmin_pin.py" \
  "$ROOT/scripts/machine_provenance.py" "$WORK/bin/"
chmod +x "$WORK/bin/h2h_sc" "$WORK/bin/GMIN" "$WORK/bin/run_sc.py" "$WORK/bin/run_sc_n12.py"

cat > "$WORK/bin/inside-n12.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/usr/bin:/bin
export RGPOT_SC_LIB=$LIB
export LD_LIBRARY_PATH=$(dirname "$LIB")\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd /work
taskset -c 0 /usr/bin/python3 $ROOT/scripts/session_metadata.py \\
  --output $OUTPUT/sessions.jsonl \\
  --campaign sutton-chen-n12 --root $ROOT -- \\
  snakemake -s workflow/Snakefile sc_h2h_n12 --cores 1 --force
exec taskset -c 0 /usr/bin/python3 -u $WORK/bin/run_sc_n12.py \\
  --binary $WORK/bin/h2h_sc \\
  --gmin $WORK/bin/GMIN \\
  --runs $RUNS \\
  --logs $LOGS \\
  --summary $OUTPUT/results.csv \\
  --seeds-out $OUTPUT/seeds.csv \\
  --source $WORK/gmin-source.txt \\
  --svnrev /work/metal-darby/src/wales/GMIN/source/SVNREV \\
  --head-file $WORK/anneal-head \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside-n12.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-n12.sh "$@"' _ "$@"
