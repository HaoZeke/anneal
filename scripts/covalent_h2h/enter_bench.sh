#!/usr/bin/env bash
# Every code, one UTS namespace, hostname bench, working directory under /work.
# Arguments go to run_campaign.py. BENCH_OUT names the output directory
# under bench/covalent-h2h, default "before".
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/covalent-h2h
OUT=$ROOT/bench/covalent-h2h/${BENCH_OUT:-before}

bash "$HERE/build_anneal.sh"
bash "$HERE/build_gmin.sh"

mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" "$OUT"
RUN=$WORK/${BENCH_OUT:-before}
mkdir -p "$RUN/runs" "$RUN/starts"
cp -a "$HERE/run_campaign.py" "$HERE/ase_bh.py" "$WORK/bin/"
cp -a "$ROOT/examples/h2h_gmin_sw.data" "$WORK/bin/h2h_gmin_sw.data"
cp -a /work/gmin-h2h/bin/GMIN "$WORK/bin/GMIN"
cp -a /work/gmin-h2h/gmin-source.txt "$WORK/gmin-source.txt"
chmod +x "$WORK/bin/h2h_covalent" "$WORK/bin/GMIN"

cat > "$WORK/bin/inside.sh" << EOF2
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:\$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
export RGPOT_COVALENT_LIB=$WORK/lib/librgpot_covalent.so
cd $WORK
exec /usr/bin/python3 -u $WORK/bin/run_campaign.py \\
  --binary $WORK/bin/h2h_covalent \\
  --library $WORK/lib/librgpot_covalent.so \\
  --gmin $WORK/bin/GMIN \\
  --data $WORK/bin/h2h_gmin_sw.data \\
  --ase $WORK/bin/ase_bh.py \\
  --source $WORK/gmin-source.txt \\
  --head-file $WORK/anneal-head \\
  --runs $RUN/runs \\
  --starts $RUN/starts \\
  --rows $OUT/seeds.csv \\
  --summary $OUT/results.csv \\
  "\$@"
EOF2
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/covalent-h2h/bin/inside.sh "$@"' _ "$@"
