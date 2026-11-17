#!/usr/bin/env bash
# Copyright (c) anneal developers
# All three arms, one UTS namespace, hostname bench, working directory under
# /work. The campaign and every timed child are pinned to core 0.
#
# With no arguments this runs the campaign against bench/abcluster-h2h and
# then rewrites the summary and the README table from the committed rows.
# Extra arguments go to run_campaign.py after the defaults, so --out and
# --runs can point a single timed seed (the asv benchmark) at scratch, and
# --pilot runs the ABCluster population pilot on the spare cores.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/abcluster-h2h
OUT=$ROOT/bench/abcluster-h2h

bash "$HERE/build_anneal.sh"
bash "$HERE/fetch_abcluster.sh"

mkdir -p "$WORK/home" "$WORK/tmp" "$WORK/runs" "$OUT"
cp -a "$HERE/run_campaign.py" "$WORK/bin/run_campaign.py"
user=$(id -un)

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd $WORK
exec taskset -c 0 /usr/bin/python3 -u $WORK/bin/run_campaign.py \\
  --anneal $WORK/bin/h2h_abcluster \\
  --atom $WORK/abcluster-Release/atom \\
  --shim $WORK/bin/abc_shim.so \\
  --pins $OUT/abcluster-source.txt \\
  --head-file $WORK/anneal-head \\
  --runs $WORK/runs \\
  --out $OUT \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
status=0
sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u "$0" '"$WORK"'/bin/inside.sh "$@"' "$user" "$@" || status=$?
if [[ $status -eq 0 && $# -eq 0 ]]; then
  python3 "$HERE/summarize.py"
fi
exit $status
