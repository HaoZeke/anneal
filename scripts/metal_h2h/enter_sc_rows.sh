#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build the Sutton-Chen binary and append raw rows on hostname bench.
# One campaign per argument: restart, pop, chain, or portfolio.
# Finished seeds already in the table are kept.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
LIB=${RGPOT_SC_LIB:-/work/rgpot/bdir-xtb/CppCore/librgpot.so}

if [ "$#" -lt 1 ]; then
  echo "usage: enter_sc_rows.sh restart|pop|chain|portfolio ..." >&2
  exit 2
fi

bash "$HERE/build_sc.sh"
mkdir -p "$WORK/bin" "$WORK/home" "$WORK/tmp" /work
cp -a "$HERE/run_sc_rows.py" "$WORK/bin/run_sc_rows.py"
chmod +x "$WORK/bin/h2h_sc" "$WORK/bin/run_sc_rows.py"

cat > "$WORK/bin/inside-rows.sh" << EOF
#!/bin/bash
set -euo pipefail
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:/usr/bin:/bin
export RGPOT_SC_LIB=$LIB
export LD_LIBRARY_PATH=$(dirname "$LIB")\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
cd /work
for campaign in "\$@"; do
  python3 $WORK/bin/run_sc_rows.py "\$campaign" \\
    --binary $WORK/bin/h2h_sc \\
    --head-file $WORK/anneal-head \\
    --root $ROOT
done
EOF
chmod +x "$WORK/bin/inside-rows.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; exec bash /work/metal-sc/bin/inside-rows.sh "$@"' _ "$@"
