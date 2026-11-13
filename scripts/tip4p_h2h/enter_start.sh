#!/usr/bin/env bash
# TIP4P with GMIN opening anneal's rigid start: anneal at 4 and 8, GMIN with
# and without a container radius at 4, 8, 15, 20. One UTS namespace,
# hostname bench, one core. Rows are appended.
# H2H_SEED_BIN is an h2h_seed build that writes the chart vectors unchanged.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SEED_BIN=${H2H_SEED_BIN:?set H2H_SEED_BIN to an h2h_seed binary}

if [[ ! -x $SEED_BIN ]]; then
  echo "h2h_seed missing at $SEED_BIN" >&2
  exit 1
fi
if [[ ! -x /work/gmin-h2h/bin/GMIN ]]; then
  echo "GMIN binary missing; run scripts/gmin_h2h/build_gmin.sh" >&2
  exit 1
fi
mkdir -p /work/tip4p-h2h/home /work/tip4p-h2h/tmp

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; cd /work; exec sudo -u '"$(id -un)"' env ANNEAL_ROOT='"$ROOT"' H2H_SEED_BIN='"$SEED_BIN"' HOME=/work/tip4p-h2h/home TMPDIR=/work/tip4p-h2h/tmp OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 /usr/bin/python3 -u '"$HERE"'/run_start.py'
