#!/usr/bin/env bash
# Adaptive GMIN, the 4fdc37e8 anneal arm, and the LJ75/LJ104 check.
# One UTS namespace, hostname bench, one core. Rows are appended.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if [[ ! -x /work/anneal-4fdc37e8/target/release/examples/h2h_seed ]]; then
  echo "develop binary missing" >&2
  exit 1
fi
if [[ ! -x /work/tip4p-fix/target/release/examples/h2h_gmin_proof ]]; then
  echo "tip binary missing" >&2
  exit 1
fi

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; cd /work; exec sudo -u ubuntu env ANNEAL_ROOT='"$ROOT"' HOME=/work/tip4p-h2h/home TMPDIR=/work/tip4p-h2h/tmp OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 /usr/bin/python3 -u '"$HERE"'/run_followup.py'
