#!/usr/bin/env bash
# Shell-turn anneal arm, (H2O)21 against fixed-step GMIN, and the LJ check.
# One UTS namespace, hostname bench, one core. Rows are appended.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if [[ ! -x /work/tip4p-fix/target/release/examples/h2h_seed ]]; then
  echo "tip binary missing" >&2
  exit 1
fi
if [[ ! -x /work/tip4p-fix/target/release/examples/h2h_gmin_proof ]]; then
  echo "proof binary missing" >&2
  exit 1
fi
if [[ ! -x /work/gmin-h2h/bin/GMIN ]]; then
  echo "GMIN binary missing" >&2
  exit 1
fi

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; cd /work; exec sudo -u ubuntu env ANNEAL_ROOT='"$ROOT"' HOME=/work/tip4p-h2h/home TMPDIR=/work/tip4p-h2h/tmp OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 /usr/bin/python3 -u '"$HERE"'/run_shell.py'
