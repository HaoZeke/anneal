#!/usr/bin/env bash
# One M80 rho=3 seed, hostname bench, serial. Rewrites the summary with
# the parent rows measured on this VM and a fresh row from the current
# binary. Pass --budget to time a single budget and leave the summary.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /usr/bin/python3 -u '"$HERE"'/run_hop.py --root '"$ROOT"' "$@"' _ "$@"
