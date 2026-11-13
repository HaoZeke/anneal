#!/usr/bin/env bash
# Morse rho = 14, twelve seeds, one UTS namespace, hostname bench.
# Anneal is run serially. GMIN rows already stored at the paper budget
# are copied and not run again.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /usr/bin/python3 -u '"$HERE"'/run_panel.py --root '"$ROOT"' "$@"' _ "$@"
