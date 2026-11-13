#!/usr/bin/env bash
# Gupta Au55, twelve seeds, one UTS namespace, hostname bench.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /usr/bin/python3 -u '"$HERE"'/run_panel.py --root '"$ROOT"' "$@"' _ "$@"
