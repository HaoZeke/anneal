#!/usr/bin/env bash
# M55, M70 and M80 Morse panel. Hostname bench, one core.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

PY=/usr/bin/python3
if [ -x /tmp/anneal-venv/bin/python3 ]; then
  PY=/tmp/anneal-venv/bin/python3
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u ubuntu $PY -u $HERE/run_panel.py --root $ROOT \"\$@\"" _ "$@"
