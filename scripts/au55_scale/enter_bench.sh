#!/usr/bin/env bash
# Copyright (c) anneal developers
# Gupta Au55, twelve seeds, one UTS namespace, hostname bench.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${GMIN_DARBY_WORK:-/work/metal-darby}
GMIN_SVNREV_FILE=${GMIN_SVNREV_FILE:-$WORK/src/wales/GMIN/source/SVNREV}
GMIN_SOURCE_FILE=${GMIN_SOURCE_FILE:-$WORK/gmin-source.txt}

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'svnrev=$1; source=$2; panel=$3; root=$4; shift 4; hostname bench; exec sudo -u ubuntu env GMIN_SVNREV_FILE="$svnrev" GMIN_SOURCE_FILE="$source" /usr/bin/python3 -u "$panel" --root "$root" "$@"' \
  _ "$GMIN_SVNREV_FILE" "$GMIN_SOURCE_FILE" "$HERE/run_panel.py" "$ROOT" "$@"
