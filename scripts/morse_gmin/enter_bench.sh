#!/usr/bin/env bash
# Copyright (c) anneal developers
# Morse against GMIN. One UTS namespace, hostname bench, one core.
# A stored GMIN row is copied when GMIN/source/SVNREV is still
# aac1548a8aec13b53ad12a8d6904625081685d2e. Anneal is this tree.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /usr/bin/python3 -u '"$HERE"'/run_panel.py --root '"$ROOT"' "$@"' _ "$@"
