#!/usr/bin/env bash
# Copyright (c) anneal developers
# One UTS namespace, hostname bench, serial on core 0.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
cd "$ROOT"
cargo build --release --locked --example cmcts_au
BIN=$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/cmcts_au
PY="${CMCTS_PYTHON:-/tmp/cmcts-venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
mkdir -p "$ROOT/bench/cmcts-au/logs"
INSIDE=/tmp/cmcts-au-inside.sh
cat > "$INSIDE" << EOF
#!/bin/bash
set -eu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export PYTHONUNBUFFERED=1
cd "$ROOT"
exec taskset -c 0 "$PY" -u "$HERE/run_campaign.py" --binary "$BIN" "\$@"
EOF
chmod 755 "$INSIDE"
if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c 'hostname bench; exec sudo -u ubuntu /tmp/cmcts-au-inside.sh "$@"' _ "$@"
