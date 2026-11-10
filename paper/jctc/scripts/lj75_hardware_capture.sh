#!/bin/sh
# Copyright (c) anneal developers
# Writes paper/jctc/logs/lj75-hardware-receipt.txt from a live process.
# Usage: BUILD_HEAD=... BUILD_TREE=... sh scripts/lj75_hardware_capture.sh PID
set -eu
pid=${1:?pid}
out=${2:-paper/jctc/logs/lj75-hardware-receipt.txt}
head_sha=${BUILD_HEAD:?}
tree_sha=${BUILD_TREE:?}
utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$(dirname "$out")"
{
  printf 'UTC %s\n' "$utc"
  printf 'HEAD %s\n' "$head_sha"
  printf 'TREE %s\n' "$tree_sha"
  printf 'host host-a\n'
  printf 'nproc\n'
  nproc
  printf 'lscpu\n'
  lscpu
  printf 'cmdline\n'
  python3 - "$pid" << 'PY'
import sys
from pathlib import Path
pid = sys.argv[1]
parts = [p.decode() for p in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if p]
if parts:
    name = parts[0].rsplit("/", 1)[-1]
    parts[0] = "target/release/examples/" + name
sys.stdout.write(" ".join(parts) + "\n")
PY
  printf 'sha256\n'
  hash=$(sha256sum "/proc/${pid}/exe" | awk 'NR==1 {print $1}')
  printf '%s target/release/examples/lj_cluster_search\n' "$hash"
} > "$out"
