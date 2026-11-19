#!/bin/bash
# Copyright (c) anneal developers
# Serial nm3 gate. Hostname bench, core 0, then the summarizer.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

if [[ ${1:-} == --summarize ]]; then
  exec python3 bench/nm3-gate/summarize.py
fi

if [[ -z ${NM3_A_BIN:-} || -z ${NM3_B_BIN:-} || -z ${NM3_A_CWD:-} || -z ${NM3_B_CWD:-} ]]; then
  echo "NM3_A_BIN, NM3_B_BIN, NM3_A_CWD and NM3_B_CWD are required" >&2
  exit 2
fi

run() {
  python3 bench/nm3-gate/run_gate.py
  python3 bench/nm3-gate/summarize.py
}

if [[ $(hostname) == bench ]] && python3 -c 'import os,sys; sys.exit(0 if os.sched_getaffinity(0)=={0} else 1)'; then
  run
  exit 0
fi

exec unshare -r --uts bash -c 'hostname bench; exec taskset -c 0 bash bench/nm3-gate/enter.sh --inside'
