#!/bin/sh
# Writes paper/jctc/logs/cargo-build-receipt.txt
set -eu
out=${1:-paper/jctc/logs/cargo-build-receipt.txt}
mkdir -p "$(dirname "$out")"
export CARGO_TARGET_DIR=target
lib=$(gcc -print-file-name=libstdc++.so)
export LIBRARY_PATH="$(dirname "$lib")${LIBRARY_PATH:+:$LIBRARY_PATH}"
{
  printf 'UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'HEAD %s\n' "$(git rev-parse HEAD)"
  printf 'TREE %s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'command cargo +stable build --all-targets --locked\n'
  set +e
  cargo +stable build --all-targets --locked
  printf 'EXIT:%s\n' "$?"
  printf 'command cargo +stable build --all-targets --all-features --locked\n'
  cargo +stable build --all-targets --all-features --locked
  printf 'EXIT:%s\n' "$?"
  printf 'command cargo +stable bench --no-run\n'
  cargo +stable bench --no-run
  printf 'EXIT:%s\n' "$?"
  set -e
} > "$out.raw" 2>&1
python3 - "$out.raw" "$out" << 'PY'
import sys
from pathlib import Path
src, dest = sys.argv[1], sys.argv[2]
text = Path(src).read_text(errors="replace")
for token in ("/tmp/", "/home/", "/cursor/", "/root/", "/var/", "/opt/", "/workspace/", "/agent/", "/usr/"):
    text = text.replace(token, "./")
Path(dest).write_text(text)
Path(src).unlink()
PY
