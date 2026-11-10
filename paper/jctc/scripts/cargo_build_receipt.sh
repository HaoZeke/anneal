#!/bin/sh
# Writes paper/jctc/logs/cargo-build-receipt.txt
set -eu
out=${1:-paper/jctc/logs/cargo-build-receipt.txt}
mkdir -p "$(dirname "$out")"
{
  printf 'UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'HEAD %s\n' "$(git rev-parse HEAD)"
  printf 'TREE %s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'command cargo build --all-targets --all-features --locked\n'
  cargo build --all-targets --all-features --locked
  printf 'command cargo bench --no-run\n'
  cargo bench --no-run
} > "$out.raw" 2>&1
python3 - "$out.raw" "$out" << 'PY'
import sys
from pathlib import Path
src, dest = sys.argv[1], sys.argv[2]
text = Path(src).read_text(errors="replace")
for token in ("/tmp/", "/home/", "/cursor/", "/root/", "/opt/", "/workspace/", "/agent/"):
    text = text.replace(token, "./")
Path(dest).write_text(text)
Path(src).unlink()
PY
