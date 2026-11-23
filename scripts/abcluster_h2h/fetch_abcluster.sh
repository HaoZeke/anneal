#!/usr/bin/env bash
# Copyright (c) anneal developers
# The official ABCluster 3.4 Linux release, checked against the pinned
# checksums, and the counting shim built against it. Nothing here needs a
# registration or a licence step: the executable link on the ABCluster page
# is a plain download.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PINS=$ROOT/bench/abcluster-h2h/abcluster-source.txt
WORK=/work/abcluster-h2h
if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK/download" "$WORK/bin"

pin() { grep -E "^$1=" "$PINS" | head -n 1 | cut -d= -f2-; }
url=$(pin url)
tarball_sha=$(pin tarball_sha256)
atom_sha=$(pin atom_sha256)
tarball=$WORK/download/$(basename "$url")

if [[ ! -f "$tarball" ]]; then
  curl -fsSL --retry 4 -o "$tarball" "$url"
fi
echo "$tarball_sha  $tarball" | sha256sum -c -
if [[ ! -x "$WORK/abcluster-Release/atom" ]]; then
  tar -xzf "$tarball" -C "$WORK"
fi
echo "$atom_sha  $WORK/abcluster-Release/atom" | sha256sum -c -

gcc -O2 -fPIC -shared -Wall -Wextra -o "$WORK/bin/abc_shim.so" "$HERE/abc_shim.c" -ldl
echo "abcluster atom=$WORK/abcluster-Release/atom shim=$WORK/bin/abc_shim.so"
