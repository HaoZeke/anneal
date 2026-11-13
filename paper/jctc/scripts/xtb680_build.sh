#!/bin/sh
# Copyright (c) anneal developers
# Build xtb 6.8.0 from the tagged release. The sources and the binary
# stay under /work. Run on a host named bench:
#   sudo unshare --uts sh -c 'hostname bench; sh paper/jctc/scripts/xtb680_build.sh'
set -eu
if [ "$(hostname)" != bench ]; then
  echo "xtb680_build: hostname is $(hostname), want bench" >&2
  exit 1
fi
root=/work/xtb-6.8.0
mkdir -p /work
if [ ! -d "$root/.git" ]; then
  git clone --depth 1 --branch v6.8.0 https://github.com/grimme-lab/xtb.git "$root"
fi
cd "$root"
if [ ! -f build/build.ninja ]; then
  CC=gcc meson setup build -Dlapack=openblas --buildtype=release
fi
ninja -C build
