#!/bin/sh
# Writes paper/jctc/logs/cargo-build-receipt.txt
# vesin-nl needs the CI checkout of vesin at .ci/vesin.
# ira, lammps, and gpr need libraries this machine does not have.
set -eu
out=${1:-paper/jctc/logs/cargo-build-receipt.txt}
mkdir -p "$(dirname "$out")" .ci
if [ ! -f .ci/vesin/vesin/src/vesin.cpp ]; then
  git clone --quiet https://github.com/Luthaf/vesin.git .ci/vesin
  git -C .ci/vesin checkout --quiet 5ac39b6e57a8f00c717f08c83f93ee4746d6e471
fi
root=$(pwd)
export VESIN_SRC="$root/.ci/vesin"
export CARGO_TARGET_DIR=target
export CXX=g++
export CC=gcc
lib=$(gcc -print-file-name=libstdc++.so)
export LIBRARY_PATH="$(dirname "$lib")${LIBRARY_PATH:+:$LIBRARY_PATH}"
others=graphkey,featomic,vesin-nl,bank-rpc,history-nng,rgsaddle,capi,python,rgpot-ex
{
  printf 'UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'HEAD %s\n' "$(git rev-parse HEAD)"
  printf 'TREE %s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'VESIN %s\n' "$(git -C .ci/vesin rev-parse HEAD)"
  printf 'command cargo +stable build --all-targets --locked\n'
  set +e
  cargo +stable build --all-targets --locked
  printf 'EXIT:%s\n' "$?"
  printf 'command cargo +stable build --all-targets --all-features --locked\n'
  cargo +stable build --all-targets --all-features --locked
  all_status=$?
  printf 'EXIT:%s\n' "$all_status"
  if [ "$all_status" -ne 0 ]; then
    printf 'ira needs IRA_LIB_DIR and libira, which are absent\n'
    printf 'lammps needs LAMMPS_LIB_DIR and liblammps, which are absent\n'
    printf 'gpr needs GPR_OPTIM_LIB_DIR and libgpr_optim, which are absent\n'
    printf 'command cargo +stable build --all-targets --features %s --locked\n' "$others"
    cargo +stable build --all-targets --features "$others" --locked
    printf 'EXIT:%s\n' "$?"
  fi
  printf 'command cargo +stable bench --no-run --locked\n'
  cargo +stable bench --no-run --locked
  printf 'EXIT:%s\n' "$?"
  set -e
} > "$out.raw" 2>&1
python3 - "$out.raw" "$out" "$root" << 'PY'
import sys
from pathlib import Path
src, dest, root = sys.argv[1:]
text = Path(src).read_text(errors="replace")
text = text.replace(root, ".")
Path(dest).write_text(text)
Path(src).unlink()
PY
