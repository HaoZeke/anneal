#!/bin/sh
# Copyright (c) anneal developers
# Rebuild the LJ75 example from the source archive.
# The archive is a subset of tree 37535ec1a6e99d47a2afe0752072a33af64dccb3,
# the tree of the unpublished build commit, and it has no commit history.
# That tree has the #56 copies of bias.rs, neighbors.rs and potentials.rs,
# the bench file, the newsfragment and precision-policy.org.
# It lacks the cluster_hopping.rs and tip4p.rs parts and the [[bench]] entry.
# Its rigid_body.rs is a third version.
# The log records the toolchain, the command, the exit code and the sha256.
# logs/lj75-rebuild-log.txt shows sha256
# 6887e36a368f9fa81eded591dcfa22904fa317fe2208b1bc58ba6500c8937118,
# the same digest as the running binary.
set -eu
archive=${1:-paper/jctc/logs/lj75-source-37535ec1.tar.xz}
dest=${2:-lj75-source}
log=${3:-paper/jctc/logs/lj75-rebuild-log.txt}
mkdir -p "$dest" "$(dirname "$log")"
{
  rustc +stable --version
  cargo +stable --version
  printf 'command cargo +stable build --release --locked --features bank-rpc --example lj_cluster_search\n'
  xz -dc "$archive" | tar -x -C "$dest"
  set +e
  (
    cd "$dest"
    cargo +stable build --release --locked --features bank-rpc --example lj_cluster_search
  )
  status=$?
  set -e
  printf 'EXIT:%s\n' "$status"
  sha256sum "$dest/target/release/examples/lj_cluster_search"
} > "$log.raw" 2>&1
python3 - "$log.raw" "$log" "$(pwd)" << 'PY'
import sys
from pathlib import Path
src, dest, root = sys.argv[1:]
text = Path(src).read_text(errors="replace")
text = text.replace(root, ".")
text = text.replace(str(Path.home()), ".")
Path(dest).write_text(text)
Path(src).unlink()
PY
