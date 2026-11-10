#!/bin/sh
# Copyright (c) anneal developers
# Rebuild the LJ75 example from the source archive.
# The archive is a subset of tree 37535ec1a6e99d47a2afe0752072a33af64dccb3,
# the tree of the unpublished build commit, and it has no commit history.
# That tree has the #56 copies of bias.rs, neighbors.rs and potentials.rs,
# the bench file, the newsfragment and precision-policy.org.
# It lacks the cluster_hopping.rs and tip4p.rs parts and the [[bench]] entry.
# Its rigid_body.rs is a third version.
# The log records the time, the archive sha, the toolchain, the command,
# the exit code and the sha256.
# redact_paths.py writes the redacted log and the sha map.
# With no raw path, mktemp creates the raw log outside the work tree.
# logs/lj75-rebuild-log.txt shows sha256
# 6887e36a368f9fa81eded591dcfa22904fa317fe2208b1bc58ba6500c8937118,
# the same digest as the running binary.
set -eu
archive=${1:-paper/jctc/logs/lj75-source-37535ec1.tar.xz}
dest=${2:-lj75-source}
log=${3:-paper/jctc/logs/lj75-rebuild-log.txt}
if [ $# -ge 4 ] && [ -n "$4" ]; then
  raw=$4
else
  raw=$(mktemp)
fi
printf '%s\n' "$raw" >&2
map=${5:-$(dirname "$log")/lj75-rebuild-log-map.txt}
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
mkdir -p "$dest" "$(dirname "$log")" "$(dirname "$raw")"
{
  printf 'UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'archive_sha256 %s\n' "$(sha256sum "$archive" | awk '{print $1}')"
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
} > "$raw" 2>&1
python3 "$here/redact_paths.py" "$raw" "$log" "$map" "$(pwd)" "$here"
