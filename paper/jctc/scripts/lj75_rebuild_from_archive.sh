#!/bin/sh
# Rebuild the LJ75 example from the source archive.
# The archive is the tree 37535ec1a6e99d47a2afe0752072a33af64dccb3
# and contains no commit history.
# That tree does not contain main's #56 hot-path edits.
set -eu
archive=${1:-paper/jctc/logs/lj75-source-37535ec1.tar.xz}
dest=${2:-lj75-source}
mkdir -p "$dest"
xz -dc "$archive" | tar -x -C "$dest"
(
  cd "$dest"
  cargo build --release --locked --example lj_cluster_search
)
