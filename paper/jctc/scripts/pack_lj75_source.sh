#!/bin/sh
# Copyright (c) anneal developers
# Pack the LJ75 source subset.
# Usage: pack_lj75_source.sh TREE-ISH OUT
# git archive, then xz -9e -T1.
# Pass the commit whose tree is
# 37535ec1a6e99d47a2afe0752072a33af64dccb3.
# That command reproduces the shipped archive.
# A bare tree id omits the commit comment, so the bytes differ.
set -eu
treeish=${1:?tree-ish}
out=${2:?out}
kind=$(git cat-file -t "$treeish")
if [ "$kind" = tree ]; then
  commit=6146da08dbab883c21a5607bfa1da7d4677a8008
  if ! git cat-file -e "$commit" 2>/dev/null; then
    echo "the commit of this tree is not present" >&2
    exit 1
  fi
  got=$(git rev-parse "${commit}^{tree}")
  if [ "$got" != "$treeish" ]; then
    echo "that commit has a different tree" >&2
    exit 1
  fi
  treeish=$commit
fi
git archive --format=tar "$treeish" -- \
  Cargo.toml Cargo.lock build.rs cbindgen.toml LICENSE.txt \
  src examples benches vendor/nng vendor/nng-sys include \
  | xz -9e -T1 > "$out"
