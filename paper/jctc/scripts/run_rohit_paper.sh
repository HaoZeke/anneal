#!/bin/sh
# Copyright (c) anneal developers
# Run proseguard on the manuscript, the SI and the cover letter.
# The profile file is not in this repo. Pass its path.
# Usage: run_rohit_paper.sh PROFILE OUT
set -eu
profile=${1:?profile path}
out=${2:?receipt path}
expected=654d0de1c2632feb93ddfc2f72a08372ad672a8c7cb4a6ad30cca69b2783ddc6
got=$(sha256sum "$profile" | awk '{print $1}')
if [ "$got" != "$expected" ]; then
  echo "profile sha $got" >&2
  exit 1
fi
tip=$(git rev-parse HEAD)
version=$(proseguard --version)
cmd="proseguard -p <profile> --no-llm paper/jctc/manuscript.tex paper/jctc/si.tex paper/jctc/cover-letter.tex"
{
  printf 'proseguard %s\n' "$version"
  printf 'profile_sha %s\n' "$expected"
  printf 'tip %s\n' "$tip"
  printf 'command %s\n' "$cmd"
  printf 'output\n'
  proseguard -p "$profile" --no-llm \
    paper/jctc/manuscript.tex \
    paper/jctc/si.tex \
    paper/jctc/cover-letter.tex
} >"$out" 2>&1 || true
if grep -q "profile not found" "$out"; then
  echo "profile not found" >&2
  exit 1
fi
