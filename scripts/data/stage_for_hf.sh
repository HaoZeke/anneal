#!/bin/sh
# Copyright (c) anneal developers
# Copy the files named in data/hf-files.csv into a staging directory.
# The staging layout is the Hugging Face dataset layout for
# rgoswami/anneal-paper-data (repo type dataset). Each path is
# relative to the anneal repository root.
# Upload that directory from another machine. This script does not
# authenticate.
#
# Usage: scripts/data/stage_for_hf.sh [STAGING_DIR]
# The default staging directory is build/hf-stage.

set -eu
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)
staging=${1:-$root/build/hf-stage}
manifest=$root/data/hf-files.csv
if [ ! -f "$manifest" ]; then
  printf 'missing %s\n' "$manifest" >&2
  exit 1
fi
mkdir -p "$staging"
count=0
{
  IFS= read -r _header
  while IFS=, read -r path size sha
  do
    [ -n "$path" ] || continue
    src=$root/$path
    dest=$staging/$path
    if [ ! -f "$src" ]; then
      printf 'missing %s\n' "$src" >&2
      exit 1
    fi
    got=$(wc -c < "$src" | tr -d ' ')
    if [ "$got" != "$size" ]; then
      printf 'size mismatch %s got %s manifest %s\n' "$path" "$got" "$size" >&2
      exit 1
    fi
    sum=$(sha256sum "$src" | awk '{print $1}')
    if [ "$sum" != "$sha" ]; then
      printf 'sha256 mismatch %s\n' "$path" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$dest")"
    cp -p "$src" "$dest"
    count=$((count + 1))
  done
} < "$manifest"
cat > "$staging/README.md" << 'EOF'
# anneal-paper-data

Dataset repository: rgoswami/anneal-paper-data (repo type dataset).
Paper dumps for HaoZeke/anneal. Paths are relative to the anneal
repository root and match `data/hf-files.csv` in that repository
(`path,size,sha256`). The manifest itself stays in git.

The dataset holds minima dumps, landscape dumps, xyz dumps, call
traces, and raw xtb, CREST, and seed logs. Small summary CSVs, test
fixtures, rendered figures, and source stay in the git repository.
EOF
printf 'staged %s files in %s\n' "$count" "$staging" >&2
