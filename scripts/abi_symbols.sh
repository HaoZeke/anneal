#!/bin/sh
# Print the stable C symbols of librgpot, one per line.
# Neighbour and fast-list names are always exported. Engine and RPC
# names appear only in the builds that include them. A missing baseline
# name fails. An extra name does not: it is an addition.
# comm needs both lists in one collation. The baseline is sorted here, so
# its line order in the file does not matter, and C collation keeps the
# printed list the same on every machine.
set -eu
LC_ALL=C
export LC_ALL
lib=${1:?librgpot.so}
baseline=${2:-}
found=$(mktemp)
want=$(mktemp)
trap 'rm -f "$found" "$want"' EXIT
nm -D --defined-only "$lib" | awk '$3 ~ /^rgpot_(nlist_|fast_)/ {print $3}' | sort -u > "$found"
if [ -z "$baseline" ]; then
  cat "$found"
  exit 0
fi
sort -u "$baseline" > "$want"
missing=$(comm -23 "$want" "$found")
if [ -n "$missing" ]; then
  echo "C ABI lost symbols:" >&2
  echo "$missing" >&2
  exit 1
fi
echo "C ABI keeps $(wc -l < "$want") baseline symbols"
