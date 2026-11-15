#!/bin/sh
# Print the stable C symbols of librgpot, one per line.
# Usage: abi_symbols.sh LIB [BASELINE...]
# Each baseline lists names the library must export. The build passes the
# fast-list baseline only when it links the fast list. Engine and RPC
# names appear only in the builds that include them. A missing baseline
# name fails. An extra name does not: it is an addition.
# comm needs both lists in one collation. The baselines are sorted here, so
# their line order in the files does not matter, and C collation keeps the
# printed list the same on every machine.
set -eu
LC_ALL=C
export LC_ALL
lib=${1:?librgpot.so}
shift
found=$(mktemp)
want=$(mktemp)
trap 'rm -f "$found" "$want"' EXIT
# Mach-O nm has no -D and prefixes C names with an underscore. An archive
# has no dynamic symbol table, so its external symbols are read instead.
if [ "$(uname -s)" = Darwin ]; then
  nm -gU "$lib" | awk 'NF == 3 {sub(/^_/, "", $3); print $3}'
elif [ "${lib%.a}" != "$lib" ]; then
  nm -g --defined-only "$lib" | awk 'NF == 3 {print $3}'
else
  nm -D --defined-only "$lib" | awk 'NF == 3 {print $3}'
fi | awk '/^rgpot_(nlist_|fast_)/' | sort -u > "$found"
if [ "$#" -eq 0 ]; then
  cat "$found"
  exit 0
fi
cat "$@" | sort -u > "$want"
missing=$(comm -23 "$want" "$found")
if [ -n "$missing" ]; then
  echo "C ABI lost symbols:" >&2
  echo "$missing" >&2
  exit 1
fi
echo "C ABI keeps $(wc -l < "$want") baseline symbols"
