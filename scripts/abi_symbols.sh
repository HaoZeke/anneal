#!/bin/sh
# Print the stable C symbols of a library, one per line.
# Usage: abi_symbols.sh [-p PATTERN] LIB [BASELINE...]
# The default names are the neighbour and fast-list exports of librgpot.
# -p replaces the name pattern; the rgpot-core library is checked with
# '^rgpot_' against abi_core_baseline.txt.
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
pattern='^rgpot_(nlist_|fast_)'
if [ "${1:-}" = -p ]; then
  pattern=${2:?pattern}
  shift 2
fi
lib=${1:?librgpot.so}
shift
found=$(mktemp)
want=$(mktemp)
trap 'rm -f "$found" "$want"' EXIT
# Mach-O nm has no -D and prefixes C names with an underscore. A static
# archive (lib*.a, or *.lib from an MSVC-style toolchain) has no dynamic
# symbol table, so nm -D finds nothing in it; its defined external
# symbols are read instead. A shared object keeps nm -D, so hidden
# symbols stay out.
if [ "$(uname -s)" = Darwin ]; then
  nm -gU "$lib" | awk 'NF == 3 {sub(/^_/, "", $3); print $3}'
else
  case $lib in
    *.a | *.lib) nm -g --defined-only "$lib" ;;
    *) nm -D --defined-only "$lib" ;;
  esac | awk 'NF == 3 {print $3}'
fi | awk -v pat="$pattern" '$0 ~ pat' | sort -u > "$found"
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
