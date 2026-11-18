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
# Shared libraries may not add C++ names in an internal namespace. Those
# namespaces are rgpot::detail and every rgpot namespace named detail or
# detail_fill (nlist, simd, abi, ipi, aoti_unpack, and the same kind of
# helper elsewhere under rgpot). Public component namespaces are the C++
# API and are not part of this check. pocketfft::detail belongs to that
# third party header, so this check leaves it out. The qualified names
# already exported are recorded in abi_detail_baseline.txt, without
# parameter lists: a libstdc++ and a libc++ demangle of the same function
# do not spell std::string the same way, and the macOS build would
# otherwise report those signatures as additions. A new overload of a
# name already in the baseline is not distinguished. A weak out of line
# copy of an inline (nm type W or V) is omitted: a debug build emits it
# and a release build does not, so it is not a stable addition.
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
symbols=$(mktemp)
detail=$(mktemp)
detail_want=$(mktemp)
typed=$(mktemp)
trap 'rm -f "$found" "$want" "$symbols" "$detail" "$detail_want" "$typed"' EXIT
# Mach-O nm has no -D and prefixes C names with an underscore. A static
# archive (lib*.a, or *.lib from an MSVC-style toolchain) has no dynamic
# symbol table, so nm -D finds nothing in it; its defined external
# symbols are read instead. A shared object keeps nm -D, so hidden
# symbols stay out.
if [ "$(uname -s)" = Darwin ]; then
  nm -gU "$lib" | awk 'NF == 3 {sub(/^_/, "", $3); print $2, $3}' > "$typed"
else
  case $lib in
    *.a | *.lib) nm -g --defined-only "$lib" ;;
    *) nm -D --defined-only "$lib" ;;
  esac | awk 'NF == 3 {print $2, $3}' > "$typed"
fi
awk '{print $2}' "$typed" > "$symbols"
case $lib in
  *.a | *.lib) ;;
  *)
    # W and V are weak defined symbols. Inline helpers show up that way
    # when the compiler emits an out-of-line copy, which depends on the
    # optimisation level.
    awk '$1 != "W" && $1 != "V" {print $2}' "$typed" | c++filt | awk '
      function qual(s, p) {
        gsub(/\[abi:cxx11\]/, "", s)
        p = index(s, "(")
        if (p > 0) return substr(s, 1, p - 1)
        return s
      }
      {
        name = qual($0)
        if (index(name, "rgpot::") != 1) next
        if (index(name, "::detail::") || index(name, "::detail_fill::"))
          print name
      }
    ' | sort -u > "$detail"
    sort -u "$(dirname "$0")/abi_detail_baseline.txt" > "$detail_want"
    added=$(comm -13 "$detail_want" "$detail")
    if [ -n "$added" ]; then
      echo "C++ detail symbols added:" >&2
      echo "$added" >&2
      exit 1
    fi
    ;;
esac
awk -v pat="$pattern" '$0 ~ pat' "$symbols" | sort -u > "$found"
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
