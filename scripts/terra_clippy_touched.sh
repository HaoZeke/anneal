#!/usr/bin/env bash
# Clippy -D warnings, failing only on diagnostics in the named files.
set -euo pipefail

if (($# < 2)); then
  echo "usage: $0 CARGO_ARGS... -- FILE [FILE...]" >&2
  exit 2
fi

cargo_args=()
files=()
seen_sep=0
for arg in "$@"; do
  if [[ $arg == -- ]]; then
    seen_sep=1
    continue
  fi
  if [[ $seen_sep -eq 1 ]]; then
    files+=("$arg")
  else
    cargo_args+=("$arg")
  fi
done
if [[ $seen_sep -eq 0 || ${#files[@]} -eq 0 ]]; then
  echo "usage: $0 CARGO_ARGS... -- FILE [FILE...]" >&2
  exit 2
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
set +e
cargo clippy "${cargo_args[@]}" -- -D warnings >"$log" 2>&1
status=$?
set -e
cat "$log"
if [[ $status -eq 0 ]]; then
  echo "CLIPPY_TOUCHED_CLEAN files=${files[*]}"
  exit 0
fi

pattern=$(printf '%s|' "${files[@]}")
pattern=${pattern%|}
if rg -n -- "--> (${pattern})" "$log" | rg -q .; then
  echo "clippy diagnostics in touched files:" >&2
  rg -n -C 3 -- "--> (${pattern})" "$log" >&2 || true
  exit 1
fi
echo "CLIPPY_TOUCHED_CLEAN files=${files[*]} (crate had unrelated diagnostics)"
exit 0
