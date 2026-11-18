#!/bin/sh
# mpirun -n 4 calculator_group_test.
# share: rank 3's buffer matches the values rank 0 filled.
# bad-owner: an owner outside the calculator count aborts every rank.
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: calculator_group_mpi.sh EXE MODE" >&2
  exit 2
fi

EXE=$1
MODE=$2

if command -v mpirun >/dev/null 2>&1; then
  LAUNCH=mpirun
elif command -v mpiexec >/dev/null 2>&1; then
  LAUNCH=mpiexec
else
  echo "mpirun or mpiexec is required" >&2
  exit 1
fi

help=$("$LAUNCH" --help 2>&1 || true)
ver=$("$LAUNCH" --version 2>&1 || true)
OVER=
TAG=
# Open MPI 5 --help is a short usage line. Both flags are in the man page.
if printf '%s\n' "$help" "$ver" | grep -q 'Open MPI'; then
  OVER=--oversubscribe
  TAG=--tag-output
else
  if printf '%s\n' "$help" | grep -q -- '--oversubscribe'; then
    OVER=--oversubscribe
  fi
  if printf '%s\n' "$help" | grep -q -- '--tag-output'; then
    TAG=--tag-output
  fi
fi

run_bounded() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 30 "$@"
  else
    "$@"
  fi
}

log=$(mktemp)
trap 'rm -f "$log"' EXIT

if [ "$MODE" = "share" ]; then
  unset RGPOT_CALCULATOR_BAD_OWNER || true
  set +e
  # shellcheck disable=SC2086
  run_bounded "$LAUNCH" -n 4 $OVER "$EXE" --order decl >"$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "calculator share status $rc" >&2
    cat "$log" >&2
    exit 1
  fi
  if ! grep -q 'calculator-share rank 3 1.000000 -2.000000 3.500000 0.250000' "$log"; then
    echo "rank 3 buffer does not match rank 0" >&2
    cat "$log" >&2
    exit 1
  fi
  exit 0
fi

if [ "$MODE" = "bad-owner" ]; then
  export RGPOT_CALCULATOR_BAD_OWNER=1
  set +e
  # shellcheck disable=SC2086
  run_bounded "$LAUNCH" -n 4 $OVER $TAG "$EXE" --order decl >"$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 124 ]; then
    echo "bad owner status $rc" >&2
    cat "$log" >&2
    exit 1
  fi
  if [ -n "$TAG" ]; then
    for rank in 0 1 2 3; do
      if ! grep -E -q "[0-9]+,${rank}>.*shareFromCalculator owner out of range|\\[${rank}\\].*shareFromCalculator owner out of range|,${rank}\\].*shareFromCalculator owner out of range" "$log"; then
        echo "rank $rank did not print the owner error" >&2
        cat "$log" >&2
        exit 1
      fi
    done
    exit 0
  fi
  if ! grep -q 'shareFromCalculator owner out of range' "$log"; then
    echo "missing owner error" >&2
    cat "$log" >&2
    exit 1
  fi
  exit 0
fi

echo "unknown mode $MODE" >&2
exit 2
