#!/usr/bin/env bash
# Copyright (c) anneal developers
# First-parent bisect, then the side of the merge when that commit has two parents.
# Usage: run_bisect.sh GOOD BAD SEED EXPECTED WORKTREE TARGET
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: run_bisect.sh GOOD BAD SEED EXPECTED WORKTREE TARGET" >&2
  exit 2
fi

GOOD=$1
BAD=$2
SEED=$3
EXPECTED=$4
WORK=$5
TARGET=$6
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$HERE/bisect_prefix.py

if [[ ! -d $WORK/.git && ! -f $WORK/.git ]]; then
  git worktree add --detach "$WORK" "$BAD"
fi

prefix_cmd=(
  python3 "$SCRIPT"
  --worktree "$WORK"
  --target "$TARGET"
  --expected "$EXPECTED"
  --seed "$SEED"
  --system m38r14
  --budget 400000
  --cap 11.142690
)

report_bad() {
  local stage=$1
  if ! git bisect log | grep -q 'first bad commit'; then
    echo "$stage did not finish" >&2
    git bisect log >&2 || true
    exit 1
  fi
  echo "stage $stage"
  git rev-parse HEAD
  git log -1 --format='%s'
}

cd "$WORK"
git bisect start --first-parent "$BAD" "$GOOD"
set +e
git bisect run "${prefix_cmd[@]}"
set -e
report_bad first-parent
set +e
"${prefix_cmd[@]}"
step_status=$?
set -e
if [[ $step_status -ne 1 ]]; then
  echo "prefix comparison on the first bad commit exited $step_status" >&2
  exit 1
fi

mapfile -t parents < <(git rev-parse HEAD^@)
if [[ ${#parents[@]} -lt 2 ]]; then
  echo "single-parent"
  exit 0
fi

merge=$(git rev-parse HEAD)
first=${parents[0]}
echo "into-merge $merge"
git bisect reset
git bisect start "$merge" "$first"
set +e
git bisect run "${prefix_cmd[@]}"
set -e
report_bad merge
set +e
"${prefix_cmd[@]}"
step_status=$?
set -e
if [[ $step_status -ne 1 ]]; then
  echo "prefix comparison inside the merge exited $step_status" >&2
  exit 1
fi
exit 0
