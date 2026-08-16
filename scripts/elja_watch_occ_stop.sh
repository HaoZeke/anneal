#!/usr/bin/env bash
# One-shot occupancy harvest for anneal-stop ensembles.
set -euo pipefail
echo "=== jobs ==="
squeue -u rog32 -h -o "%i %P %j %t %M %R" | grep -E "lj(38|75|98)-occ" || echo none
for spec in \
  "lj38 /users/home/rog32/ljwork/jcc/lj38-occ-sb/lj38/shared/lj38-shared-0001 -173.928427" \
  "lj75 /users/home/rog32/ljwork/jcc/lj75-occ-sb/lj75/shared/lj75-shared-0004 -397.492331" \
  "lj98 /users/home/rog32/ljwork/jcc/lj98-occ-sb/lj98/shared/lj98-shared-0004 -543.665361"
do
  set -- $spec
  label=$1
  root=$2
  gm=$3
  echo "=== $label ==="
  if [[ ! -d $root/workers ]]; then
    echo missing
    continue
  fi
  echo "workers=$(ls -1 "$root/workers"/*.out 2>/dev/null | wc -l)"
  echo "personal=$(grep -h "personal best" "$root/workers"/*.out 2>/dev/null | wc -l)"
  echo "score=$(grep -h "score $gm" "$root/workers"/*.out 2>/dev/null | wc -l)"
  echo "done=$(grep -h "^  done " "$root/workers"/*.out 2>/dev/null | wc -l)"
  echo "best=$(grep -h "best $gm" "$root/workers"/*.out 2>/dev/null | wc -l)"
  grep -h "personal best\|score \|done \|seed .*: best " "$root/workers"/*.out 2>/dev/null | tail -8 || true
done
