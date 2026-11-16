#!/usr/bin/env bash
# Copyright (c) anneal developers
# Anneal's Sutton-Chen wall with an rgpot build that is not anneal's
# Cargo.lock revision. RGPOT_SRC is a clean rgpot checkout and RGPOT_REV
# the revision it must be at. First sc_wall.sh with that build, appended to
# bench/metal-cluster-h2h/sc-wall; then anneal alone on the 55- and 80-atom
# sc-n12 systems, seeds 2000-2011 at PAPER_BUDGETS, appended to
# bench/metal-cluster-h2h/sc-wall2/seeds.csv with the rgpot revision. One
# UTS namespace with hostname bench, working directory /work, one job at a
# time on core 0. The GMIN rows are not rerun. The harness must be
# committed, because its HEAD is the harness_revision of every row.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
: "${RGPOT_SRC:?set RGPOT_SRC to an rgpot checkout}"
: "${RGPOT_REV:?set RGPOT_REV to the rgpot revision of RGPOT_SRC}"
[[ $(git -C "$RGPOT_SRC" rev-parse HEAD) == "$RGPOT_REV" ]] || { echo "$RGPOT_SRC is not at $RGPOT_REV" >&2; exit 1; }
LIBDIR=$WORK/rgpot-sc-${RGPOT_REV:0:8}
RUNS=$WORK/runs-sc-wall2
LOGS=$WORK/logs-sc-wall2

harness=(scripts/metal_h2h scripts/sc96_78_common.py scripts/sc96_78_hits.py)
if ! git -C "$ROOT" diff --quiet HEAD -- "${harness[@]}" ||
  [[ -n $(git -C "$ROOT" ls-files --others --exclude-standard -- "${harness[@]}") ]]; then
  echo "commit the harness before the run" >&2
  exit 1
fi

RGPOT_SRC=$RGPOT_SRC bash "$HERE/sc_wall.sh"
mkdir -p "$RUNS" "$LOGS"

cat > "$WORK/bin/inside-sc-wall2.sh" << INNER
#!/bin/bash
set -eu
export HOME=$WORK/home TMPDIR=$WORK/tmp PATH=$WORK/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 RGPOT_SC_LIB=$LIBDIR/librgpot.so
cd /work
args=()
for system in sc126-55 sc108-55 sc96-55 sc126-80 sc108-80 sc96-80; do args+=(--panel-system "\$system"); done
exec /usr/bin/python3 -u $HERE/run_sc_anneal.py \\
  --binary $WORK/bin/h2h_sc \\
  --runs $RUNS \\
  --logs $LOGS \\
  --panel-seeds $ROOT/bench/metal-cluster-h2h/sc-wall2/seeds.csv \\
  --rgpot-revision \$(cat $LIBDIR/rgpot-revision) \\
  --head-file $WORK/anneal-head \\
  --no-n78 --no-check \\
  "\${args[@]}"
INNER
chmod +x "$WORK/bin/inside-sc-wall2.sh"

exec sudo -n unshare --uts sh -c 'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-sc-wall2.sh'
