#!/usr/bin/env bash
# Copyright (c) anneal developers
# Wales's public GMIN with the evaluator cap and a Stillinger-Weber timer.
# The public build brackets LJ and TIP4P only. SWTWO is timed the way the
# Morse bench times MORSEGH, so pot_s is the time inside the potential.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${GMIN_H2H_WORK:-/work/gmin-h2h}
GMIN=$WORK/src/wales/GMIN
PIN=$ROOT/scripts/gmin_pin.py
eval "$(python3 "$PIN" shell)"

GMIN_H2H_WORK=$WORK bash "$ROOT/scripts/gmin_h2h/build_gmin.sh" \
  --extra-patch "$HERE/sw-timer.patch" "$@"
python3 "$PIN" read-svnrev "$GMIN/source/SVNREV" >/dev/null
grep -q H2H_POT_ENTER "$GMIN/source/SW.f90"
